"""
The run store kept in an object store (a bucket), so that a machine without the store (the deployed dashboard) can fetch it.

What is kept there, relative to the project root: the folder store/ and the MLflow index mlruns.db. The bucket is an S3 API bucket (Cloudflare R2, Backblaze B2,
Storj, AWS ...) or, for a mounted disk or a test, a plain folder. What is in it is read from the bucket itself (its listing: every file with its size); there is
no index file to keep up to date.

    upload(root, store)     put what is new or has another size in the bucket
    sync(root, store)       fetch what is missing on this machine, except the files the caller names on demand; a file that is already there is never overwritten
                            (the local store may be newer than the bucket)
    OnDemand                fetches those big files one by one when they are first read (ensure / ensure_folder / exists, called by the readers)

Neither deletes anything. A file that was changed to exactly the same size is not noticed by upload (the runs of the store are written once, so this does not
happen in practice). The remote is named by one string, see open_store: "s3://<bucket>/<prefix>" or the path of a folder. The progress callbacks are called
from worker threads (one at a time): a caller that draws must only record the numbers there.
"""

from __future__ import annotations

import os
import shutil
import threading
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

KEPT = ("store", "mlruns.db")               # what the bucket holds, relative to the project root
IGNORED_NAMES = {".DS_Store"}
S3_SETTINGS = ("S3_ENDPOINT_URL", "S3_ACCESS_KEY_ID", "S3_SECRET_ACCESS_KEY", "S3_REGION")    # what an s3:// remote needs besides its name

# the bucket of this project: not secrets (the keys are), so they are written here and not in an environment. The deployed dashboard and upload_store.py use them.
PROJECT_REMOTE = "s3://varprojectbucket/MLProject-standalone"
PROJECT_ENDPOINT = "https://s3.us-east-005.backblazeb2.com"
PROJECT_REGION = "us-east-005"
PROJECT_KEYS = ("S3_ACCESS_KEY_ID", "S3_SECRET_ACCESS_KEY")       # the secrets of the bucket: read only for the dashboard, read and write for upload_store.py


class ObjectStore(Protocol):
    def get(self, key: str, target: Path) -> None: ...
    def put(self, key: str, source: Path) -> None: ...
    def list(self) -> dict[str, int]: ...                   # every file under the store's folder / prefix: {key relative to it: size in bytes}


class DirStore:
    """A folder as a bucket (a mounted disk, or a test)."""

    def __init__(self, folder: Path):
        self.folder = Path(folder)

    def get(self, key: str, target: Path) -> None:
        shutil.copyfile(self.folder / key, target)

    def put(self, key: str, source: Path) -> None:
        destination = self.folder / key
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)

    def list(self) -> dict[str, int]:
        return {p.relative_to(self.folder).as_posix(): p.stat().st_size for p in self.folder.rglob("*") if p.is_file()} if self.folder.is_dir() else {}


class S3Store:
    """An S3 API bucket under a prefix (boto3 is imported here, so that nothing else needs it)."""

    def __init__(self, bucket: str, prefix: str, endpoint_url: str, access_key_id: str, secret_access_key: str, region: str):
        import boto3
        self.bucket, self.prefix = bucket, prefix.strip("/")
        self.client = boto3.client(
            "s3", endpoint_url=endpoint_url, aws_access_key_id=access_key_id, aws_secret_access_key=secret_access_key, region_name=region,
        )

    def _key(self, key: str) -> str:
        return f"{self.prefix}/{key}" if self.prefix else key

    def get(self, key: str, target: Path) -> None:
        self.client.download_file(self.bucket, self._key(key), str(target))

    def put(self, key: str, source: Path) -> None:
        self.client.upload_file(str(source), self.bucket, self._key(key))

    def list(self) -> dict[str, int]:
        start = f"{self.prefix}/" if self.prefix else ""
        found: dict[str, int] = {}
        for page in self.client.get_paginator("list_objects_v2").paginate(Bucket=self.bucket, Prefix=start):
            for entry in page.get("Contents", []):
                if not entry["Key"].endswith("/"):                  # a "folder" placeholder is not a file
                    found[entry["Key"][len(start):]] = entry["Size"]
        return found


def project_settings(secrets: Mapping[str, str | None]) -> dict[str, str | None]:
    """The S3_SETTINGS of this project's bucket: its endpoint and region from above, its keys from secrets (None where they are not given)."""
    return {"S3_ENDPOINT_URL": PROJECT_ENDPOINT, "S3_REGION": PROJECT_REGION, **{name: secrets.get(name) for name in PROJECT_KEYS}}


def open_store(remote: str, settings: Mapping[str, str | None]) -> ObjectStore:
    """
    The object store a remote string names: "s3://<bucket>/<prefix>" (the S3_SETTINGS are read from settings, all of them must be there) or the path of a
    folder. Nothing is assumed: a missing setting is an error that names it.
    """
    if not remote.startswith("s3://"):
        return DirStore(Path(remote))
    bucket, _, prefix = remote.removeprefix("s3://").partition("/")
    if not bucket:
        raise ValueError(f"The remote '{remote}' has no bucket: it must be s3://<bucket>/<prefix>.")
    missing = [name for name in S3_SETTINGS if not settings.get(name)]
    if missing:
        raise ValueError(f"The remote '{remote}' needs these settings, which are not set: {', '.join(missing)}.")
    return S3Store(bucket, prefix, settings["S3_ENDPOINT_URL"], settings["S3_ACCESS_KEY_ID"], settings["S3_SECRET_ACCESS_KEY"], settings["S3_REGION"])


# --------------------------------------------------------------------------------------------------
# what is where
# --------------------------------------------------------------------------------------------------

def local_files(root: Path) -> dict[str, int]:
    """{path relative to root (posix): size} of the files of the project that the bucket holds."""
    root, found = Path(root), {}
    for name in KEPT:
        path = root / name
        if path.is_file():
            found[name] = path.stat().st_size
        elif path.is_dir():
            found.update({p.relative_to(root).as_posix(): p.stat().st_size for p in path.rglob("*") if p.is_file() and p.name not in IGNORED_NAMES})
    return dict(sorted(found.items()))


def remote_files(store: ObjectStore) -> dict[str, int]:
    """{path: size} of the files in the bucket that belong to the store (store/ and mlruns.db; whatever else is under the prefix is not looked at)."""
    return {
        rel: size for rel, size in sorted(store.list().items())
        if rel.split("/")[0] in KEPT and rel.rsplit("/", 1)[-1] not in IGNORED_NAMES
    }


NOTHING_THERE = "The bucket has no store/ or mlruns.db under its prefix: upload the store first (training/upload_store.py)."


# --------------------------------------------------------------------------------------------------
# upload and sync
# --------------------------------------------------------------------------------------------------

@dataclass
class UploadResult:
    uploaded: list[str] = field(default_factory=list)
    unchanged: int = 0
    bytes: int = 0


def upload(root: Path, store: ObjectStore, *, dry_run: bool = False, workers: int = 8, progress: Callable[[int, int], None] | None = None) -> UploadResult:
    """Put the files that are not in the bucket or have another size there. dry_run: only say what would be put."""
    root = Path(root)
    local, remote = local_files(root), remote_files(store)
    todo = [rel for rel, size in local.items() if remote.get(rel) != size]
    result = UploadResult(uploaded=todo, unchanged=len(local) - len(todo), bytes=sum(local[rel] for rel in todo))
    if dry_run or not todo:
        return result
    done, lock = 0, threading.Lock()

    def put(rel: str) -> None:
        nonlocal done
        store.put(rel, root / rel)
        with lock:
            done += 1
            if progress:
                progress(done, len(todo))

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(put, todo))                       # list(): an exception of a put is raised here
    return result


@dataclass
class SyncResult:
    downloaded: list[str] = field(default_factory=list)
    present: int = 0
    differs_from_bucket: list[str] = field(default_factory=list)     # there locally with another size: left as it is
    left_on_demand: int = 0                                           # not fetched: the caller named them on demand (OnDemand fetches them when they are read)
    bytes: int = 0


def _fetch_file(root: Path, store: ObjectStore, rel: str, size: int) -> None:
    """One file of the bucket to its place: to <name>.part, checked against its size in the bucket, then moved into place (no half file under its real name)."""
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    store.get(rel, partial)
    if partial.stat().st_size != size:
        partial.unlink()
        raise OSError(f"{rel} came back from the bucket with another size than the bucket lists ({size} bytes): not kept.")
    os.replace(partial, path)


def sync(
    root: Path, store: ObjectStore, *, is_on_demand: Callable[[str], bool], workers: int = 8, progress: Callable[[int, int, int, int], None] | None = None,
) -> SyncResult:
    """
    Fetch the files of the bucket that are not on this machine, except those that is_on_demand(path relative to root) names: they stay in the bucket until
    OnDemand fetches them (pass `lambda rel: False` to fetch everything). A file that is there is not touched (a local one with another size is only reported).
    progress(files done, files to fetch, bytes done, bytes to fetch).
    """
    root = Path(root)
    files = remote_files(store)
    if not files:
        raise FileNotFoundError(NOTHING_THERE)
    result, todo = SyncResult(), []
    for rel, size in files.items():
        path = root / rel
        if not path.exists():
            if is_on_demand(rel):
                result.left_on_demand += 1
            else:
                todo.append(rel)
        elif path.stat().st_size != size:
            result.differs_from_bucket.append(rel)
        else:
            result.present += 1
    total_bytes, done, done_bytes, lock = sum(files[rel] for rel in todo), 0, 0, threading.Lock()

    def fetch(rel: str) -> None:
        nonlocal done, done_bytes
        _fetch_file(root, store, rel, files[rel])
        with lock:
            done += 1
            done_bytes += files[rel]
            if progress:
                progress(done, len(todo), done_bytes, total_bytes)

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(fetch, todo))
    result.downloaded, result.bytes = todo, total_bytes
    return result


# --------------------------------------------------------------------------------------------------
# on demand: the big files that sync left in the bucket
# --------------------------------------------------------------------------------------------------

class OnDemand:
    """
    Fetches a file of the bucket when it is first read. The readers of those files (walk_forward.LoadedRun, virtual_runs) ask through ensure / ensure_folder /
    exists below, which do nothing until the dashboard has called enable_on_demand: a local store behaves as it always did.

    announce(text) is a context manager the fetch runs inside (the dashboard shows a spinner); the fetches are one at a time (a lock), and a file that another
    visitor's fetch has put in place meanwhile is not fetched again.
    """

    def __init__(self, root: Path, store: ObjectStore, announce: Callable[[str], AbstractContextManager], workers: int = 8):
        self.root, self.store, self.announce, self.workers = Path(root), store, announce, workers
        self.files = remote_files(store)
        if not self.files:
            raise FileNotFoundError(NOTHING_THERE)
        self._lock = threading.Lock()

    def _rel(self, path: Path) -> str | None:
        """The path relative to the project root, as the bucket lists it; None for a path outside the root."""
        try:
            return Path(os.path.abspath(path)).relative_to(os.path.abspath(self.root)).as_posix()
        except ValueError:
            return None

    def has(self, path: Path) -> bool:
        """On this machine or in the bucket."""
        return Path(path).exists() or self._rel(path) in self.files

    def fetch(self, path: Path) -> None:
        """The file, when it is in the bucket and not here; nothing otherwise (the reader then finds what it always finds: the file, or its own error)."""
        rel = self._rel(path)
        if rel in self.files:
            self._fetch_missing([rel], Path(path).name)

    def fetch_folder(self, folder: Path) -> None:
        """Every file of the bucket under this folder that is not here."""
        rel = self._rel(folder)
        if rel is not None:
            self._fetch_missing([name for name in self.files if name.startswith(rel + "/")], Path(folder).name)

    def _fetch_missing(self, rels: list[str], name: str) -> None:
        with self._lock:
            missing = [rel for rel in rels if not (self.root / rel).exists()]
            if not missing:
                return
            megabytes = sum(self.files[rel] for rel in missing) / 1e6
            with self.announce(f"Loading {name} from the store ({len(missing)} file{'s' if len(missing) > 1 else ''}, {megabytes:,.0f} MB) ..."):
                with ThreadPoolExecutor(self.workers) as pool:
                    list(pool.map(lambda rel: _fetch_file(self.root, self.store, rel, self.files[rel]), missing))


_on_demand: OnDemand | None = None


def enable_on_demand(on_demand: OnDemand) -> None:
    """From now on ensure / ensure_folder / exists fetch from this OnDemand's bucket (once per process, after the sync)."""
    global _on_demand
    _on_demand = on_demand


def ensure(path: Path) -> None:
    """Call before reading a file that sync may have left in the bucket: fetches it when it is not here. Nothing without enable_on_demand."""
    if _on_demand is not None:
        _on_demand.fetch(path)


def ensure_folder(folder: Path) -> None:
    """ensure for every file under a folder."""
    if _on_demand is not None:
        _on_demand.fetch_folder(folder)


def exists(path: Path) -> bool:
    """Path.exists(), for a file that sync may have left in the bucket: it exists when it is here or in the bucket (nothing is fetched to say so)."""
    return Path(path).exists() if _on_demand is None else _on_demand.has(path)
