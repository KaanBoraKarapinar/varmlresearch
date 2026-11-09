"""
Put the run store (store/ and mlruns.db) into the bucket the dashboard fetches it from (core/remote_store.py).

    python training/upload_store.py --dry-run      what would be put (nothing is sent)
    python training/upload_store.py                put what is new or changed, then the manifest

The remote is --remote or the setting STORE_REMOTE: "s3://<bucket>/<prefix>" (then S3_ENDPOINT_URL, S3_ACCESS_KEY_ID, S3_SECRET_ACCESS_KEY and S3_REGION
must be set) or the path of a folder. The settings are read from the environment and from the .env file of the project root; they are never printed.
Nothing is deleted in the bucket. Start it again after every new run: the dashboard only knows what the manifest lists.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

from core import remote_store as rso


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--remote", help="s3://<bucket>/<prefix> or a folder; default: the setting STORE_REMOTE")
    parser.add_argument("--root", type=Path, default=ROOT, help="the project folder whose store/ and mlruns.db are put (default: this project)")
    parser.add_argument("--dry-run", action="store_true", help="only list what would be put")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    remote = args.remote or os.environ.get("STORE_REMOTE")
    if not remote:
        sys.exit("No remote: give --remote or set STORE_REMOTE (s3://<bucket>/<prefix> or a folder).")
    store = rso.open_store(remote, os.environ)

    def progress(done: int, total: int) -> None:
        print(f"\r  {done}/{total} files", end="", flush=True)

    show_progress = sys.stdout.isatty() and not args.dry_run       # a redrawn line in a terminal; a log file gets only the summary
    result = rso.upload(args.root, store, dry_run=args.dry_run, workers=args.workers, progress=progress if show_progress else None)
    print()
    verb = "would be put" if args.dry_run else "put"
    print(f"{len(result.uploaded)} files {verb} ({result.bytes / 1e6:,.1f} MB), {result.unchanged} unchanged.")
    if args.dry_run:
        for rel in result.uploaded[:20]:
            print("  ", rel)
        if len(result.uploaded) > 20:
            print(f"   ... and {len(result.uploaded) - 20} more")


if __name__ == "__main__":
    main()
