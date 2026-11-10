"""
Where the data of the pages comes from, and who may see it.

STORE_IN_BUCKET below is the switch. False (the machine that has the run store, store/ and mlruns.db): nothing here does anything. True (the deployed dashboard): the
store is kept in the project's bucket (core/remote_store.py, PROJECT_REMOTE) and the password is asked, with no setting that switches it on. Until a visitor gives
the password (the setting DASHBOARD_PASSWORD) the Results and
Experiments pages show the results cache that is in the repository (core/results_cache.py, results_cache/): the saved experiments at their reading, no bucket. The
password is asked where more than that is wanted: the other pages, another VaR reading, the per ticker variances. The password only saves the bucket's bandwidth,
it does not hide anything. After it the store is fetched once for the whole process (files that are there are not fetched again) and the page opens. The big
files (the per ticker banks of a run and the residuals of the saved portfolio runs, together most of the store) are not part of that fetch: they are fetched one
by one when a page first reads them (core.remote_store.OnDemand).

The settings are read from the environment, from the .env file of the project root (not tracked) and from st.secrets (.streamlit/secrets.toml, or the secrets of
the deployment), in this order:

    DASHBOARD_PASSWORD      required when STORE_IN_BUCKET is True (the dashboard does not open its data without one)
    S3_ACCESS_KEY_ID, S3_SECRET_ACCESS_KEY      the project bucket's keys (a read only key is enough); its endpoint and region are in core/remote_store.py
    STORE_REMOTE            optional: another bucket ("s3://<bucket>/<prefix>") or a folder than the project's, for a test

The Approach page needs no data and is open. The password is never written in the code or the repository.
"""

from __future__ import annotations

import hmac
import os
import threading
import time
from pathlib import Path

import streamlit as st
from dotenv import dotenv_values
from streamlit.errors import StreamlitSecretNotFoundError

from core import remote_store as rso
from core import virtual_runs as vr
from core import walk_forward as wf

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / ".env"
STORE_IN_BUCKET = True                  # True: the store is in the project's bucket and the data pages ask for the password (the deployed dashboard). False: it is on this disk
REMOTE_SETTING = "STORE_REMOTE"         # optional: another bucket or a folder than rso.PROJECT_REMOTE (a test); not needed
PASSWORD_SETTING = "DASHBOARD_PASSWORD"
UNLOCKED_KEY = "data_unlocked"          # session state: this visitor gave the password
WRONG_PASSWORD_DELAY = 1.0              # seconds a wrong password costs
PASSWORD_HINT = "Please enter the password in my CV to access."    # shown under every password question
ON_DEMAND_FILES = {wf.EWMA_BANK_FILE, wf.FIRST_WINDOW_TRAINING_BANK_FILE}      # the two big banks of a run: read by the per ticker tab and the re-weighting pages only
ON_DEMAND_FOLDER = f"/{vr.VIRTUAL_RUNS_DIR}/{vr.RESIDUALS_DATASET}/"           # the residuals of the saved portfolio runs: read for the model that is shown only


def is_on_demand(rel: str) -> bool:
    """True for a file of the store that the fetch after the password leaves in the bucket (core.remote_store.OnDemand fetches it when it is read)."""
    return rel.rsplit("/", 1)[-1] in ON_DEMAND_FILES or ON_DEMAND_FOLDER in f"/{rel}"


def setting(name: str) -> str | None:
    """
    A setting of the deployment: the environment first, then the .env file of the project root, then st.secrets; None when it is in none of them. The .env file
    is only asked for the setting that is wanted (dotenv_values does not put it into the environment): its other lines, the Alpaca keys, stay out of this process.
    """
    if os.environ.get(name):
        return os.environ[name]
    from_file = dotenv_values(ENV_FILE).get(name) if ENV_FILE.exists() else None
    if from_file:
        return from_file
    try:
        value = st.secrets.get(name)
    except StreamlitSecretNotFoundError:
        return None
    return None if value is None else str(value)


def _all_settings() -> dict[str, str | None]:
    """What open_store needs: the project bucket's endpoint and region (written in core/remote_store.py) and its keys (a setting)."""
    return rso.project_settings({name: setting(name) for name in rso.PROJECT_KEYS})


def remote() -> str | None:
    """The bucket the store is fetched from; None when STORE_IN_BUCKET is False (the store is on this disk and nothing is asked)."""
    return (setting(REMOTE_SETTING) or rso.PROJECT_REMOTE) if STORE_IN_BUCKET else None


def password_is_right(entered: str, expected: str) -> bool:
    return hmac.compare_digest(entered.encode("utf-8"), expected.encode("utf-8"))


class _Fetch:
    """
    The fetch of the store, once per process, by ONE worker thread. A visitor's script is stopped and started again whenever a widget changes (and a second visitor has
    a script of their own): each of them attaches to the worker that is running and draws its progress, none starts another. A thread cannot be stopped from outside,
    so a worker that a stopped script had started goes on, and a new one beside it would fetch the same files at the same time.
    """

    def __init__(self) -> None:
        self.lock = threading.Lock()            # only for starting the worker
        self.done = False
        self.worker: threading.Thread | None = None
        self.numbers = [0, 0, 0, 0]             # files done, files to fetch, bytes done, bytes to fetch: written by the worker, drawn by a script thread
        self.failure: list[Exception] = []


@st.cache_resource
def _fetch_state() -> _Fetch:
    return _Fetch()


def is_locked() -> bool:
    """True when the store is in a bucket and this visitor has not given the password: the pages then show the results cache (results_cache/) only."""
    return remote() is not None and not st.session_state.get(UNLOCKED_KEY)


def ask_password(key: str, intro: str | None) -> None:
    """
    The password form, where it is called (a page, or inside a tab: the rest of the script goes on). The right password opens the data for this visitor and reruns the
    script, which then fetches the store (require_data). key: the form's key (two forms of one script need two keys). intro: a line above the form, None for none.
    """
    expected = setting(PASSWORD_SETTING)
    if not expected:
        st.error(f"{PASSWORD_SETTING} is not set: the dashboard does not open its data from the bucket without a password.")
        return
    if intro is not None:
        st.write(intro)
    st.write(PASSWORD_HINT)
    with st.form(key):
        entered = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Open")
    if submitted:
        if password_is_right(entered, expected):
            st.session_state[UNLOCKED_KEY] = True
            st.rerun()
        time.sleep(WRONG_PASSWORD_DELAY)
        st.error("Wrong password.")


def _fetch_store(root: Path, bucket: str) -> None:
    state = _fetch_state()
    if state.done:
        return
    bar = st.progress(0.0, text="Connecting to the store ...")
    with state.lock:
        if state.worker is None or (not state.worker.is_alive() and state.failure):      # not started yet, or the last try failed: start it (again)
            state.failure.clear()
            state.numbers[:] = [0, 0, 0, 0]
            settings_now = _all_settings()      # read here, in the script thread: st.secrets is not read from another one
            state.worker = threading.Thread(target=_run_fetch, args=(state, root, bucket, settings_now), daemon=True)
            state.worker.start()
        worker = state.worker
    while worker.is_alive():                    # only a script thread may draw
        done, files, done_bytes, total_bytes = state.numbers
        bar.progress(done_bytes / total_bytes if total_bytes else 0.0, text=f"Loading the data: {done}/{files} files, {done_bytes / 1e6:,.0f} of {total_bytes / 1e6:,.0f} MB")
        time.sleep(0.2)
    worker.join()
    bar.empty()
    if state.failure:
        st.error(f"The data could not be loaded from the store: {state.failure[0]}")
        st.stop()


def _run_fetch(state: _Fetch, root: Path, bucket: str, settings: dict[str, str | None]) -> None:
    """The fetch itself, in a thread of its own: it touches no Streamlit call (the settings are read before it starts), only the numbers and the result it reports."""
    try:
        store = rso.open_store(bucket, settings)
        rso.sync(root, store, is_on_demand=is_on_demand, progress=lambda *counted: state.numbers.__setitem__(slice(None), counted))
        rso.enable_on_demand(rso.OnDemand(root, store, announce=st.spinner))      # the spinner is drawn when a page reads a big file, in the script thread
        state.done = True
    except Exception as error:      # whatever it is (no keys, an empty bucket, a file that fails its size check, the bucket not reachable): the page shows it, a thread's own would be lost
        state.failure.append(error)


def require_data(root: Path) -> None:
    """
    Call it first in every page that reads the run store. STORE_IN_BUCKET False: nothing. Otherwise: the password form until the visitor has given the
    password, then the store is fetched once; the page goes on when this returns (st.stop() ends the script here while the form is shown).
    """
    bucket = remote()
    if bucket is None:
        return
    if not setting(PASSWORD_SETTING):
        ask_password("unlock", None)                # says that there is no password configured
        st.stop()
    if not st.session_state.get(UNLOCKED_KEY):
        st.title("🔒 Password")
        ask_password("unlock", None)
        st.stop()
    _fetch_store(root, bucket)
