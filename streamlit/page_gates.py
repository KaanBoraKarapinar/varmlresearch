"""
Which data a page reads, and when the password is asked (see data_access.py): the two kinds of page of the dashboard.

    data_page(render, settings)                                   a page that needs the whole run store
    cached_page(render, settings, cache_settings, any_reading)    a page that the results cache can show

Both return the function that st.Page runs. render(settings) draws the page with the settings it is given. Nothing changes when the run store is local.
"""

from __future__ import annotations

from collections.abc import Callable

import dashboard_data as dd
import data_access
from core import results_cache as rc


def data_page(render: Callable[[dd.Settings], None], settings: dd.Settings) -> Callable[[], None]:
    """A page that reads the run store: when the store is in a bucket the password comes first and then the fetch (data_access.require_data)."""
    def page() -> None:
        data_access.require_data(dd.ROOT)
        render(settings)
    return page


def cached_page(render: Callable[[dd.Settings], None], settings: dd.Settings, cache_settings: dd.Settings, *, any_reading: bool) -> Callable[[], None]:
    """
    A page that the results cache can show: until the visitor has given the password (only when the store is in a bucket) it reads the cache, which holds the saved
    experiments at their reading. Another VaR reading in the sidebar needs the store, so the password is asked for then, unless the page does not use the reading
    (any_reading). After the password, or with a local store, the page reads the whole store like the others.
    """
    def page() -> None:
        if data_access.is_locked():
            if not any_reading and settings.reading != rc.CACHED_READING:
                data_access.require_data(dd.ROOT)              # asks for the password and stops the page
            render(cache_settings)
        else:
            data_access.require_data(dd.ROOT)
            render(settings)
    return page
