"""
Page: approach. The first page of the dashboard: how the project works (the README.md of the repository, up to the line that ends the approach).

The text is the README, so that the page and the README cannot differ: it is edited there, not here. A markdown image line

    ![alt text](img/name.png)

is shown with st.image (Streamlit's markdown does not load the files of a repository); a figure that is not in the repository yet is named on the page instead of
leaving a broken picture. The first heading of the README is the title of the project and is not repeated.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import streamlit as st

import dashboard_data as dd
import page_results as pr        # its table_frame: a markdown table as a DataFrame

README = dd.ROOT / "README.md"
APPROACH_END = "<!-- approach page ends here"        # the line of the README after which the page stops
IMAGE_LINE = re.compile(r"^!\[(?P<alt>[^\]]*)\]\((?P<path>[^)\s]+)\)\s*$")


@dataclass(frozen=True)
class Block:
    kind: str       # 'text', 'table' or 'image'
    text: str = ""  # the markdown of a text block or of a table, the alt text of an image
    path: str = ""  # an image: its path relative to the project root


def approach_text(readme: str) -> str:
    """The README without its first heading and from the start up to the line that ends the approach."""
    lines = readme.splitlines()
    end = next((i for i, line in enumerate(lines) if line.startswith(APPROACH_END)), None)
    if end is None:
        raise ValueError(f"README.md has no line that starts with '{APPROACH_END}' to end the approach page.")
    body = lines[:end]
    if body and body[0].startswith("# "):
        body = body[1:]
    return "\n".join(body)


def blocks(text: str) -> list[Block]:
    """The text as markdown blocks and images, in order."""
    out, prose = [], []

    def flush() -> None:
        if "\n".join(prose).strip():
            out.append(Block("text", "\n".join(prose)))
        prose.clear()

    lines, i = text.splitlines(), 0
    while i < len(lines):
        image = IMAGE_LINE.match(lines[i].strip())
        if image:
            flush()
            out.append(Block("image", image["alt"], image["path"]))
            i += 1
        elif lines[i].lstrip().startswith("|"):         # a markdown table: shown as a Streamlit table, like on the Results page
            flush()
            end = i
            while end < len(lines) and lines[end].lstrip().startswith("|"):
                end += 1
            out.append(Block("table", "\n".join(lines[i:end])))
            i = end
        else:
            prose.append(lines[i])
            i += 1
    flush()
    return out


def render() -> None:
    st.title("Approach")
    for block in blocks(approach_text(README.read_text(encoding="utf-8"))):
        if block.kind == "text":
            st.markdown(block.text)
        elif block.kind == "table":
            frame = pr.table_frame(block.text)
            st.dataframe(frame, hide_index=True, width="stretch", height=35 * (len(frame) + 1) + 3)
        elif (dd.ROOT / block.path).exists():
            st.image(str(dd.ROOT / block.path), caption=block.text)
        else:
            st.info(f"Figure not in the repository yet: {block.path} ({block.text})")
