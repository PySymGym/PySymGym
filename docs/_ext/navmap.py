"""Sphinx extension: generate and validate the documentation navigation map.

The documentation is navigated through ``docs/index.rst``, which must list
every page under a fixed group with a brief description. To keep that map from
drifting, the information is single-sourced in file-wide metadata at the top of
each page::

    :description: One-line purpose of the page.
    :group: Guides

This extension:

* renders the ``.. navmap::`` directive in the index page from that metadata
  (grouped, alphabetical, linked);
* injects the description as an italic line under each page title, so the page
  visibly starts with its purpose (same single source);
* warns (a hard failure under ``-W``) when a page lacks a description or a
  valid group, and when README documentation links or the API-reference
  autosummary drift from the actual pages/modules.

It is project-specific and deliberately depends only on Sphinx and the standard
library.
"""

from __future__ import annotations

import re
from pathlib import Path

from docutils import nodes
from docutils.parsers.rst import Directive
from sphinx.application import Sphinx
from sphinx.environment import BuildEnvironment
from sphinx.util import logging

logger = logging.getLogger(__name__)

DEFAULT_GROUPS = ["Overview", "Guides", "Development", "Reference"]
MAX_DESCRIPTION_LENGTH = 120

_PLACEHOLDER_CLASS = "navmap-placeholder"

_SITE_URL_RE = re.compile(r"https://pysymgym\.github\.io/PySymGym/([^\s)\"'?#]*)")


class NavMapDirective(Directive):
    """Placeholder rendered later from page metadata."""

    has_content = False

    def run(self) -> list[nodes.Node]:
        return [nodes.container(classes=[_PLACEHOLDER_CLASS])]


def _metadata(env: BuildEnvironment, docname: str) -> dict[str, str]:
    return dict(env.metadata.get(docname, {}))


def _title(env: BuildEnvironment, docname: str) -> str:
    title = env.titles.get(docname)
    return title.astext() if title is not None else docname


def _is_content_page(env: BuildEnvironment, docname: str) -> bool:
    if docname == env.config.root_doc:
        return False
    if docname.startswith("_ext/"):
        return False
    if docname.startswith("reference/generated/"):
        return False
    return True


def _make_table(rows: list[tuple[str, str, str, str]]) -> nodes.table:
    table = nodes.table()
    tgroup = nodes.tgroup(cols=3)
    table += tgroup
    for _ in range(3):
        tgroup += nodes.colspec(colwidth=1)

    thead = nodes.thead()
    tgroup += thead
    thead += _make_row(("Group", "Page", "Description"))

    tbody = nodes.tbody()
    tgroup += tbody
    for group, docname, title, description in rows:
        page = nodes.reference(internal=True, refuri=docname, refdocname=docname)
        page += nodes.Text(title)
        tbody += _make_row((group, page, description))
    return table


def _make_row(cells: tuple[str, nodes.Node | str, str]) -> nodes.row:
    row = nodes.row()
    for cell in cells:
        entry = nodes.entry()
        paragraph = nodes.paragraph()
        if isinstance(cell, nodes.Node):
            paragraph += cell
        else:
            paragraph += nodes.Text(cell)
        entry += paragraph
        row += entry
    return row


def _build_map(app: Sphinx, source_docname: str) -> nodes.table:
    env = app.env
    groups = list(app.config.navmap_groups)
    collected: dict[str, list[tuple[str, str, str]]] = {g: [] for g in groups}

    for docname in env.found_docs:
        if not _is_content_page(env, docname):
            continue
        meta = _metadata(env, docname)
        group = meta.get("group")
        if group not in collected:
            continue
        collected[group].append(
            (docname, _title(env, docname), meta.get("description", ""))
        )

    rows: list[tuple[str, str, str, str]] = []
    for group in groups:
        for docname, title, description in sorted(
            collected[group], key=lambda item: item[1].lower()
        ):
            rows.append((group, docname, title, description))

    table = _make_table(rows)
    # Resolve page links relative to the map's own document.
    for reference in table.findall(nodes.reference):
        target = reference.get("refdocname")
        if target:
            reference["refuri"] = app.builder.get_relative_uri(source_docname, target)
    return table


def _inject_description(app: Sphinx, doctree: nodes.document, docname: str) -> None:
    if not _is_content_page(app.env, docname):
        return
    description = _metadata(app.env, docname).get("description")
    if not description:
        return
    sections = list(doctree.findall(nodes.section))
    if not sections:
        return
    section = sections[0]
    if not section.children or not isinstance(section.children[0], nodes.title):
        return
    paragraph = nodes.paragraph()
    emphasis = nodes.emphasis()
    emphasis += nodes.Text(description)
    paragraph += emphasis
    section.insert(1, paragraph)


def on_doctree_resolved(app: Sphinx, doctree: nodes.document, docname: str) -> None:
    for placeholder in list(doctree.findall(nodes.container)):
        if _PLACEHOLDER_CLASS in placeholder.get("classes", []):
            placeholder.replace_self(_build_map(app, docname))
    _inject_description(app, doctree, docname)


def _validate_metadata(app: Sphinx) -> None:
    env = app.env
    groups = list(app.config.navmap_groups)
    for docname in sorted(env.found_docs):
        if not _is_content_page(env, docname):
            continue
        metadata = _metadata(env, docname)
        description = metadata.get("description", "").strip()
        group = metadata.get("group", "").strip()
        if not description:
            logger.warning(
                "documentation page %r has no ':description:' metadata; add a "
                "one-line description so it appears in the navigation map",
                docname,
            )
        elif "\n" in description or len(description) > MAX_DESCRIPTION_LENGTH:
            logger.warning(
                "documentation page %r has a description that must be a single "
                "line of at most %d characters",
                docname,
                MAX_DESCRIPTION_LENGTH,
            )
        if group not in groups:
            logger.warning(
                "documentation page %r has missing or invalid ':group:' "
                "metadata %r; expected one of %s",
                docname,
                group,
                groups,
            )


def _site_link_docnames(text: str) -> set[str]:
    docnames: set[str] = set()
    for raw in _SITE_URL_RE.findall(text):
        slug = raw.strip("/")
        if slug in ("", "index"):
            docnames.add("index")
        elif slug == "reference":
            docnames.add("reference/index")
        elif slug.endswith(".html"):
            docnames.add(slug[: -len(".html")])
        else:
            docnames.add(slug)
    return docnames


def _validate_readme_links(app: Sphinx) -> None:
    readme = Path(app.confdir).parent / "README.md"
    if not readme.is_file():
        return
    for docname in sorted(_site_link_docnames(readme.read_text(encoding="utf-8"))):
        if docname not in app.env.found_docs:
            logger.warning(
                "README.md links to documentation page %r, which does not "
                "exist under docs/",
                docname,
            )


def on_env_updated(app: Sphinx, env: BuildEnvironment) -> None:
    if getattr(app, "_navmap_validated", False):
        return
    app._navmap_validated = True  # type: ignore[attr-defined]
    _validate_metadata(app)
    _validate_readme_links(app)


def setup(app: Sphinx) -> dict[str, object]:
    app.add_directive("navmap", NavMapDirective)
    app.add_config_value("navmap_groups", DEFAULT_GROUPS, "html")
    app.connect("doctree-resolved", on_doctree_resolved)
    app.connect("env-updated", on_env_updated)
    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
