"""Dataset card generation for synthpix-hosted PIV datasets."""

from __future__ import annotations

import subprocess  # nosec B404 - used for read-only `git rev-parse`
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version

from synthpix.hf.layout import LayoutSummary

_LICENSE_NOTE = (
    "This dataset is hosted under terms set by the original authors. "
    "See source URL for details."
)
_DEFAULT_TAGS: tuple[str, ...] = ("PIV", "synthetic", "optical-flow")


@dataclass
class DatasetCardMeta:
    """User-facing metadata for a synthpix-hosted dataset card.

    Attributes:
        name: Repository-style short name of the dataset.
        source_url: URL of the original dataset.
        citation: BibTeX or free-form citation text.
        license: SPDX-like license identifier exposed in the YAML frontmatter.
        license_name: Optional license display name.
        synthpix_version: Version of ``synthpix`` used to produce the card.
            Auto-filled from ``importlib.metadata`` when left as ``None``.
        synthpix_commit: Git commit of the working tree where the card was
            generated. Auto-filled via ``git rev-parse HEAD`` when possible;
            stays ``None`` outside a repository.
        pretty_name: Display name used in the H1 heading and frontmatter.
        tags: Hub tags exposed in the YAML frontmatter.
    """

    name: str
    source_url: str
    citation: str
    license: str = "other"
    license_name: str = "research-only-arr"
    synthpix_version: str | None = None
    synthpix_commit: str | None = None
    pretty_name: str | None = None
    tags: tuple[str, ...] = field(default_factory=lambda: _DEFAULT_TAGS)


def _lookup_synthpix_version() -> str | None:
    """Look up the installed ``synthpix`` version, if any."""
    try:
        return version("synthpix")
    except PackageNotFoundError:
        return None


def _lookup_git_commit() -> str | None:
    """Return the current git commit, or ``None`` if unavailable."""
    try:
        output = subprocess.check_output(  # nosec B603 B607
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None
    return output.decode("utf-8").strip() or None


def _resolved_meta(meta: DatasetCardMeta) -> DatasetCardMeta:
    """Return a copy of ``meta`` with auto-fillable fields populated."""
    synthpix_version = meta.synthpix_version
    if synthpix_version is None:
        synthpix_version = _lookup_synthpix_version()

    synthpix_commit = meta.synthpix_commit
    if synthpix_commit is None:
        synthpix_commit = _lookup_git_commit()

    return DatasetCardMeta(
        name=meta.name,
        source_url=meta.source_url,
        citation=meta.citation,
        license=meta.license,
        license_name=meta.license_name,
        synthpix_version=synthpix_version,
        synthpix_commit=synthpix_commit,
        pretty_name=meta.pretty_name,
        tags=meta.tags,
    )


def _format_tags(tags: tuple[str, ...]) -> str:
    quoted = ", ".join(f"\"{tag}\"" for tag in tags)
    return f"[{quoted}]"


def _frontmatter(meta: DatasetCardMeta) -> str:
    pretty = meta.pretty_name or meta.name
    lines = [
        "---",
        f"license: {meta.license}",
        f"license_name: {meta.license_name}",
        f"pretty_name: \"{pretty}\"",
        f"tags: {_format_tags(meta.tags)}",
        "---",
    ]
    return "\n".join(lines)


def _split_table(layout: LayoutSummary) -> str:
    if not layout.splits:
        return "_No splits detected._"

    rows = ["| Split | .mat files |", "| --- | --- |"]
    for split, count in layout.splits.items():
        rows.append(f"| {split} | {count} |")
    return "\n".join(rows)


def _reynolds_section(layout: LayoutSummary) -> str:
    populated = {
        split: names
        for split, names in layout.reynolds_by_split.items()
        if names
    }
    if not populated:
        return ""

    lines = ["", "### Reynolds / scenario subdirectories", ""]
    lines.append("| Split | Subdirectories |")
    lines.append("| --- | --- |")
    for split, names in populated.items():
        lines.append(f"| {split} | {', '.join(names)} |")
    return "\n".join(lines)


def _provenance(meta: DatasetCardMeta) -> str:
    parts = []
    if meta.synthpix_version:
        parts.append(f"synthpix {meta.synthpix_version}")
    if meta.synthpix_commit:
        parts.append(f"commit `{meta.synthpix_commit}`")
    if not parts:
        return ""
    joined = ", ".join(parts)
    return f"\nGenerated with {joined}.\n"


def make_dataset_card(
    meta: DatasetCardMeta, layout: LayoutSummary
) -> str:
    """Render a Hugging Face dataset card README for ``meta`` and ``layout``.

    Args:
        meta: User-facing metadata. Auto-fillable fields
            (``synthpix_version``, ``synthpix_commit``) are populated when
            ``None``.
        layout: Local layout summary used to render the split table.

    Returns:
        str: The rendered README content, starting with YAML frontmatter.
    """
    resolved = _resolved_meta(meta)
    pretty = resolved.pretty_name or resolved.name

    sections = [
        _frontmatter(resolved),
        "",
        f"# {pretty}",
        "",
        f"Source: {resolved.source_url}",
        "",
        "## Citation",
        "",
        "```",
        resolved.citation,
        "```",
        "",
        "## License",
        "",
        _LICENSE_NOTE,
        "",
        "## Contents",
        "",
        _split_table(layout),
    ]
    reynolds = _reynolds_section(layout)
    if reynolds:
        sections.append(reynolds)

    provenance = _provenance(resolved)
    if provenance:
        sections.append(provenance)

    return "\n".join(sections).rstrip() + "\n"
