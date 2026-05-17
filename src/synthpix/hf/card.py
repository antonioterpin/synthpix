"""Dataset card generation for synthpix-hosted PIV datasets."""

from __future__ import annotations

import io
import re
import subprocess  # nosec B404 - used for read-only `git rev-parse`
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from ruamel.yaml import YAML

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
    """Look up the installed ``synthpix`` version, if any.

    Returns:
        str | None: The package version from ``importlib.metadata``, or
            ``None`` when ``synthpix`` is not installed.
    """
    try:
        return version("synthpix")
    except PackageNotFoundError:
        return None


def _lookup_git_commit() -> str | None:
    """Return the current synthpix-source git commit, if available.

    The commit is resolved relative to the installed ``synthpix`` package
    location (``Path(__file__).parent``) rather than the process working
    directory, so the recorded value always describes the synthpix tree the
    card was generated against — never an unrelated user project that happens
    to be the CWD at invocation time. When synthpix is installed from a wheel
    (no surrounding ``.git``), this returns ``None``.

    Returns:
        str | None: The full commit SHA, or ``None`` when ``git`` is missing
            or the synthpix package is not inside a git checkout.
    """
    try:
        output = subprocess.check_output(  # nosec B603 B607
            ["git", "rev-parse", "HEAD"],
            cwd=str(Path(__file__).resolve().parent),
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None
    return output.decode("utf-8").strip() or None


def _resolved_meta(meta: DatasetCardMeta) -> DatasetCardMeta:
    """Return a copy of ``meta`` with auto-fillable fields populated.

    Args:
        meta: Caller-supplied metadata; ``synthpix_version`` /
            ``synthpix_commit`` are filled in when left as ``None``.

    Returns:
        DatasetCardMeta: A new instance with provenance fields resolved.
    """
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


def _frontmatter(meta: DatasetCardMeta) -> str:
    """Serialize the YAML frontmatter block via ``ruamel.yaml``.

    Using a real YAML serializer (instead of string interpolation) means
    embedded quotes, backslashes, colons or newlines in ``pretty_name`` or
    ``tags`` cannot produce malformed frontmatter that fails to parse on the
    Hub.

    Args:
        meta: Resolved card metadata to serialize.

    Returns:
        str: A ``---``-delimited YAML block, no trailing newline.
    """
    pretty = meta.pretty_name or meta.name
    payload = {
        "license": meta.license,
        "license_name": meta.license_name,
        "pretty_name": pretty,
        "tags": list(meta.tags),
    }
    yaml = YAML(typ="safe", pure=True)
    yaml.default_flow_style = False
    buf = io.StringIO()
    yaml.dump(payload, buf)
    return "---\n" + buf.getvalue().rstrip("\n") + "\n---"


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
        for split, names in layout.subdirs_by_split.items()
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


def _fence_for(text: str) -> str:
    """Return a backtick fence longer than the longest run inside ``text``.

    Citation strings can legitimately contain triple-backtick code blocks; a
    naive fixed ```` ``` ```` fence around the citation would terminate
    prematurely. We scan for the longest run of backticks in the input and
    pick a fence one longer (minimum length 3).

    Args:
        text: Content that will sit between the returned fence markers.

    Returns:
        str: A backtick string of length ``max(3, longest_run + 1)``.
    """
    longest = max(
        (len(m.group(0)) for m in re.finditer(r"`+", text)), default=0
    )
    return "`" * max(3, longest + 1)


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


def make_dataset_card(meta: DatasetCardMeta, layout: LayoutSummary) -> str:
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
    fence = _fence_for(resolved.citation)

    sections = [
        _frontmatter(resolved),
        "",
        f"# {pretty}",
        "",
        f"Source: {resolved.source_url}",
        "",
        "## Citation",
        "",
        fence,
        resolved.citation,
        fence,
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
