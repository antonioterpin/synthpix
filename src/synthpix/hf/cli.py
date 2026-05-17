"""Argparse entry point for the ``synthpix-hf`` console script."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from synthpix.hf.card import DatasetCardMeta, make_dataset_card
from synthpix.hf.layout import inspect_local_layout
from synthpix.utils import SYNTHPIX_SCOPE, get_logger

logger = get_logger(__name__, scope=SYNTHPIX_SCOPE)

_NOT_IMPLEMENTED = (
    "This subcommand is not yet implemented in PR1. "
    "Stay tuned for the upcoming push/pull rollouts."
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="synthpix-hf",
        description=("Tooling for synthpix-hosted Hugging Face PIV datasets."),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    card = sub.add_parser(
        "card", help="Generate a dataset card README from a local layout."
    )
    card.add_argument("local_dir", type=Path)
    card.add_argument("--source-url", required=True)
    citation_group = card.add_mutually_exclusive_group(required=True)
    citation_group.add_argument(
        "--citation",
        help="Literal citation text. Mutually exclusive with --citation-file.",
    )
    citation_group.add_argument(
        "--citation-file",
        type=Path,
        help="Path to a file whose contents are used as the citation.",
    )
    card.add_argument("--name", default=None)
    card.add_argument("--pretty-name", default=None)
    card.add_argument("--license", default="other")
    card.add_argument("--license-name", default="research-only-arr")
    card.add_argument("--output", type=Path, default=None)
    card.add_argument("--force", action="store_true")

    push = sub.add_parser("push", help="(PR2) Push a dataset to the Hub.")
    push.add_argument("local_dir", nargs="?", default=None)
    push.add_argument("--repo-id", default=None)

    pull = sub.add_parser("pull", help="(PR3) Pull a dataset from the Hub.")
    pull.add_argument("repo_id", nargs="?", default=None)
    pull.add_argument("--local-dir", default=None)

    return parser


def _resolve_citation(args: argparse.Namespace) -> str:
    """Return the citation contents from the chosen mutually exclusive flag.

    Args:
        args: Parsed ``argparse`` namespace from the ``card`` subparser.

    Returns:
        str: Either the literal ``--citation`` value or the contents of the
            file pointed at by ``--citation-file``.
    """
    if args.citation_file is not None:
        return args.citation_file.read_text()
    return args.citation


def _run_card(args: argparse.Namespace) -> int:
    local_dir = args.local_dir
    if not local_dir.is_dir():
        logger.error(f"Local directory does not exist: {local_dir}")
        return 2

    output = args.output if args.output is not None else local_dir / "README.md"
    if output.exists() and not args.force:
        logger.error(f"Refusing to overwrite {output} without --force.")
        return 1

    layout = inspect_local_layout(local_dir)
    citation = _resolve_citation(args)
    name = args.name or local_dir.name
    meta = DatasetCardMeta(
        name=name,
        source_url=args.source_url,
        citation=citation,
        license=args.license,
        license_name=args.license_name,
        pretty_name=args.pretty_name,
    )

    card = make_dataset_card(meta, layout)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(card)
    logger.info(f"Wrote dataset card to {output}")
    return 0


def _run_stub(name: str) -> int:
    print(f"`synthpix-hf {name}`: {_NOT_IMPLEMENTED}", file=sys.stderr)
    return 2


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``synthpix-hf``.

    Args:
        argv: Optional explicit argument list; defaults to ``sys.argv[1:]``.

    Returns:
        int: A POSIX-style exit code.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "card":
        return _run_card(args)
    if args.command in {"push", "pull"}:
        return _run_stub(args.command)

    # ``add_subparsers(required=True)`` makes this unreachable; ``parser.error``
    # exits in any case but mypy/basedpyright need a concrete return.
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
