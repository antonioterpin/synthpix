"""Tests for the ``synthpix-hf`` argparse CLI."""

from __future__ import annotations

from pathlib import Path

import pytest

from synthpix.hf import cli


def _populate(tmp_path: Path) -> None:
    """Drop a single ``.mat`` file so ``inspect_local_layout`` succeeds."""
    train = tmp_path / "train"
    train.mkdir()
    (train / "flow.mat").write_bytes(b"x" * 8)


def test_card_subcommand_writes_readme(tmp_path: Path):
    _populate(tmp_path)
    rc = cli.main(
        [
            "card",
            str(tmp_path),
            "--source-url",
            "https://example.org/data",
            "--citation",
            "fake citation",
            "--pretty-name",
            "Example",
        ]
    )

    readme = tmp_path / "README.md"
    assert rc == 0
    assert readme.exists()
    assert "fake citation" in readme.read_text()


def test_card_subcommand_refuses_overwrite_without_force(tmp_path: Path):
    _populate(tmp_path)
    readme = tmp_path / "README.md"
    readme.write_text("existing content")

    rc = cli.main(
        [
            "card",
            str(tmp_path),
            "--source-url",
            "https://example.org/data",
            "--citation",
            "fake citation",
        ]
    )

    assert rc != 0
    assert readme.read_text() == "existing content"


def test_card_subcommand_overwrites_with_force(tmp_path: Path):
    _populate(tmp_path)
    readme = tmp_path / "README.md"
    readme.write_text("existing content")

    rc = cli.main(
        [
            "card",
            str(tmp_path),
            "--source-url",
            "https://example.org/data",
            "--citation",
            "fake citation",
            "--force",
        ]
    )

    assert rc == 0
    assert "fake citation" in readme.read_text()


def test_card_subcommand_reads_citation_from_file(tmp_path: Path):
    _populate(tmp_path)
    citation_file = tmp_path / "CITATION.bib"
    citation_file.write_text("@misc{example, title={Example}}")

    rc = cli.main(
        [
            "card",
            str(tmp_path),
            "--source-url",
            "https://example.org/data",
            "--citation-file",
            str(citation_file),
        ]
    )

    readme = tmp_path / "README.md"
    assert rc == 0
    assert "@misc{example" in readme.read_text()


def test_card_citation_literal_not_treated_as_path(tmp_path: Path):
    # A literal --citation that matches a CWD file must not be silently read.
    _populate(tmp_path)
    # Create a file whose name matches the literal we will pass.
    decoy = tmp_path / "lookup-by-coincidence.txt"
    decoy.write_text("file-contents-should-not-leak")

    rc = cli.main(
        [
            "card",
            str(tmp_path),
            "--source-url",
            "https://example.org/data",
            "--citation",
            str(decoy),
        ]
    )

    readme = (tmp_path / "README.md").read_text()
    assert rc == 0
    # The literal string was preserved verbatim — the decoy content did not leak in.
    assert "file-contents-should-not-leak" not in readme
    assert str(decoy) in readme


def test_card_citation_and_citation_file_are_mutually_exclusive(tmp_path: Path):
    _populate(tmp_path)
    citation_file = tmp_path / "CITATION.bib"
    citation_file.write_text("@misc{example}")

    with pytest.raises(SystemExit):
        cli.main(
            [
                "card",
                str(tmp_path),
                "--source-url",
                "https://example.org/data",
                "--citation",
                "literal",
                "--citation-file",
                str(citation_file),
            ]
        )


def test_card_output_creates_missing_parent_directory(tmp_path: Path):
    _populate(tmp_path)
    output = tmp_path / "nested" / "deeper" / "card.md"
    assert not output.parent.exists()

    rc = cli.main(
        [
            "card",
            str(tmp_path),
            "--source-url",
            "https://example.org/data",
            "--citation",
            "fake citation",
            "--output",
            str(output),
        ]
    )

    assert rc == 0
    assert output.exists()


def test_card_subcommand_respects_output(tmp_path: Path):
    _populate(tmp_path)
    output = tmp_path / "custom.md"

    rc = cli.main(
        [
            "card",
            str(tmp_path),
            "--source-url",
            "https://example.org/data",
            "--citation",
            "fake citation",
            "--output",
            str(output),
        ]
    )

    assert rc == 0
    assert output.exists()
    assert not (tmp_path / "README.md").exists()


@pytest.mark.parametrize("subcommand", ["push", "pull"])
def test_stub_subcommands_report_not_implemented(
    subcommand: str, capsys
):
    rc = cli.main([subcommand])

    captured = capsys.readouterr()
    assert rc != 0
    assert "not yet implemented" in (captured.out + captured.err).lower()
