"""Tests for the ``synthpix-hf`` argparse CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from synthpix.hf import cli


def _populate(tmp_path: Path) -> None:
    """Populate.

    Args:
        tmp_path: Temporary directory for this test.
    """
    # Drop a single ``.mat`` file so ``inspect_local_layout`` succeeds.
    train = tmp_path / "train"
    train.mkdir()
    (train / "flow.mat").write_bytes(b"x" * 8)


def test_card_subcommand_writes_readme(tmp_path: Path) -> None:
    """Verify card subcommand writes readme.

    Args:
        tmp_path: Temporary directory for this test.
    """
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


def test_card_subcommand_refuses_overwrite_without_force(
    tmp_path: Path,
) -> None:
    """Verify card subcommand refuses overwrite without force.

    Args:
        tmp_path: Temporary directory for this test.
    """
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


def test_card_subcommand_overwrites_with_force(tmp_path: Path) -> None:
    """Verify card subcommand overwrites with force.

    Args:
        tmp_path: Temporary directory for this test.
    """
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


def test_card_subcommand_reads_citation_from_file(tmp_path: Path) -> None:
    """Verify card subcommand reads citation from file.

    Args:
        tmp_path: Temporary directory for this test.
    """
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


def test_card_citation_literal_not_treated_as_path(tmp_path: Path) -> None:
    """Verify card citation literal not treated as path.

    Args:
        tmp_path: Temporary directory for this test.
    """
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
    # The literal string was preserved verbatim — the decoy content did not leak
    # in.
    assert "file-contents-should-not-leak" not in readme
    assert str(decoy) in readme


def test_card_citation_and_citation_file_are_mutually_exclusive(
    tmp_path: Path,
) -> None:
    """Verify card citation and citation file are mutually exclusive.

    Args:
        tmp_path: Temporary directory for this test.
    """
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


def test_card_output_creates_missing_parent_directory(tmp_path: Path) -> None:
    """Verify card output creates missing parent directory.

    Args:
        tmp_path: Temporary directory for this test.
    """
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


def test_card_subcommand_respects_output(tmp_path: Path) -> None:
    """Verify card subcommand respects output.

    Args:
        tmp_path: Temporary directory for this test.
    """
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


def test_cli_push_requires_allow_public_when_public(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify cli push requires allow public when public.

    Args:
        monkeypatch: Fixture for restoring patched attributes and
            environment variables.
        tmp_path: Temporary directory for this test.
        capsys: Capture of standard output and standard error.
    """
    monkeypatch.setattr(cli, "push_dataset", lambda *a, **kw: "should-not-call")

    rc = cli.main(["push", str(tmp_path), "user/repo", "--public"])

    captured = capsys.readouterr()
    assert rc != 0
    assert "--allow-public" in captured.err


def test_cli_push_public_with_allow_public_skips_tty_prompt_when_not_tty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify cli push public with allow public skips tty prompt when not
    tty.

    Args:
        monkeypatch: Fixture for restoring patched attributes and
            environment variables.
        tmp_path: Temporary directory for this test.
    """
    captured: dict = {}

    def _fake(local_dir: Path, repo_id: str, **kwargs: Any) -> str:
        """Fake.

        Args:
            local_dir: Local dir for this test case.
            repo_id: Repo id for this test case.
            **kwargs: Keyword arguments accepted by the test double.

        Returns:
            The value produced for the test scenario.
        """
        captured["local_dir"] = local_dir
        captured["repo_id"] = repo_id
        captured.update(kwargs)
        return "sha-abc"

    monkeypatch.setattr(cli, "push_dataset", _fake)
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: False)

    rc = cli.main(
        [
            "push",
            str(tmp_path),
            "user/repo",
            "--public",
            "--allow-public",
        ]
    )

    assert rc == 0
    assert captured["private"] is False
    assert captured["allow_public"] is True


def test_cli_push_public_prompts_on_tty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify cli push public prompts on tty.

    Args:
        monkeypatch: Fixture for restoring patched attributes and
            environment variables.
        tmp_path: Temporary directory for this test.
    """
    calls: list[dict] = []

    def _fake(local_dir: Path, repo_id: str, **kwargs: Any) -> str:
        """Fake.

        Args:
            local_dir: Local dir for this test case.
            repo_id: Repo id for this test case.
            **kwargs: Keyword arguments accepted by the test double.

        Returns:
            The value produced for the test scenario.
        """
        calls.append({"local_dir": local_dir, "repo_id": repo_id, **kwargs})
        return "sha-abc"

    monkeypatch.setattr(cli, "push_dataset", _fake)
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda *a, **kw: "yes")

    rc = cli.main(
        [
            "push",
            str(tmp_path),
            "user/repo",
            "--public",
            "--allow-public",
        ]
    )
    assert rc == 0
    assert len(calls) == 1

    monkeypatch.setattr("builtins.input", lambda *a, **kw: "no")
    rc = cli.main(
        [
            "push",
            str(tmp_path),
            "user/repo",
            "--public",
            "--allow-public",
        ]
    )
    assert rc != 0
    assert len(calls) == 1


def test_cli_push_card_partial_args_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify cli push card partial args errors.

    Args:
        monkeypatch: Fixture for restoring patched attributes and
            environment variables.
        tmp_path: Temporary directory for this test.
        capsys: Capture of standard output and standard error.
    """
    monkeypatch.setattr(cli, "push_dataset", lambda *a, **kw: "should-not-call")

    rc = cli.main(
        [
            "push",
            str(tmp_path),
            "user/repo",
            "--card-source-url",
            "https://example.org",
        ]
    )

    captured = capsys.readouterr()
    assert rc != 0
    assert "card" in captured.err.lower()


def test_cli_push_card_full_args_builds_meta(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify cli push card full args builds meta.

    Args:
        monkeypatch: Fixture for restoring patched attributes and
            environment variables.
        tmp_path: Temporary directory for this test.
    """
    captured: dict = {}

    def _fake(local_dir: Path, repo_id: str, **kwargs: Any) -> str:
        """Fake.

        Args:
            local_dir: Local dir for this test case.
            repo_id: Repo id for this test case.
            **kwargs: Keyword arguments accepted by the test double.

        Returns:
            The value produced for the test scenario.
        """
        captured.update(kwargs)
        return "sha-1"

    monkeypatch.setattr(cli, "push_dataset", _fake)

    rc = cli.main(
        [
            "push",
            str(tmp_path),
            "user/repo",
            "--card-source-url",
            "https://example.org/src",
            "--card-citation",
            "free-form citation",
            "--card-name",
            "Pretty Name",
        ]
    )

    assert rc == 0
    meta = captured["card_meta"]
    assert meta is not None
    assert meta.source_url == "https://example.org/src"
    assert meta.citation == "free-form citation"
    assert meta.pretty_name == "Pretty Name"


def test_cli_push_no_card_passes_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify cli push no card passes none.

    Args:
        monkeypatch: Fixture for restoring patched attributes and
            environment variables.
        tmp_path: Temporary directory for this test.
    """
    captured: dict = {}

    def _fake(local_dir: Path, repo_id: str, **kwargs: Any) -> str:
        """Fake.

        Args:
            local_dir: Local dir for this test case.
            repo_id: Repo id for this test case.
            **kwargs: Keyword arguments accepted by the test double.

        Returns:
            The value produced for the test scenario.
        """
        captured.update(kwargs)
        return "sha-1"

    monkeypatch.setattr(cli, "push_dataset", _fake)

    rc = cli.main(
        [
            "push",
            str(tmp_path),
            "user/repo",
            "--no-card",
            "--card-source-url",
            "https://example.org/src",
            "--card-citation",
            "ignored",
        ]
    )

    assert rc == 0
    assert captured["card_meta"] is None


def test_cli_push_dry_run_propagates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify cli push dry run propagates.

    Args:
        monkeypatch: Fixture for restoring patched attributes and
            environment variables.
        tmp_path: Temporary directory for this test.
    """
    captured: dict = {}

    def _fake(local_dir: Path, repo_id: str, **kwargs: Any) -> str:
        """Fake.

        Args:
            local_dir: Local dir for this test case.
            repo_id: Repo id for this test case.
            **kwargs: Keyword arguments accepted by the test double.

        Returns:
            The value produced for the test scenario.
        """
        captured.update(kwargs)
        return "dry-run"

    monkeypatch.setattr(cli, "push_dataset", _fake)

    rc = cli.main(["push", str(tmp_path), "user/repo", "--dry-run"])

    assert rc == 0
    assert captured["dry_run"] is True


def test_cli_push_token_not_logged(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verify cli push token not logged.

    Args:
        monkeypatch: Fixture for restoring patched attributes and
            environment variables.
        tmp_path: Temporary directory for this test.
        capsys: Capture of standard output and standard error.
        caplog: Capture of emitted log records.
    """

    def _fake(local_dir: Path, repo_id: str, **kwargs: Any) -> str:
        """Fake.

        Args:
            local_dir: Local dir for this test case.
            repo_id: Repo id for this test case.
            **kwargs: Keyword arguments accepted by the test double.

        Returns:
            The value produced for the test scenario.
        """
        return "sha-secret-free"

    monkeypatch.setattr(cli, "push_dataset", _fake)

    rc = cli.main(
        [
            "push",
            str(tmp_path),
            "user/repo",
            "--token",
            "hf_secret",
        ]
    )

    captured = capsys.readouterr()
    assert rc == 0
    assert "hf_secret" not in captured.out
    assert "hf_secret" not in captured.err
    for record in caplog.records:
        assert "hf_secret" not in record.getMessage()


def test_cli_pull_invokes_pull_dataset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify cli pull invokes pull dataset.

    Args:
        monkeypatch: Fixture for restoring patched attributes and
            environment variables.
        tmp_path: Temporary directory for this test.
    """
    captured: dict = {}

    def _fake(repo_id: str, local_dir: Path, **kwargs: Any) -> Path:
        """Fake.

        Args:
            repo_id: Repo id for this test case.
            local_dir: Local dir for this test case.
            **kwargs: Keyword arguments accepted by the test double.

        Returns:
            The value produced for the test scenario.
        """
        captured["repo_id"] = repo_id
        captured["local_dir"] = local_dir
        captured.update(kwargs)
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        return Path(local_dir)

    monkeypatch.setattr(cli, "pull_dataset", _fake)

    rc = cli.main(
        [
            "pull",
            "user/repo",
            str(tmp_path / "data"),
            "--revision",
            "v1",
            "--splits",
            "train,val",
        ]
    )

    assert rc == 0
    assert captured["repo_id"] == "user/repo"
    assert captured["local_dir"] == tmp_path / "data"
    assert captured["revision"] == "v1"
    assert captured["splits"] == ("train", "val")
    assert captured["token"] is None


def test_cli_pull_multiple_include_flags(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify cli pull multiple include flags.

    Args:
        monkeypatch: Fixture for restoring patched attributes and
            environment variables.
        tmp_path: Temporary directory for this test.
    """
    captured: dict = {}

    def _fake(repo_id: str, local_dir: Path, **kwargs: Any) -> Path:
        """Fake.

        Args:
            repo_id: Repo id for this test case.
            local_dir: Local dir for this test case.
            **kwargs: Keyword arguments accepted by the test double.

        Returns:
            The value produced for the test scenario.
        """
        captured.update(kwargs)
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        return Path(local_dir)

    monkeypatch.setattr(cli, "pull_dataset", _fake)

    rc = cli.main(
        [
            "pull",
            "user/repo",
            str(tmp_path / "data"),
            "--include",
            "train/**",
            "--include",
            "val/**",
        ]
    )

    assert rc == 0
    assert captured["include_globs"] == ("train/**", "val/**")


def test_cli_pull_exit_nonzero_on_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify cli pull exit nonzero on runtime error.

    Args:
        monkeypatch: Fixture for restoring patched attributes and
            environment variables.
        tmp_path: Temporary directory for this test.
        capsys: Capture of standard output and standard error.
    """

    def _boom(*_args: Any, **_kwargs: Any) -> None:
        """Boom.

        Args:
            *_args: Positional arguments accepted by the test double.
            **_kwargs: Keyword arguments accepted by the test double.

        Raises:
            RuntimeError: When the simulated failure is triggered.
        """
        raise RuntimeError("boom!")

    monkeypatch.setattr(cli, "pull_dataset", _boom)

    rc = cli.main(["pull", "user/repo", str(tmp_path / "data")])

    captured = capsys.readouterr()
    assert rc != 0
    assert "boom!" in captured.err


def test_cli_pull_splits_strips_whitespace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify cli pull splits strips whitespace.

    Args:
        monkeypatch: Fixture for restoring patched attributes and
            environment variables.
        tmp_path: Temporary directory for this test.
    """
    captured: dict = {}

    def _fake(repo_id: str, local_dir: Path, **kwargs: Any) -> Path:
        """Fake.

        Args:
            repo_id: Repo id for this test case.
            local_dir: Local dir for this test case.
            **kwargs: Keyword arguments accepted by the test double.

        Returns:
            The value produced for the test scenario.
        """
        captured.update(kwargs)
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        return Path(local_dir)

    monkeypatch.setattr(cli, "pull_dataset", _fake)

    rc = cli.main(
        [
            "pull",
            "user/repo",
            str(tmp_path / "data"),
            "--splits",
            "train, val ,  ,test",
        ]
    )

    assert rc == 0
    # Whitespace stripped, empty parts dropped.
    assert captured["splits"] == ("train", "val", "test")


def test_cli_pull_splits_all_empty_becomes_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify cli pull splits all empty becomes none.

    Args:
        monkeypatch: Fixture for restoring patched attributes and
            environment variables.
        tmp_path: Temporary directory for this test.
    """
    captured: dict = {}

    def _fake(repo_id: str, local_dir: Path, **kwargs: Any) -> Path:
        """Fake.

        Args:
            repo_id: Repo id for this test case.
            local_dir: Local dir for this test case.
            **kwargs: Keyword arguments accepted by the test double.

        Returns:
            The value produced for the test scenario.
        """
        captured.update(kwargs)
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        return Path(local_dir)

    monkeypatch.setattr(cli, "pull_dataset", _fake)

    rc = cli.main(
        [
            "pull",
            "user/repo",
            str(tmp_path / "data"),
            "--splits",
            " , , ",
        ]
    )

    assert rc == 0
    assert captured["splits"] is None


def test_cli_pull_summary_counts_root_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify cli pull summary counts root files.

    Args:
        monkeypatch: Fixture for restoring patched attributes and
            environment variables.
        tmp_path: Temporary directory for this test.
        capsys: Capture of standard output and standard error.
    """
    # The summary line must count root-level files (e.g. README.md) that
    # ``pull_dataset(..., splits=...)`` brings along, not just files under
    # the recognized split directories.
    target = tmp_path / "data"

    def _fake(repo_id: str, local_dir: Path, **_kwargs: Any) -> Path:
        """Fake.

        Args:
            repo_id: Repo id for this test case.
            local_dir: Local dir for this test case.
            **_kwargs: Keyword arguments accepted by the test double.

        Returns:
            The value produced for the test scenario.
        """
        local = Path(local_dir)
        (local / "train").mkdir(parents=True, exist_ok=True)
        (local / "train" / "a.mat").write_bytes(b"x" * 4)
        (local / "README.md").write_text("# card")
        return local

    monkeypatch.setattr(cli, "pull_dataset", _fake)

    rc = cli.main(
        [
            "pull",
            "user/repo",
            str(target),
            "--splits",
            "train",
        ]
    )

    out = capsys.readouterr().out
    assert rc == 0
    # 1 .mat + 1 README.md = 2 files
    assert "2 files" in out


def test_cli_pull_token_not_logged(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verify cli pull token not logged.

    Args:
        monkeypatch: Fixture for restoring patched attributes and
            environment variables.
        tmp_path: Temporary directory for this test.
        capsys: Capture of standard output and standard error.
        caplog: Capture of emitted log records.
    """

    def _fake(repo_id: str, local_dir: Path, **_kwargs: Any) -> Path:
        """Fake.

        Args:
            repo_id: Repo id for this test case.
            local_dir: Local dir for this test case.
            **_kwargs: Keyword arguments accepted by the test double.

        Returns:
            The value produced for the test scenario.
        """
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        return Path(local_dir)

    monkeypatch.setattr(cli, "pull_dataset", _fake)

    rc = cli.main(
        [
            "pull",
            "user/repo",
            str(tmp_path / "data"),
            "--token",
            "hf_secret",
        ]
    )

    captured = capsys.readouterr()
    assert rc == 0
    assert "hf_secret" not in captured.out
    assert "hf_secret" not in captured.err
    for record in caplog.records:
        assert "hf_secret" not in record.getMessage()
