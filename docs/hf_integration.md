# Hugging Face Hub integration

`SynthPix` ships an optional integration that round-trips a built PIV dataset
through a [Hugging Face Hub](https://huggingface.co/) dataset repository. The
on-disk layout is preserved byte-for-byte, so a dataset assembled on one
machine can be pulled onto Lambda, Kaggle, Colab, or a laptop and fed straight
to the existing `.mat` schedulers without any conversion step.

## What this integration is for

The intended workflow is: build a dataset locally with `scripts/download_piv_*.py`
(or your own pipeline), push it to a Hub repo, and pull it back on whatever
machine actually runs your training. You bring an HF token; the integration
handles repo creation, idempotent transfers, optional `hf_transfer`
acceleration, and an auto-generated dataset card.

The canonical use case is **private hosting**. The upstream PIV public
datasets (e.g. the class-1 set from `shengzesnail/PIV_dataset`) are released
under research-only / all-rights-reserved terms, and redistributing them
publicly is not permitted. The push API defaults to `private=True` and
requires an explicit `--allow-public` (or `allow_public=True`) flag before it
will create a public repo. See the License reminder at the bottom.

## Installation

```bash
uv sync --extra hf
# or
pip install synthpix[hf]
```

The `[hf]` extra pulls in `huggingface_hub` plus `hf_transfer`. The latter is
a Rust wheel that significantly accelerates uploads and downloads; the
integration works without it but falls back to the pure-Python downloader and
prints a one-line warning. If you skip the extra entirely, importing
`synthpix.hf` still works (the `huggingface_hub` import is deferred), but
`push_dataset` / `pull_dataset` will raise `ImportError` with a clear message.

## Authentication

Tokens are resolved in this order:

1. An explicit `token=...` (Python) or `--token` (CLI) argument.
2. The `HF_TOKEN` environment variable.
3. The `HF_HUB_TOKEN` environment variable.
4. The cached credentials managed by `huggingface-cli login`.

Empty strings are treated as missing. The integration **never persists or
prints tokens** — the resolution helper only reads.

Pick whichever flavor is least surprising in your environment:

```bash
# One-shot for the current shell
export HF_TOKEN=hf_xxxxxxxx

# Persistent on a workstation
uv run huggingface-cli login
```

For a fine-grained token scoped to one dataset, create it under
`Settings → Access Tokens → Create new token → Fine-grained`, select **Write
access to one specific repository**, and pin it to your dataset repo. That is
all `push_dataset` and `pull_dataset` need.

## Pushing a built dataset

There are three ways in, ordered from "quickest" to "most flexible".

### From the CLI

```bash
# Preview what would be uploaded; safe to run without a real token resolved.
uv run synthpix-hf push ./piv_dataset_class1_256 user/piv-dataset-class1-256 \
    --dry-run

# Real push (default: private).
uv run synthpix-hf push ./piv_dataset_class1_256 user/piv-dataset-class1-256
```

A dry run lists which files would be added and which already exist on the
remote. The first dry-run against a repo that does not exist yet treats every
local file as new — that is expected and not an error.

To create a public repo you must opt in twice and confirm on the TTY:

```bash
uv run synthpix-hf push ./piv_dataset_class1_256 user/piv-dataset-class1-256 \
    --public --allow-public
# Refusing to push public unless you type 'yes': yes
```

The two-step `--public` + `--allow-public` gate exists because the upstream
PIV class-1 license forbids redistribution. The TTY prompt prevents a
copy/paste from silently flipping a private dataset public; non-TTY callers
(CI, pipelines) skip the prompt and are trusted to mean what they say.

### From a build script

`scripts/download_piv_1.py` and `scripts/download_piv_2.py` accept a
`--push-to` flag that chains build → push in one invocation:

```bash
uv run python scripts/download_piv_1.py \
    --out-dir ~/data/piv_dataset_class1_256 \
    --push-to user/piv-dataset-class1-256 \
    --push-token "$HF_TOKEN"
```

Relevant push flags on those scripts:

- `--push-to <owner>/<name>` — activate the push step.
- `--push-private` — default; private repo.
- `--push-public --allow-public` — opt-in public, same gate as the CLI.
- `--push-token <token>` — explicit token; falls back to `HF_TOKEN` / cache.
- `--no-push-card` — skip auto-generated dataset-card metadata.

The scripts ship a sensible default `DatasetCardMeta` (source URL, citation,
license, tags) for each PIV class.

### Programmatic

```python
from synthpix.hf import DatasetCardMeta, push_dataset

card = DatasetCardMeta(
    name="piv-dataset-class1-256",
    source_url="https://github.com/shengzesnail/PIV_dataset",
    citation="Cai et al., Exp Fluids 2019 (see upstream README)",
)
sha = push_dataset(
    local_dir="~/data/piv_dataset_class1_256",
    repo_id="user/piv-dataset-class1-256",
    private=True,
    card_meta=card,
)
print(sha)
```

The function returns the commit sha (or `"dry-run"` when `dry_run=True`). It
is idempotent: rerunning with the same `local_dir` and `repo_id` re-uploads
only the files whose content actually changed.

## Pulling a dataset elsewhere

Same three flavors, mirroring the push side.

### From the CLI

```bash
uv run synthpix-hf pull user/piv-dataset-class1-256 ~/data/piv_dataset_class1_256
```

To grab only a subset of splits:

```bash
uv run synthpix-hf pull user/piv-dataset-class1-256 ~/data/piv \
    --splits train,val
```

`--splits` accepts a comma-separated list and is translated into
`("train/**", "val/**", "README.md")` glob patterns. The dataset card always
travels with the data. For more control, `--include PATTERN` (repeatable)
and `--ignore PATTERN` (repeatable) accept raw glob patterns; `--include`
overrides `--splits`.

### Programmatic

```python
from synthpix.hf import pull_dataset

local = pull_dataset(
    repo_id="user/piv-dataset-class1-256",
    local_dir="~/data/piv_dataset_class1_256",
    splits=("train", "val"),  # optional
)
print(local)  # Path object pointing at the resolved local directory
```

`pull_dataset` wraps `huggingface_hub.snapshot_download`, so it is resumable
and idempotent — rerun on a partially downloaded directory and it picks up
where it left off. Symlinks are disabled, so the on-disk tree mirrors the
repo and can be fed directly to the `.mat` schedulers.

### In a Kaggle or Colab notebook

The recipe is the same shape everywhere with internet + Python + an HF token:

```python
# In Kaggle, expose the token via Secrets and read it once.
import os
from kaggle_secrets import UserSecretsClient
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")

# Install and pull.
%pip install --quiet "synthpix[hf]"

from synthpix.hf import pull_dataset
pull_dataset(
    "user/piv-dataset-class1-256",
    "/kaggle/working/piv_dataset_class1_256",
)
```

In Colab, replace the Secrets dance with `from google.colab import userdata`
and the same idea. Anywhere else, just set `HF_TOKEN` in the environment.

## The dataset card

When `card_meta` is passed (or when the CLI receives `--card-source-url` and
`--card-citation`, or when the build scripts run with their default card),
`push_dataset` writes a `README.md` under `local_dir` immediately before the
upload. The card is consumed by the Hub UI and carries provenance,
license/citation, and a footer recording the `synthpix` version and git
commit used to build it.

Field reference (see `synthpix.hf.DatasetCardMeta`):

- `name` — short repository name (required).
- `source_url` — URL of the upstream dataset (required).
- `citation` — BibTeX or free-form citation text (required).
- `license` — SPDX-like identifier; defaults to `"other"`.
- `license_name` — display name; defaults to `"research-only-arr"`.
- `pretty_name` — used in the H1 heading and YAML frontmatter.
- `tags` — list of Hub tags; defaults to `("PIV", "synthetic", "optical-flow")`.
- `synthpix_version`, `synthpix_commit` — auto-filled when left as `None`.

The rendered YAML frontmatter looks like:

```yaml
---
license: other
license_name: research-only-arr
pretty_name: "piv-dataset-class1-256"
tags: ["PIV", "synthetic", "optical-flow", "class-1"]
---
```

To skip card generation entirely:

- CLI: `synthpix-hf push ... --no-card`.
- Build scripts: `--no-push-card`.
- Programmatic: leave `card_meta=None` (the default).

The standalone `synthpix-hf card` subcommand writes a card without pushing,
which is handy for previewing what the README will look like:

```bash
uv run synthpix-hf card ~/data/piv_dataset_class1_256 \
    --source-url https://github.com/shengzesnail/PIV_dataset \
    --citation "@article{cai2019dense, ...}" \
    --output /tmp/preview-README.md
```

## Layout contract

The integration preserves your local directory tree byte-for-byte. The
default include globs are:

```
train/**
val/**
test/**
tune/**
splits/**
README.md
```

The default ignore globs strip the intermediate `raw_*/` and `packed_*/`
working directories produced by the PIV build scripts, plus the usual
`.DS_Store`, `__pycache__/`, and `.git/` noise. A typical pushed repo looks
exactly like its local source:

```
piv_dataset_class1_256/
  README.md
  splits/
    train.txt
    val.txt
    test.txt
    tune.txt
  train/
    backstep/Re1000/<name>.mat
    ...
  val/
  test/
  tune/
```

Pulling that repo back on a different machine reproduces the same tree under
`local_dir`. There is no Parquet, no `datasets` loader, no Hub-specific glue
— the `.mat` files are stored as plain Hub blobs and consumed straight by
the existing schedulers.

## Integration with flowgym

A flowgym dataset YAML typically points its `file_list:` at a directory of
`.mat` files, for example:

```yaml
file_list: ["/home/user/data/piv_dataset_class1_256/train"]
scheduler_class: ".mat"
```

On a fresh machine the workflow is:

```bash
# 1. Install the extra and authenticate.
uv add "synthpix[hf]"
export HF_TOKEN=hf_xxxxxxxx

# 2. Pull the dataset to a stable local path.
uv run synthpix-hf pull user/piv-dataset-class1-256 \
    ~/data/piv_dataset_class1_256
```

```yaml
# 3. Update the dataset YAML to point at the pulled location.
file_list: ["/home/user/data/piv_dataset_class1_256/train"]
scheduler_class: ".mat"
```

That is the entire integration: no extra Python, no `hf://` URI handling, no
custom loader. The layout contract above is what makes this work — the
pulled tree is the same tree your `.mat` scheduler already globs over.

> The YAML snippet above is paraphrased; check your actual flowgym
> experiment YAML for the exact key names and any project-specific overrides.

## Live tests

The repo ships one live, network-touching round-trip test for `push_dataset`.
It is gated behind a custom pytest marker and a CLI flag, so it never runs
by accident:

```bash
HF_TOKEN=hf_xxxxxxxx uv run pytest tests/hf/test_push_live.py \
    -m hf_live --run-hf-live
```

The test creates a unique private repo derived from the authenticated user,
pushes a tiny fixture, pulls it back, asserts byte equality, and deletes the
repo in a finalizer. Contributors do not need to run it locally unless they
are modifying push/pull code paths.

## Troubleshooting

**`ImportError: install with pip install synthpix[hf]`** — the lazy import
of `huggingface_hub` failed. Run `uv sync --extra hf` (or
`pip install synthpix[hf]`).

**HTTP 401/403 from the Hub** — your token is missing scope, or it is a
read-only token. Recreate it with write access on the target repo (or, for
fine-grained tokens, select the dataset repo explicitly).

**`PermissionError: Refusing to create a public HF Hub dataset without
--allow-public`** — that is the safety gate. The upstream PIV class-1 license
forbids public redistribution, so `--allow-public` (or
`allow_public=True`) is required by design. Confirm you actually have
permission to redistribute before flipping it.

**Slow uploads or downloads** — check that `hf_transfer` is installed:

```bash
uv pip list | grep hf-transfer
```

A missing wheel logs a one-line warning the first time `push_dataset` /
`pull_dataset` runs; install the `[hf]` extra to pick it up. Network and
remote shard layout also matter; both endpoints fan out across
`max_workers` (default 8) which you can raise on a fast link.

**Interrupted upload or download** — both `push_dataset` and `pull_dataset`
are idempotent and resumable. Rerun the same command; nothing is wiped and
already-transferred files are skipped.

**`RepositoryNotFoundError` during `--dry-run`** — expected when the repo
does not exist yet. Dry-run treats the remote file set as empty and prints
every local file as new.

## License reminder

The PIV class-1 source from
[shengzesnail/PIV_dataset](https://github.com/shengzesnail/PIV_dataset) is
released **for research only, all rights reserved** by the original authors.
The integration defaults to private hosting for that reason, and the public
safety gate exists to make accidental redistribution impossible.

If you publish work that uses this data, cite the upstream paper:

> Cai, S., Zhou, S., Xu, C., Gao, Q. (2019). *Dense motion estimation of
> particle images via a convolutional neural network.* Experiments in
> Fluids 60, 73.

```bibtex
@article{cai2019dense,
  title={Dense motion estimation of particle images via a convolutional neural network},
  author={Cai, Shengze and Zhou, Shichao and Xu, Chuanqi and Gao, Qi},
  journal={Experiments in Fluids},
  volume={60},
  number={4},
  pages={73},
  year={2019},
  publisher={Springer}
}
```
