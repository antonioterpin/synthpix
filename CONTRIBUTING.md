# Contributing
This page summarizes the development workflow and software engineering practices adopted in this codebase. This project is written in Python 3.10+.

- [Development workflow](#development-workflow)
- [Testing a feature](#testing-a-feature)
- [Preparing for a PR](#preparing-for-a-pr)
### 📁 Project Structure

```
├── .github/
│   ├── workflows/          # CI/CD pipelines
│   ├── ISSUE_TEMPLATE/     # Issue templates
│   └── pull_request_template.md
├── src/                    # Source code
├── tests/                  # Test suite
├── docs/                   # Documentation
├── scripts/                # Utils, like custom hooks
├── .pre-commit-config.yaml # Code quality hooks
├── pyproject.toml          # Project configuration
└── CONTRIBUTING.md         # Development guidelines
```

We use [uv](https://docs.astral.sh/) to manage the virtual environment and dependencies.
If you do not have it yet, install it with:
```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

To provision the local environment with development tools:
```sh
uv sync --extra dev
```

Run the application with (see the [README.md](README.md) for details on the usage):
```sh
uv run examples/<the-example>.py
```
To run the pytests:
```sh
uv run pytest
```

## Coding style
The `Coding style validation` action will fail if the pre-commit checks do not pass. To make sure that these are checked automatically on push, run:
```sh
pre-commit install --hook-type pre-commit
```

Our pre-commit configuration includes several automated checks:

- **Black**: Code formatting
- **Ruff**: Linting, docstring checks, modern typing enforcement, and unused import detection
- **PyUpgrade**: Automatic modernization of Python syntax (Python 3.10+)
- **Standard hooks**: YAML validation, trailing whitespace, end-of-file fixes

To run the pre-commit checks on specific files:
```bash
pre-commit run --files <files>
```
Or all files:
```bash
pre-commit run --all-files
```

If for some reason you really want to ignore them during commit/push, add `--no-verify`.

Our coding style is inspired to the [Linux kernel coding style](https://www.kernel.org/doc/html/v4.10/process/coding-style.html), to the [Python Google coding style](https://google.github.io/styleguide/pyguide.html), and to the [JAX framework](https://docs.jax.dev). We elaborate on some key elements below. Please read both of the above documents, and prioritize the elements described below (sometimes we simply copy paste from the above guides to emphasize a principle).

### General principles
1. Clarity:
   - The code should basically read as sentences. This makes it beautiful. Some examples of choices:
        - Prefer boolean namings so that the positive case comes first in conditional statements: `if it_rains: ...`'
        - Give meaningful names, but keep them concise. Avoid unnecessary specifiers in names.
   - Indentation: 4 characters. Avoid more than 3 indentation levels.
   - Short lines: 80 characters max. Do not cheat with linebreaks "\".
   - Functions should be short and specialized. The maximum length of a function is inversely proportional to the complexity and indentation level of that function. If a function implemented to use inside a public API offers functionalities that are not needed across the repo, make it private. If they are useful across the repo, put them in utils. If the implemented functionality is useful across repos (e.g., observability utils) and they match the scope of [goggles](https://github.com/antonioterpin/goggles), open a PR there.
   - Spacing: Do not add spaces around (inside) parenthesized expressions. This example is bad: `my_fun( param1 )`. Use one space around (on each side of) most binary and ternary operators, such as any of these: `=  +  -  <  >  *  /  %  |  &  ^  <=  >=  ==  !=  ?  :`. In Python, `:` is also used after class/functions definition. In that case, do not put spaces.
   - Naming: Use `snake_case` for functions and variables, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants.
   - Default to `dataclasses.dataclass` for simple record-like structures. Prefer `frozen=True` dataclasses.
   - We use [goggles](https://github.com/antonioterpin/goggles) for logging. Avoid plain `print` statements or using the default logging library. Please read carefully the [goggles](https://github.com/antonioterpin/goggles) documentation.
   - Always use type hints. See also [the docstrings section](#docstrings-and-documentation). For tensors, specify the shape.
   - Use [Object Oriented Programming](https://www.geeksforgeeks.org/java/understanding-encapsulation-inheritance-polymorphism-abstraction-in-oops/) principles, but **keep the state separate from the instance!** That is, keep the methods pure and without side effects, and delegate the state of an instance to a separate object passed as argument.
   Example:
   ```python
   from dataclasses import dataclass, replace

   @dataclass(frozen=True)
   class RobotState:
    """Immutable state container for robot data."""
    position: tuple[float, float]
    velocity: tuple[float, float]
    battery_level: float = 100.0

    def update(self, **kwargs) -> 'RobotState':
        """Create new state with updated fields.

        Args:
            **kwargs: Fields to update

        Returns:
            RobotState: New state instance with updates
        """
        result = replace(self, **kwargs)
        return result

   class RobotController:
    """Example of pure methods and external state."""

    def calculate_next_position(
        self,
        state: RobotState,
        time_delta: float
    ) -> tuple[float, float]:
        """Calculate next position.

        Args:
            state (RobotState): Current robot state
            time_delta (float): Time step in seconds

        Returns:
            tuple[float, float]: Next position coordinates
        """
        # Single calculation path with single return
        new_x = state.position[0] + state.velocity[0] * time_delta
        new_y = state.position[1] + state.velocity[1] * time_delta
        result = (new_x, new_y)
        return result

    def update_robot_state(
        self,
        state: RobotState,
        time_delta: float
    ) -> RobotState:
        """Update robot state with new position and battery drain.

        Args:
            state (RobotState): Current robot state
            time_delta (float): Time step in seconds

        Returns:
            RobotState: New state with updated position and battery
        """
        next_position = self.calculate_next_position(state, time_delta)
        battery_drain = time_delta * 0.1
        new_battery = max(0.0, state.battery_level - battery_drain)

        result = state.update(
            position=next_position,
            battery_level=new_battery
        )
        return result
   ```
   - Have a single return for each function, at the end.
   - Before implementing a functionality, always check that it does not exist yet (in this repo or in goggles) and that you cannot obtaining with a simple modification of an existing one (ensuring compatibility or with a refactor).
2. Performance:
   - Write clarity-first code. Optimize **only** after profiling.
   - Prefer vectorized JAX operations. Avoid Python loops when possible.
   - Use `jax.jit` and `jax.vmap` judiciously. Keep functions testable both with and without jit. When writing tests, effectively check that functions that should be jitted are jitted effectively.
   - Add tests to measure the speed of functions. This is particularly important for jitted functions. For this, use [timeit](https://docs.python.org/3/library/timeit.html).

### Type hints and modern typing practices
We use modern Python typing patterns following PEP 585, PEP 604, and current best practices:

#### Modern type imports (Python 3.10+):
```python
# ✅ PREFERRED - Modern typing
from typing import Any, ClassVar, Literal  # No builtin replacement
from typing_extensions import TypeAlias
from collections.abc import Mapping, Sequence  # PEP 585 generics

# ✅ PREFERRED - Builtin generics (Python 3.9+)
def process_data(items: list[str]) -> dict[str, int]:
    """Use builtin list, dict, tuple, set, frozenset."""
    result: dict[str, int] = {}
    return result

# ✅ PREFERRED - Union syntax (Python 3.10+)
def handle_value(x: int | float | None) -> str | None:
    """Use | instead of Union and Optional."""
    if x is None:
        return None
    return str(x)

# ❌ AVOID - Legacy typing (pre-PEP 585)
from typing import List, Dict, Tuple, Set, FrozenSet, Union, Optional
def old_style(items: List[str]) -> Optional[Dict[str, int]]:
    pass
```

#### Type alias definitions:
```python
# ✅ PREFERRED - Modern type aliases
from typing_extensions import TypeAlias

UserId: TypeAlias = int
UserData: TypeAlias = dict[str, Any]
Coordinates: TypeAlias = tuple[float, float]

# For JAX arrays (forward references to avoid import issues)
Array: TypeAlias = "jax.Array"
PRNGKey: TypeAlias = "jax.Array"
```

#### Specific guidelines:
- **Any**: Still use `typing.Any` (no builtin replacement)
- **ClassVar**: Still use `typing.ClassVar` (no builtin replacement)
- **Literal**: Still use `typing.Literal` (no builtin replacement)
- **FrozenSet**: Use `frozenset[T]` instead of `typing.FrozenSet[T]`
- **Mapping/Sequence**: Use `collections.abc.Mapping[K, V]` and `collections.abc.Sequence[T]`
- **Optional[T]**: Use `T | None` instead of `typing.Optional[T]`
- **Union**: Use `T | U` instead of `typing.Union[T, U]`
- **List/Dict/Tuple**: Use builtin `list[T]`, `dict[K, V]`, `tuple[T, ...]`

### Docstrings and documentation
We adopt the [Google style docstrings](https://google.github.io/styleguide/pyguide.html#383-functions-and-methods), but always keep type information both in the signature and in the docstring sections (`Args`, `Returns`, `Raises`) so that editors get accurate autocomplete while readers can rely on the rendered documentation.

Principles:
- Keep docstrings focused on what the callable does, its inputs, outputs, and side effects. Provide only the smallest example necessary to clarify ambiguous behavior.
- Avoid mirroring high-level guidance, design principles, or extensive how-to material in docstrings; redirect that content to the project documentation instead of duplicating it in code.
- When functionality relies on domain concepts, reference the documentation section that covers them rather than restating the underlying principles inline.
- Do NOT explain HOW your code works in a comment. Make sure the code is not obscure and describe WHAT it is doing.

## Development workflow
We follow a [Git feature branch](https://www.atlassian.com/git/tutorials/comparing-workflows/feature-branch-workflow) workflow with test-driven-development. Prefer many, small PRs over single, large ones.

Principles:
- The basic workflow is as follows:
  1. Open an issue for the feature to implement, and describe in detail the goal of the feature. Describe the tests that should pass for the feature to be considered implemented.
  2. Open a branch from `dev` for the feature:
    ```bash
    git checkout dev
    git checkout -b feature-<issue-number>
    ```
  3. Add the tests; see [Testing](#testing-a-feature).
  4. Implement the feature and make sure the tests pass.
  5. Open a PR to the `dev` branch. Note that the PR requires to `squash` the commit. See [Preparing for a PR](#preparing-for-a-pr).
  6. Close the branch.

- `main` and `dev` branches are protected from push, and require a PR.
- We run github actions, [code-style](https://github.com/antonioterpin/pinet/blob/main/.github/workflows/code-style.yaml) and [tests](https://github.com/antonioterpin/pinet/blob/main/.github/workflows/tests.yaml) to check the test status on push on any branch. The rationale is that we want to know the state of each feature without polling the developer.
- We open a PR to `main` only for milestones.

### Testing a feature
To test a new feature, simply add a `test_<feature_to_test>` inside the folder `src/test`. For this, refer to the [`pytest` documentation](https://docs.pytest.org/en/stable/).

To run the tests,
```bash
uv run pytest -q --cov=src/my_package --cov-report=term-missing
```

Principles:
- All public functions must have tests. Better if all functions have tests.
- We measure coverage, but it should not be an objective. It is a good metric until someone starts to game it.
- Tests should be written BEFORE the feature. That is, we follow a [Test-Driven-Development](https://en.wikipedia.org/wiki/Test-driven_development) pattern.
- Divide integration tests from unit tests.
- Use the [Arrange/Act/Assert](https://semaphore.io/blog/aaa-pattern-test-automation) pattern:
  - Arrange: Set up the test environment.
  - Act: Execute the code to test.
  - Assert: Verify the results.
- Prefer [property-based tests](https://semaphore.io/blog/property-based-testing-python-hypothesis-pytest) where appropriate (hypothesis).

### Preparing for a PR
Before opening a PR to `dev`, you need to `squash` your commits into a single one. First, review your commit history to identify how many commits need to be squashed:
```bash
git log --oneline
```
For example, you may get
```bash
abc123 Feature added A
def456 Fix for bug in feature A
ghi789 Update documentation for feature A
```
Suppose you want to squash the three above into a single commit, `Implement feature <issue-number>`. You can rebase interactively to squash the commits:
```bash
git rebase -i HEAD~<number-of-commits>
```
For example, if you want to squash the last 3 commits:
```bash
git rebase -i HEAD~3
```
An editor will open, showing a list of commits:
```bash
pick abc123 Feature added A
pick def456 Fix for bug in feature A
pick ghi789 Update documentation for feature A
```
- Keep the first commit as `pick`.
- Change `pick` to `squash` (or `s`) for the subsequent commits:
```bash
pick abc123 Feature added A
squash def456 Fix for bug in feature A
squash ghi789 Update documentation for feature A
```
Save and close the editor.
Git will prompt you to edit the combined commit message. You’ll see:
```bash
# This is a combination of 3 commits.
# The first commit's message is:
Feature added A

# The following commit messages will also be included:
Fix for bug in feature A
Update documentation for feature A
```
Edit it into a single meaningful message, like:
```bash
Add feature A with bug fixes and documentation updates
```
Save and close the editor; Git will squash the commits. If there are conflicts during the rebase, resolve them and continue:
```bash
git rebase --continue
```
Verify the commit history:
```bash
git log --oneline
```
You should see one clean commit instead of multiple. If you’ve already pushed the branch to a remote repository, you need to force-push after squashing:
```bash
git push --force
```
Now that the feature branch has a clean history, create the PR from your feature branch to the main branch. The reviewers will see a single, concise commit summarizing your changes. See the [guidelines for commit messages](https://www.conventionalcommits.org/en/v1.0.0/#summary).

## Commit style
To ease writing commit messages that conform to the [standard](https://www.conventionalcommits.org/en/v1.0.0/#summary), you can configure the template with:
```bash
git config commit.template .gitmessage
```
To fill in the template, run
```bash
git commit
```
When you have edited the commit, press `Esc` and then type `:wq` to save. In `Visual Studio Code`, you should setup the editor with
```bash
git config core.editor "code --wait"
```
You may need to [setup the `code` command](https://code.visualstudio.com/docs/setup/mac).
The `Commit style validation` action will fail if you do not adhere to the recommended style.

Tip: When something fails, fix the issue and use:
```bash
git commit --amend
git push --force
```
