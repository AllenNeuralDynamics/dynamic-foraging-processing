# Contributing

This page covers the **mechanics** — environment, linting, tests, commit/PR
conventions, and releasing. For **where in the code to plug in** (adding a trials-table
column, a QC check, or prototyping in a notebook), see
[`docs/architecture/extending.md`](docs/architecture/extending.md).

### Linters and testing

This project uses [uv](https://docs.astral.sh/uv/) to manage its environment and
[ruff](https://docs.astral.sh/ruff/) for linting and formatting.

- Install dependencies (including the `dev` group):

```bash
uv sync
```

- Run tests with **pytest** under **coverage**:

```bash
uv run coverage run -m pytest && uv run coverage report
```

- Use **interrogate** to check that modules, methods, etc. have been documented thoroughly:

```bash
uv run interrogate .
```

- Use **ruff** to lint the code (catches unused imports, style issues, sorts imports, etc.):

```bash
uv run ruff check .
```

- Use **ruff** to automatically format the code:

```bash
uv run ruff format .
```

- To apply ruff's autofixes in one shot:

```bash
uv run ruff check --fix . && uv run ruff format .
```

### Pull requests

For internal members, please create a branch. For external members, please fork the repository and open a pull request from the fork. We'll primarily use [Angular](https://github.com/angular/angular/blob/main/CONTRIBUTING.md#commit) style for commit messages. Roughly, they should follow the pattern:
```text
<type>(<scope>): <short summary>
```

where scope (optional) describes the packages affected by the code changes and type (mandatory) is one of:

- **build**: Changes that affect build tools or external dependencies (example scopes: pyproject.toml, setup.py)
- **ci**: Changes to our CI configuration files and scripts (examples: .github/workflows/ci.yml)
- **docs**: Documentation only changes
- **feat**: A new feature
- **fix**: A bugfix
- **perf**: A code change that improves performance
- **refactor**: A code change that neither fixes a bug nor adds a feature
- **test**: Adding missing tests or correcting existing tests

### Semantic Release

The table below, from [semantic release](https://github.com/semantic-release/semantic-release), shows which commit message gets you which release type when `semantic-release` runs (using the default configuration):

| Commit message                                                                                                                                                                                   | Release type                                                                                                    |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------- |
| `fix(pencil): stop graphite breaking when too much pressure applied`                                                                                                                             | ~~Patch~~ Fix Release, Default release                                                                          |
| `feat(pencil): add 'graphiteWidth' option`                                                                                                                                                       | ~~Minor~~ Feature Release                                                                                       |
| `perf(pencil): remove graphiteWidth option`<br><br>`BREAKING CHANGE: The graphiteWidth option has been removed.`<br>`The default graphite width of 10mm is always used for performance reasons.` | ~~Major~~ Breaking Release <br /> (Note that the `BREAKING CHANGE: ` token must be in the footer of the commit and need to be all caps as shown) |

### Documentation
To generate the rst files source files for documentation, run
```bash
sphinx-apidoc -o docs/source/ src
```
Then to create the documentation HTML files, run
```bash
sphinx-build -b html docs/source/ docs/build/html
```
More info on sphinx installation can be found [here](https://www.sphinx-doc.org/en/master/usage/installation.html).

### Read the Docs Deployment
Note: Private repositories require **Read the Docs for Business** account. The following instructions are for a public repo.

The following are required to import and build documentations on *Read the Docs*:
- A *Read the Docs* user account connected to Github. See [here](https://docs.readthedocs.com/platform/stable/guides/connecting-git-account.html) for more details.
- *Read the Docs* needs elevated permissions to perform certain operations that ensure that the workflow is as smooth as possible, like installing webhooks. If you are not the owner of the repo, you may have to request elevated permissions from the owner/admin.
- A **.readthedocs.yaml** file in the root directory of the repo. Here is a basic template:
```yaml
# Read the Docs configuration file
# See https://docs.readthedocs.io/en/stable/config-file/v2.html for details

# Required
version: 2

# Set the OS, Python version, and other tools you might need
build:
  os: ubuntu-24.04
  tools:
    python: "3.13"

# Path to a Sphinx configuration file.
sphinx:
  configuration: docs/source/conf.py

# Declare the Python requirements required to build your documentation
python:
  install:
    - method: pip
      path: .
      extra_requirements:
        - dev
```

Here are the steps for building docs in *Read the Docs*. See [here](https://docs.readthedocs.com/platform/stable/intro/add-project.html) for detailed instructions:
- From *Read the Docs* dashboard, click on **Add project**.
- For automatic configuration, select **Configure automatically** and type the name of the repo. A repo with public visibility should appear as you type.
- Follow the subsequent steps.
- For manual configuration, select **Configure manually** and follow the subsequent steps

Once a project is created successfully, you will be able to configure/modify the project's settings; such as **Default version**, **Default branch** etc.
