# Release Checklist

VecAdvisor uses GitHub releases for alpha artifacts and a manual PyPI
publishing workflow for package distribution.

## Current Public Alpha

- GitHub release: `v0.1.0a3`
- PyPI package: `vecadvisor==0.1.0a3`
- Package version: `0.1.0a3`
- Artifacts: wheel and source distribution attached to the release
- CI gate: Python 3.11 and Python 3.12

## PyPI Trusted Publishing Setup

The repository publishes through `.github/workflows/publish.yml`. The first
Trusted Publishing setup has already been completed for `vecadvisor`.

For a new package or if the publisher is recreated:

1. Create or reserve the `vecadvisor` project on PyPI.
2. Add a Trusted Publisher for:
   - Owner: `vecadvisor`
   - Repository: `vecadvisor`
   - Workflow: `publish.yml`
   - Environment: `pypi`
3. In GitHub, confirm the `pypi` environment exists if environment approvals
   are desired.
4. Run the manual `Publish` workflow with `confirm` set to `publish`.
5. Verify:

```bash
python -m pip install --upgrade vecadvisor
vecadvisor --help
```

## Release Steps

1. Update `pyproject.toml` and `src/vecadvisor/__init__.py`.
2. Run:

```bash
python -m ruff check .
python -m mypy src/vecadvisor
python -m pytest
python -m build
```

3. Update benchmark evidence if claims changed.
4. Commit, push, and wait for CI.
5. Tag the release:

```bash
git tag -a vX.Y.Z -m "VecAdvisor vX.Y.Z"
git push origin vX.Y.Z
```

6. Create a GitHub release and attach `dist/*.whl` plus `dist/*.tar.gz`.
7. Publish to PyPI only after the GitHub release and CI are green.
8. Verify the PyPI endpoint and clean install:

```bash
python -m pip install --upgrade vecadvisor
python - <<'PY'
import vecadvisor
print(vecadvisor.__version__)
PY
```

## Guardrails

- Do not publish benchmark claims without reproducible commands and artifacts.
- Keep PyPI publishing manual until release discipline is proven.
- Keep alpha/beta versions marked with PEP 440 prerelease versions.
- Do not ship private design notes or local benchmark data by accident.
