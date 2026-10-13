---
name: test
description: Run the HighDream pytest suite (pure-logic regression tests under tests/) from the project virtualenv and report pass/fail. Use after editing logic that tests/ covers (e.g. utils/expr_type.py) or before committing such changes.
---

# /test — run the pytest baseline

## When to run

Run after editing code covered by `tests/` (currently `utils/expr_type.py` and the documented SymPy gotchas). For end-to-end server behavior, use `/verify` instead.

## Steps

1. From the repo root, run the venv's pytest:

   ```
   .venv/Scripts/python.exe -m pytest
   ```

   `pytest.ini` sets `testpaths = tests`. If the venv launcher is missing, fall back to `python -m pytest`.
2. If collection fails with `ModuleNotFoundError: pytest`, tell the user to run `.venv/Scripts/python.exe -m pip install pytest` and stop.
3. Read the summary line, then report.

## Reporting

- Print one line: `test: PASS (<n> passed)` or `test: FAIL — <which tests failed + the assertion>`.
- On failure, paste the relevant assertion output so the user can debug.
- Do NOT report PASS on a collection error or when zero tests ran.

## Notes

- `tests/conftest.py` puts the repo root on `sys.path`, so tests import `utils.*`, `core.*`, `domains.*` directly.
- Keep tests deterministic and fast — no network, no running server, pure-logic only.
