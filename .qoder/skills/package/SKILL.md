---
name: package
description: Build the HighDream Windows executable via PyInstaller and verify dist/HighDreamWeb.exe was produced. Use when the user asks to "package", "build the exe", or "release".
disable-model-invocation: true
---

# /package — build dist/HighDreamWeb.exe

## When to run

Only when the user explicitly asks for a packaged executable. Do NOT invoke automatically.

## Steps

1. Confirm the current working directory is the repo root and `app_local.spec` exists.
2. Ensure the virtual env is active (`.venv` should already be in use — check `python --version` matches 3.11.x and `pip show pyinstaller` succeeds). If PyInstaller is missing, tell the user to `pip install pyinstaller` and stop.
3. Run: `pyinstaller app_local.spec --noconfirm`.
4. After it finishes, check that `dist/HighDreamWeb.exe` exists and is non-empty.
5. (Optional) Launch `dist/HighDreamWeb.exe` briefly and verify it opens the browser to `http://127.0.0.1:5000/`. Kill the process after confirming startup.

## Reporting

- Print: `package: PASS — dist/HighDreamWeb.exe (<size>)` or `package: FAIL — <reason>`.
- On failure, paste the last ~30 lines of PyInstaller output.

## Notes from the spec

- `optimize` is pinned to `1`. Do not raise it to `2` — it breaks SymPy at runtime.
- `matplotlib`, `IPython`, `jupyter`, `tkinter` are excluded from the exe. If you added a dependency the exe uses, add it to `hiddenimports` or remove it from `excludes` in `app_local.spec` before rebuilding.
