# AGENTS.md

This file provides guidance to the AI agent when working with code in this repository.

## Running the app

- **Local (interactive, opens a browser at http://127.0.0.1:5000/):** `python app_local.py` — uses `gevent.pywsgi.WSGIServer`. Don't close the terminal window while it's running.
- **Vercel-style entry (used by the deployed site):** `python app.py` — plain `Flask.run()`.
- **Build a Windows executable:** `pyinstaller app_local.spec` → produces `dist/HighDreamWeb.exe`. Do NOT change `optimize=1` in the spec to `2`; it breaks SymPy.

## SymPy gotchas (from the user-facing README — enforce in generated examples and tests)

- Multiplication `*` is **never** implicit: `ab` parses as one symbol, not `a*b`.
- Inverse trig is `asin`, `acos`, `atan` — NOT `arcsin`/`arccos`/`arctan`.
- Natural exponential is `exp()`. `e` alone is just a symbol like `x`; `e**x` and `e^x` will not be treated as `exp(x)`.

## Project layout (quick reference — read the files for details)

- `core/` — the BFS + rule-driven solver engine. See `BaseCalculator`, `BaseStepGenerator`, `RuleRegistry`, `CommonMatrixCalculator`, `MatrixStepGenerator`.
- `domains/` — per-domain rule sets: `differentiation/`, `integral/`, `limit/`, `matrix/`, plus `expression_parser.py`.
- `routes/` — Flask blueprints: `main.py` (pages), `api.py` (compute endpoints, async via `task_manager.py`).
- `utils/` — LaTeX rendering and expression helpers.
- `templates/`, `static/` — web UI. `static/trees/` holds generated equivalence-derivation tree images.
- `help_docs/` — `/help/*` 帮助文档源文件与 pandoc 构建脚本。`help_docs/src/*.md` 是内容源；`templates/help/*.html` 由 pandoc 生成（详见 `help_docs/README.md`）。

## Help documentation (`/help/*`)

Content for the `/help/<page>` routes is authored as Markdown under
`help_docs/src/`. The HTML Flask serves (`templates/help/<page>.html`) is
**generated** from those `.md` files by pandoc via `help_docs/build_help.py`.

- After editing a `.md` source, rebuild with `python help_docs/build_help.py html`.
- Both `.md` and the generated `.html` are tracked in git (Vercel deploys the rendered pages directly); commit the pair together.
- Never hand-edit `templates/help/*.html` — changes will be clobbered on the next build.
- Two Qoder skills exist for this: `/help-build` (run the pandoc build) and `/help-new` (add / edit / delete a help page end-to-end, including the Flask route and the `SOURCE_TO_HTML` map in `help_docs/build_help.py`).

## Documentation

The project has **two** top-level READMEs that should stay in sync on content changes: `README.md` (Chinese, primary) and `README-EN.md` (English). When changing user-facing behavior, update both.

## Commit messages

Use short **Chinese** commit messages, matching the existing log (e.g. `更新文档`, `添加在墨干中调用的示例`, `Fix md`).

## Testing

There is **no test directory and no test runner**. Before marking a behavior change as done, verify it manually — either via `main.ipynb` (Python calls) or by running `python app_local.py` and exercising the web UI. Prefer adding a test file under a new `tests/` directory if you are asked to add regression coverage; use `pytest`.

## Lint

- Pylint is the configured linter: `.pylintrc` disables many rules (line-too-long, missing docstrings, too-many-*, etc.). Run `pylint <file>` — do not add a new linter without asking.
- No auto-formatter is configured. Match the surrounding 4-space, PEP-8-ish style of the file you're editing.
