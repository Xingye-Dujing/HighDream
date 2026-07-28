# AGENTS.md

Guidance for AI agents working in this repository.

## Running the app

- **Local dev (opens http://127.0.0.1:5000/):** `python app_local.py` — uses `gevent.pywsgi.WSGIServer`, opens browser automatically.
- **Vercel entrypoint:** `python app.py` — plain `Flask.run()` (for local testing of deploy path).
- **Build executable:** `pyinstaller app_local.spec` → `dist/HighDreamWeb.exe`. Do NOT set `optimize=2`; breaks SymPy.
- **Dependencies:** `requirements_local.txt` for dev, `requirements_deployment.txt` for Vercel (just `sympy`, `flask`).

## SymPy gotchas (enforce in generated code/examples)

- `*` is never implicit: `ab` parses as one symbol, not `a*b`.
- Inverse trig is `asin`, `acos`, `atan` — NOT `arcsin`/`arccos`/`arctan`.
- Natural exponential is `exp()`. `e` alone is just a symbol; `e**x` / `e^x` ≠ `exp(x)`.

## Architecture

- `core/` — BFS + rule-driven solver engine: `BaseCalculator`, `BaseStepGenerator`, `RuleRegistry`, `CommonMatrixCalculator`, `MatrixStepGenerator`, `ManualStepSolver`, `MethodTreeEnumerator`.
- `domains/` — per-domain rule sets: `differentiation/`, `integral/`, `limit/`, `matrix/`, plus `expression_parser.py`.
- `routes/` — Flask blueprints: `main.py` (pages), `api.py` (compute endpoints). Async tasks via `task_manager.py` (in-memory `ThreadPoolExecutor`). Compute dispatch in `compute_service.py` (`_calculators` dict maps operation_type → calculator class).
- `utils/` — LaTeX rendering (`latex_formatter.py`) and expression helpers.
- `config.py` — central tunables: parser depth, L'Hôpital count, method-tree cutoffs (defaults and hard limits), task timeouts.
- `app_local.py` / `app.py` both call `create_app()` which registers `main` and `api` blueprints.

## Lint & auto-format

- **Pylint** only (`.pylintrc` disables many pedantic rules). Run `pylint <file>`.
- A PostToolUse hook in `.qoder/settings.json` auto-runs `pylint` on every `.py` file after editing.
- No auto-formatter. Match existing 4-space PEP-8-ish style.

## Help documentation (`/help/*`)

- **Author in:** `help_docs/src/*.md`
- **Generate HTML:** `python help_docs/build_help.py html` (requires pandoc ≥3 on `PATH`)
- **Never hand-edit** `templates/help/*.html` — changes are clobbered on next build.
- Both `.md` and generated `.html` are git-tracked (Vercel serves the HTML directly); commit the pair together.
- Existing `.qoder/skills/help-build` and `help-new` contain the Qoder-specific workflows for this.

## Docs

Update **both** `README.md` (Chinese, primary) and `README-EN.md` (English) on user-facing changes.

## Testing

- No test directory or runner exists. Verify manually: `main.ipynb` (Python calls) or `python app_local.py` + web UI.
- If adding regression coverage, create `tests/` and use `pytest`.

## Commit messages

Short **Chinese**, matching existing log (e.g. `更新文档`, `Fix md`).

## Code style

- **Reuse** existing functions and patterns; add new ones only when existing utilities cannot be adapted.
- Prefer **loose coupling** — modules should depend on abstractions (`BaseCalculator`, `BaseStepGenerator`), not on concrete implementations.
- **Keep extension points:** when adding a new domain or operation, follow the existing registration pattern (`_calculators` dict, rule registry, step generator).
- Avoid redundant helper layers; a function with a single caller in the same module is usually unnecessary.

## Keys an agent would miss

- **Manual step solver sessions** (`/rule_select` page) are in-memory dicts in `api.py` (`_manual_sessions`) — lost on server restart.
- **Method tree API** is async: POST to `/api/method_tree_start`, poll `/api/method_tree_status`, cancel via `/api/method_tree_cancel`, interactive decisions via `/api/method_tree_respond`.
- **Calculator registry:** adding a new operation type requires registering it in `compute_service.py` `_calculators` dict and adding the route in `api.py`.
- **Result cache DB:** SQLite cache at `cache/computation.db`, managed by `utils/db_cache.py`. Hash key via SHA256 of (op_type + expression + variable/point/direction). Per-cell toggles ("缓存"/"记录") in the frontend send `use_cache` / `record_cache` flags through the compute payload. Cached calculator variants (`CachedDiffCalculator`, `CachedIntegralCalculator`, `CachedLimitCalculator`) use the `DBCachedCalculator` mixin in `core/cached_calculator.py`; switching is handled in `compute_service.py` based on the flags.
- **No CI/CD** — no `.github/`, no `vercel.json`. Deployment is manual via Vercel.
