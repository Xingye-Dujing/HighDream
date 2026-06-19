---
name: verify
description: Run the HighDream web app locally and smoke-test that /api/parse responds correctly, then shut the server down. Use after making changes to routes/, core/, domains/, or the Flask entry points to confirm the app still starts and serves math results.
---

# /verify — smoke-test the running app

## When to run

Run this skill after editing:
- `app.py`, `app_local.py`, `config.py`
- Anything under `routes/`
- Anything under `core/` or `domains/` that the web UI calls into

Skip it for pure doc / README / `.pylintrc` / spec-file edits.

## Steps

1. Make sure port 5000 is free. If not, report the blocker and stop.
2. Launch `python app_local.py` in the background from the project root.
3. Wait up to ~10 seconds for the server to be ready (loop with a short sleep; curl `/` or `/api/parse` until it responds or timeout).
4. Smoke test: POST a known-good expression to `/api/parse`, e.g.

   ```
   curl -X POST http://127.0.0.1:5000/api/parse \
     -H "Content-Type: application/json" \
     -d '{"expression": "sin(x)"}'
   ```

   - If the response is `200` and contains a LaTeX string → PASS.
   - If the response is `5xx` or the body is an error JSON → FAIL. Include the traceback.

5. (Optional) If the change touched a specific domain (e.g. `domains/integral/`), also POST a representative expression for that domain to `/api/compute` and poll `/api/task_status` until done or `config.SINGLE_TASK_EXECUTE_TIMEOUT_SECONDS` is hit.

6. Kill the background `python app_local.py` process. Always clean up, even on failure.

## Reporting

- Print one line: `verify: PASS` or `verify: FAIL — <short reason>`.
- On failure, paste the relevant traceback / error JSON so the user can debug.
- Do NOT claim success if the server didn't respond within the timeout — say so explicitly.
