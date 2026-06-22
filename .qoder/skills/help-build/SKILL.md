---
name: help-build
description: Build /help/* HTML pages from their Markdown sources under help_docs/src/ using pandoc. Run after editing any help_docs/src/*.md file (or after cloning the repo, since the generated HTML is also tracked). Use whenever the user says "build the help docs", "rebuild /help", or asks to regenerate help HTML.
---

# /help-build — regenerate help HTML from Markdown sources

## What this does

The `/help/<page>` pages served by Flask are rendered from
`templates/help/*.html`. Those HTML files are **generated** from Markdown
sources under `help_docs/src/` by pandoc via `help_docs/build_help.py`.
This skill runs that build so the served pages reflect the latest `.md`.

## Prerequisites

- `pandoc >= 3` must be on `PATH` (already installed on this machine).
- `beautifulsoup4` only needed for the one-shot `md` (reverse-import)
  command; already installed in the project env.

## Steps

1. **Confirm the user wants a build.** If the request was ambiguous (e.g.
   "refresh the docs"), ask which command they need before running
   anything. Most of the time they want the default `html` build.

2. **Run the build from the repo root** (do NOT `cd` into `help_docs/`):

   ```bash
   python help_docs/build_help.py html
   ```

   Expected output:

   ```
   [ok] base_calculator.md -> templates\help\base_calculator.html
   [ok] base_step_generator.md -> templates\help\base_step_generator.html
   [ok] common_matrix_calculator.md -> templates\help\common_matrix_calculator.html
   [ok] matrix_step_generator.md -> templates\help\matrix_step_generator.html
   [ok] rule_registry.md -> templates\help\rule_registry.html
   ```

   If any `[error]` line appears, read the failing `.md` source, look for
   YAML frontmatter issues or malformed markdown, fix, and re-run.

3. **Verify.** Run `git --no-pager diff --stat templates/help/` to confirm
   only expected files changed. A huge diff after a small `.md` edit
   usually means a structural change in the pipeline — stop and report.

4. **Report.** One line, e.g.:

   ```
   help-build: rebuilt 5 pages; N files changed in templates/help/.
   ```

   Tell the user they can commit the `.md` + `.html` pair together (both
   are tracked in git, see `.gitignore` comment).

## Other commands (rare)

- `python help_docs/build_help.py md` — reverse-import `templates/help/*.html`
  back into `help_docs/src/*.md`. **Only** use this if the user explicitly
  wants to overwrite `.md` from the HTML on disk (e.g. to recover from a
  botched edit). It will clobber uncommitted `.md` changes — ask first.
- `python help_docs/build_help.py all` — `md` then `html`. Almost never
  needed in day-to-day editing.

## When NOT to run

- Pure `.md` preview / discussion — just read the source, don't rebuild.
- Changes to `static/help.css` — CSS is read at runtime by Flask, no
  rebuild needed.
- Changes outside `help_docs/` — unrelated.
