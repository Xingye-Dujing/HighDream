---
name: help-new
description: Add, edit, or remove a /help/* documentation page. Handles the Markdown source, the Flask route, the SOURCE_TO_HTML mapping in help_docs/build_help.py, and the pandoc rebuild. Use when the user says "add a help page for X", "create a new /help/X page", "delete the X help page", or wants to change the content of an existing help page.
---

# /help-new — manage /help/* documentation pages

The `/help/*` pages are Markdown-driven. The source of truth for content is
`help_docs/src/<name>.md`; the HTML Flask serves (`templates/help/<name>.html`)
is built from it by pandoc via `help_docs/build_help.py`.

## File layout

```
help_docs/src/<name>.md          # Markdown source (edit THIS)
help_docs/template.html          # pandoc wrapper template (do not touch)
help_docs/build_help.py          # build script, holds SOURCE_TO_HTML map
templates/help/<name>.html       # generated HTML (do not hand-edit)
routes/main.py                   # Flask route @main.route('/help/<name>')
static/help.css                  # shared CSS (classes below)
```

## CSS classes available in `static/help.css`

Use these in the Markdown source; pandoc + the build script preserve them:

| Class | Markdown idiom | Purpose |
|---|---|---|
| `warning` | `<div class="warning">…</div>` | Yellow "注意" callout box |
| `core-area` | `<div class="core-area">…</div>` | Grey code-area box |
| `core-class` | `<div class="core-class">…</div>` | Link list (used by `help.html` only) |
| `method-signature` | fenced code block `` ``` `` | Python-style method signature |
| `parameter-table` | `<div class="parameter-table">` wrapped around a GFM table | Styled parameter table |

## Add a new page

1. **Pick a slug** — lowercase, snake_case, matching the core class or topic
   name (e.g. `diff_calculator`, `integration_by_parts`).

2. **Create** `help_docs/src/<slug>.md` with this skeleton:

   ```markdown
   ---
   title: "<ClassName> - <一句话中文描述>文档"
   output: "<slug>.html"
   ---

   # **<ClassName>** (<module>/<file>.py)

   简短的一段介绍。

   <div class="warning">

   **注意:**   出现的所有图中，箭头的方向无实际意义，上面文字所指可能是前对后，也可能是后对前

   </div>

   ## 一.   类概述

   介绍内容。

   - **要点 1**：说明
   - **要点 2**：说明

   ## 二.   属性

   <div class="parameter-table">

   | 属性名 | 类型 | 描述 | 访问级别 |
   |----|----|----|----|
   | `_foo` | `Foo` | 作用 | 私有 |

   </div>

   ## 三.   方法

   ### 1.   初始化

   ```
   __init__(self) -> None
   ```

   描述。

   ```
   _bar(self, x: Expr) -> Expr
   ```

   描述。
   ```

   Use `\xa0` (literal NBSP, not `&nbsp;`) in headings / after `**注意:**`
   so pandoc preserves the spacing.

3. **Register the mapping** in `help_docs/build_help.py`:

   ```python
   SOURCE_TO_HTML = {
       ...
       '<slug>': '<slug>',
   }
   ```

4. **Add a Flask route** in `routes/main.py`, mirroring the existing ones:

   ```python
   @main.route('/help/<slug>')
   def help_<slug>():
       """Render the help page for <topic>."""
       return render_template('help/<slug>.html')
   ```

5. **(Optional) Link it** from `templates/help.html` inside the
   `<div class="core-class">…</div>` block.

6. **Build** the HTML:

   ```bash
   python help_docs/build_help.py html
   ```

7. **Verify**:
   - `python app_local.py` then visit `http://127.0.0.1:5000/help/<slug>`.
   - Check that headings, tables, warning boxes, and method signatures all
     render with the expected CSS styling.

8. **Report** to the user: which files were created, what URL to visit,
   and remind them to commit both `.md` and `.html` together.

## Edit an existing page

1. **Never** edit `templates/help/<name>.html` directly — those changes
   will be clobbered on the next build.
2. Edit `help_docs/src/<name>.md`.
3. Run `python help_docs/build_help.py html`.
4. Verify in the running app (`/help/<name>`).
5. Commit `.md` + `.html` together.

## Delete a page

1. Delete `help_docs/src/<name>.md`.
2. Delete `templates/help/<name>.html`.
3. Remove the route from `routes/main.py`.
4. Remove the entry from `SOURCE_TO_HTML` in `help_docs/build_help.py`.
5. Remove any `<a href="/help/<name>">` link in `templates/help.html`.
6. Confirm with the user before deleting (this is destructive).

## Common gotchas

- `<div class="warning">` blocks **must** be separated from surrounding
  markdown by blank lines, otherwise pandoc folds them into the previous
  paragraph.
- Method signatures use a bare fenced code block; the build script wraps
  them in `<div class="method-signature">` automatically. Do not add that
  div manually around a code block (it would double-wrap).
- Parameter tables need the `<div class="parameter-table">` wrapper
  **and** blank lines before/after both the opening and closing div.
- If the build fails with a pandoc error, check the `.md` for unclosed
  raw HTML divs — pandoc treats them as raw blocks and passes them through,
  but a missing `</div>` will break the downstream structure.
