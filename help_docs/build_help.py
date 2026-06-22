#!/usr/bin/env python3
"""Convert between help-doc markdown sources and Flask-served HTML.

The help docs served at ``/help/*`` are authored as Markdown under
``help_docs/src/``. This script is the single source of truth for building
the HTML that Flask renders (``templates/help/*.html``) and for the one-time
reverse conversion from the legacy hand-written HTML back to Markdown.

Usage
-----

From the repository root:

    python help_docs/build_help.py html          # .md -> .html  (regular build)
    python help_docs/build_help.py md            # .html -> .md  (one-shot import)
    python help_docs/build_help.py all           # md then html

Prerequisites: pandoc (>= 3) must be on ``$PATH``. The ``beautifulsoup4``
package is only needed for the ``md`` (reverse-import) command.

Design
------

* Pandoc does the heavy lifting (``gfm`` for reverse import,
  ``markdown+native_divs`` for the forward build).
* Custom CSS classes used by ``static/help.css`` are preserved:

  - ``div.warning``            -> pandoc fenced div ``::: warning``
  - ``div.core-area``          -> pandoc fenced div ``::: core-area``
  - ``div.core-class``         -> pandoc fenced div ``::: core-class``
  - ``div.method-signature``   -> pandoc code block (`` ``` ``) so the raw
    Python signature is not mangled by markdown escaping
  - ``table.parameter-table``  -> GFM table wrapped in a
    ``<div class="parameter-table">`` (pandoc's GFM tables cannot carry
    classes, so the wrapper is required for the CSS to apply)

* ``&nbsp;`` entities from the original HTML are restored as literal
  ``U+00A0`` characters in the Markdown sources so the round-tripped HTML
  matches the original visual layout.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent.parent
TEMPLATES_HELP = ROOT / 'templates' / 'help'
MD_DIR = ROOT / 'help_docs' / 'src'
TEMPLATE_PATH = ROOT / 'help_docs' / 'template.html'

# Mapping: markdown source stem -> output HTML file name. The output file is
# always written under ``templates/help/<name>.html``.
SOURCE_TO_HTML = {
    'base_calculator': 'base_calculator',
    'base_step_generator': 'base_step_generator',
    'rule_registry': 'rule_registry',
    'common_matrix_calculator': 'common_matrix_calculator',
    'matrix_step_generator': 'matrix_step_generator',
    'base_manual_step_solver': 'base_manual_step_solver',
    'diff_manual_step_solver': 'diff_manual_step_solver',
    'integral_manual_step_solver': 'integral_manual_step_solver',
    'limit_manual_step_solver': 'limit_manual_step_solver',
    'method_tree_enumerator': 'method_tree_enumerator',
}


# --------------------------------------------------------------------------- #
# Markdown <- HTML (one-time reverse import)
# --------------------------------------------------------------------------- #

def _strip_outer_html(html: str) -> str:
    """Drop the ``<ul class="header">`` navigation so pandoc only sees content."""

    soup = BeautifulSoup(html, 'html.parser')
    container = soup.find('div', id='help-container')
    if container is None:
        return html
    for header in container.find_all('ul', class_='header'):
        header.extract()
    return ''.join(str(child) for child in container.children)


def _method_signature_blocks(md: str) -> str:
    """Collapse pandoc's raw ``div.method-signature`` blocks into code fences.

    Pandoc emits these as raw HTML blocks with the inner signature escaped
    (``\\_``, ``\\*``, ``\\[`` ...). We unescape and render them as a fenced
    code block, which the existing ``help.css`` styles in a matching way.
    """

    def _replace(match: re.Match) -> str:
        inner = match.group(1)
        inner = re.sub(r'^\n+|\n+$', '', inner)
        inner = (
            inner.replace('\\_', '_')
            .replace('\\*', '*')
            .replace('\\[', '[')
            .replace('\\]', ']')
            .replace('\\(', '(')
            .replace('\\)', ')')
            .replace('\\>', '>')
            .replace('\\<', '<')
            .replace('\\@', '@')
            .replace('\\|', '|')
            .replace('\\~', '~')
        )
        return f'```\n{inner}\n```'

    pattern = (
        r'<div class="method-signature">\n*\n'
        r'(.*?)'
        r'\n*</div>'
    )
    return re.sub(pattern, _replace, md, flags=re.DOTALL)


def _unwrap_parameter_tables(md: str) -> str:
    """Strip ``<div class="parameter-table">`` wrappers so tables become bare GFM.

    Used during the reverse-import pass. The forward build re-wraps every
    bare GFM table via ``_wrap_parameter_tables``.
    """
    md = re.sub(r'<div class="parameter-table">\s*', '', md)
    # Only remove a </div> that sits on its own line, immediately after a
    # table (the previous non-empty line is a ``| ... |`` row).
    lines = md.split('\n')
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if re.match(r'^\s*</div>\s*$', line):
            # Look back for the nearest non-empty line.
            j = len(out) - 1
            while j >= 0 and out[j].strip() == '':
                j -= 1
            if j >= 0 and re.match(r'^\|.*\|\s*$', out[j]):
                # Drop this </div> — it was a parameter-table closer.
                i += 1
                continue
        out.append(line)
        i += 1
    return '\n'.join(out)


def _wrap_parameter_tables(md: str) -> str:
    """Wrap bare GFM tables in ``<div class="parameter-table">``.

    Pandoc's ``html -> gfm`` drops ``class="parameter-table"`` from
    ``<table>`` (GFM tables cannot carry classes), so the ``.parameter-table
    th/td`` CSS rules stop matching and the borders disappear. We restore
    the styling by wrapping every bare GFM table in a ``<div>`` that the
    CSS already targets via ``.parameter-table table th/td``. Tables that
    already sit inside a ``<div class="parameter-table">`` are left alone.
    """
    lines = md.split('\n')
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # Detect a GFM table start: | ... | header row + | --- | separator.
        is_table_head = (
            re.match(r'^\s*\|.+\|\s*$', line)
            and i + 1 < len(lines)
            and re.match(r'^\s*\|[-:\s|]+\|\s*$', lines[i + 1])
        )
        if is_table_head:
            # Skip wrapping if already inside a parameter-table div.
            j = len(out) - 1
            while j >= 0 and out[j].strip() == '':
                j -= 1
            already_wrapped = (
                j >= 0 and re.match(
                    r'^\s*<div class="parameter-table">\s*$', out[j]
                )
            )
            if not already_wrapped:
                out.append('<div class="parameter-table">')
                out.append('')
            while i < len(lines) and re.match(r'^\s*\|', lines[i]):
                out.append(lines[i])
                i += 1
            if not already_wrapped:
                out.append('')
                out.append('</div>')
            continue
        out.append(line)
        i += 1
    return '\n'.join(out)


def _restore_nbsp(md: str) -> str:
    """Restore ``U+00A0`` characters where the original HTML used ``&nbsp;``.

    Pandoc drops ``&nbsp;`` during HTML -> Markdown conversion. The original
    docs used them in three well-defined places, which we restore so the
    re-generated HTML matches the original visual layout.
    """
    nbsp = '\xa0'
    # "**注意:**   文本"  ->  "**注意:**\xa0\xa0 文本"
    md = re.sub(r'(\*\*注意:\*\*)\s+', rf'\1{nbsp}{nbsp} ', md)
    # "一.  类概述"  ->  "一.\xa0\xa0 类概述"
    md = re.sub(r'(##\s+[一二三四五六七八九十]+[.、])\s+', rf'\1{nbsp}{nbsp} ', md)
    # "### 1.  初始化"  ->  "### 1.\xa0\xa0 初始化"
    md = re.sub(r'(###\s+\d+[.、])\s+', rf'\1{nbsp}{nbsp} ', md)
    return md


def convert_html_to_markdown(html_path: Path, md_path: Path, title: str) -> None:
    """Convert a single ``templates/help/*.html`` file to Markdown."""
    with html_path.open(encoding='utf-8') as fh:
        raw_html = fh.read()

    inner = _strip_outer_html(raw_html)

    result = subprocess.run(
        ['pandoc', '-f', 'html', '-t', 'gfm', '--wrap=none'],
        input=inner,
        capture_output=True,
        text=True,
        encoding='utf-8',
    )
    if result.returncode != 0:
        raise RuntimeError(f'pandoc failed on {html_path}: {result.stderr}')

    md = result.stdout
    md = _method_signature_blocks(md)
    md = _wrap_parameter_tables(md)
    md = _restore_nbsp(md)

    frontmatter = f'---\ntitle: "{title}"\noutput: "{html_path.name}"\n---\n\n'
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(frontmatter + md, encoding='utf-8')


def build_all_markdown() -> None:
    """Run the reverse import for every entry in ``SOURCE_TO_HTML``."""
    titles = {
        'base_calculator': 'BaseCalculator - 符号计算器基类文档',
        'base_step_generator': 'BaseStepGenerator - 步骤生成器文档',
        'rule_registry': 'RuleRegistry - 规则注册器文档',
        'common_matrix_calculator': 'CommonMatrixCalculator - 矩阵计算基类文档',
        'matrix_step_generator': 'MatrixStepGenerator - 矩阵步骤生成器文档',
    }
    for stem, html_name in SOURCE_TO_HTML.items():
        html_path = TEMPLATES_HELP / f'{html_name}.html'
        md_path = MD_DIR / f'{stem}.md'
        if not html_path.exists():
            print(f'[skip] {html_path} not found')
            continue
        convert_html_to_markdown(html_path, md_path, titles[stem])
        print(f'[ok] {html_path.name} -> {md_path.name}')


# --------------------------------------------------------------------------- #
# HTML <- Markdown (regular build)
# --------------------------------------------------------------------------- #

_FRONTMATTER_RE = re.compile(r'^---\s*\n(.*?)\n---\s*\n?', re.DOTALL)


def _parse_frontmatter(md_text: str) -> tuple[dict, str]:
    """Pull YAML frontmatter out of a Markdown source file."""
    match = _FRONTMATTER_RE.match(md_text)
    if not match:
        return {}, md_text

    meta: dict = {}
    for line in match.group(1).splitlines():
        if ':' not in line:
            continue
        key, _, value = line.partition(':')
        meta[key.strip()] = value.strip().strip('"').strip("'")
    return meta, md_text[match.end():]


def convert_markdown_to_html(md_path: Path, html_path: Path) -> None:
    """Build a single ``templates/help/*.html`` from its Markdown source."""
    md_text = md_path.read_text(encoding='utf-8')
    meta, body = _parse_frontmatter(md_text)

    result = subprocess.run(
        ['pandoc', '-f', 'markdown+native_divs-implicit_figures', '-t', 'html'],
        input=body,
        capture_output=True,
        text=True,
        encoding='utf-8',
    )
    if result.returncode != 0:
        raise RuntimeError(f'pandoc failed on {md_path}: {result.stderr}')

    html_body = _wrap_method_signatures(result.stdout)

    template = TEMPLATE_PATH.read_text(encoding='utf-8')
    title = meta.get('title', md_path.stem)
    output = (
        template
        .replace('{{title}}', title)
        .replace('{{content}}', html_body)
    )
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(output, encoding='utf-8')


def _wrap_method_signatures(html: str) -> str:
    """Wrap bare ``<pre><code>`` blocks in ``<div class="method-signature">``.

    In the Markdown sources, Python-style method signatures are written as
    fenced code blocks (`` ``` ``). ``help.css`` styles them via the
    ``.method-signature`` selector, so we re-introduce that wrapper here.
    Code blocks already inside a ``<div class="...">`` (e.g. an example) are
    left untouched.
    """
    return re.sub(
        r'(?<!\w)<pre><code>(.*?)</code></pre>',
        lambda m: f'<div class="method-signature">{m.group(1)}</div>',
        html,
        flags=re.DOTALL,
    )


def build_all_html() -> None:
    """Rebuild every ``templates/help/*.html`` from its Markdown source."""
    if not TEMPLATE_PATH.exists():
        print(f'[error] template missing: {TEMPLATE_PATH}')
        sys.exit(1)
    for md_path in sorted(MD_DIR.glob('*.md')):
        meta, _ = _parse_frontmatter(md_path.read_text(encoding='utf-8'))
        html_name = meta.get('output', md_path.stem)
        # Strip any ``.html`` suffix so "base_calculator.html" and
        # "base_calculator" both yield ``base_calculator.html`` on output.
        if html_name.endswith('.html'):
            html_name = html_name[:-5]
        html_path = TEMPLATES_HELP / f'{html_name}.html'
        convert_markdown_to_html(md_path, html_path)
        print(f'[ok] {md_path.name} -> {html_path.relative_to(ROOT)}')


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> int:
    parser = argparse.ArgumentParser(description='Build help documentation.')
    parser.add_argument(
        'command',
        choices=['md', 'html', 'all'],
        help=(
            'md: regenerate .md from existing .html (one-shot import); '
            'html: build .html from .md sources (regular build); '
            'all: run md then html.'
        ),
    )
    args = parser.parse_args()

    if args.command == 'md':
        build_all_markdown()
    elif args.command == 'html':
        build_all_html()
    elif args.command == 'all':
        build_all_markdown()
        build_all_html()
    return 0


if __name__ == '__main__':
    sys.exit(main())
