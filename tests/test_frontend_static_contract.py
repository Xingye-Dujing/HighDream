"""Static frontend contract tests.

These do not need a browser: they catch missing include files, removed layout
landmarks, unbalanced JavaScript and unescaped server values in innerHTML.
"""

import pathlib
import re
import shutil
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
TEMPLATES = ROOT / 'templates'
STATIC = ROOT / 'static'


def _js_files():
    return sorted(STATIC.glob('*.js'))


def _html_templates():
    # rglob also covers templates/help/*.html (the generated docs).  Exclude
    # vendored assets that are not authored by this project.
    return [path for path in sorted(TEMPLATES.rglob('*.html'))
            if 'vendor' not in path.parts]


def _js_bracket_problem(text):
    """Return None when brackets and quotes pair up in JS source, else a message.

    A small lexer skips ``//`` and ``/* */`` comments, quoted strings, template
    literals and regex literals, then checks the remaining ``()[]{}`` with a
    stack.  It is a real (if coarse) syntax check that also works without Node,
    so a missing Node never silently skips the whole JS check group again.
    """
    stack = []
    pairs = {')': '(', ']': '[', '}': '{'}
    expression_start = set('=([{,:;!&|?+-*%^~<>{}')
    index, size = 0, len(text)
    previous = ''
    while index < size:
        char = text[index]
        following = text[index + 1] if index + 1 < size else ''
        if char == '/' and following == '/':
            newline = text.find('\n', index)
            index = size if newline < 0 else newline
            continue
        if char == '/' and following == '*':
            end = text.find('*/', index + 2)
            index = size if end < 0 else end + 2
            continue
        if char == '/' and (previous == '' or previous in expression_start):
            # Regex literal: skip to its unescaped closing '/'.
            index += 1
            in_class = False
            closed = False
            while index < size:
                current = text[index]
                if current == '\\':
                    index += 2
                    continue
                if current == '[':
                    in_class = True
                elif current == ']':
                    in_class = False
                elif current == '/' and not in_class:
                    closed = True
                    index += 1
                    break
                elif current == '\n':
                    break
                index += 1
            if closed:
                while index < size and text[index].isalpha():
                    index += 1
                previous = '/'
                continue
        if char in "'\"`":
            quote = char
            index += 1
            closed = False
            while index < size:
                if text[index] == '\\':
                    index += 2
                    continue
                if text[index] == quote:
                    index += 1
                    closed = True
                    break
                index += 1
            if not closed:
                return f'unclosed {quote} string'
            previous = quote
            continue
        if char in '([{':
            stack.append((char, index))
        elif char in ')]}':
            if not stack or stack[-1][0] != pairs[char]:
                line = text.count('\n', 0, index) + 1
                return f'mismatched {char!r} at line {line}'
            stack.pop()
        if not char.isspace():
            previous = char
        index += 1
    if stack:
        char, position = stack[-1]
        line = text.count('\n', 0, position) + 1
        return f'unclosed {char!r} opened at line {line}'
    return None


def test_javascript_files_are_syntactically_sound():
    files = _js_files()
    assert files
    node = shutil.which('node')
    for path in files:
        text = path.read_text(encoding='utf-8')
        assert text.strip(), f'{path.name} is empty'
        # Pure-Python check always runs (no silent skip when Node is absent).
        problem = _js_bracket_problem(text)
        assert problem is None, f'{path.name}: {problem}'
        # Only when Node is available do we additionally run its real parser.
        if node is not None:
            result = subprocess.run(
                [node, '--check', str(path)],
                capture_output=True, text=True, check=False,
            )
            assert result.returncode == 0, f'{path.name}: {result.stderr}'


@pytest.mark.parametrize('broken', [
    'function f( { return 1; }',
    'var s = "unclosed;',
    'if (x) { f();',
    'var a = [1, 2;',
])
def test_js_bracket_checker_detects_broken_source(broken):
    assert _js_bracket_problem(broken) is not None, broken


def test_js_bracket_checker_accepts_valid_source():
    assert _js_bracket_problem('var s = "a(b)c"; function f(){ return 1; }') is None
    assert _js_bracket_problem('var r = /\\$([^$]+)\\$/g; s.replace(r, "x");') is None


@pytest.mark.parametrize('page', ['index.html', 'matrix_panel.html', 'matrix_lab.html'])
def test_keyboard_pages_have_body_layout(page):
    text = (TEMPLATES / page).read_text(encoding='utf-8')
    assert 'keyboard-header' not in text
    assert 'id="main-cols"' in text
    assert 'id="math-keyboard"' in text
    assert text.index('id="main-cols"') < text.index('id="math-keyboard"')


@pytest.mark.parametrize('page', ['matrix_panel.html', 'matrix_lab.html'])
def test_matrix_pages_use_universal_input(page):
    text = (TEMPLATES / page).read_text(encoding='utf-8')
    assert 'hd-universal-input.js' in text
    if page == 'matrix_panel.html':
        assert 'data-universal-input' in (STATIC / 'matrix_panel.js').read_text(encoding='utf-8')
    else:
        assert 'data-universal-input' in text


def test_matrix_analysis_has_form_and_no_palette():
    text = (TEMPLATES / 'matrix_analysis.html').read_text(encoding='utf-8')
    assert 'hd-universal-input.js' in text
    assert 'data-universal-input' in text
    assert 'matrix-analysis-form' in text
    assert 'id="math-keyboard"' not in text


def test_matrix_lab_has_start_form():
    text = (TEMPLATES / 'matrix_lab.html').read_text(encoding='utf-8')
    assert 'id="matrix-lab-start"' in text
    assert '载入矩阵' in text


def test_all_templates_have_no_placeholder_header_palette():
    for path in _html_templates():
        assert 'class="keyboard-header"' not in path.read_text(encoding='utf-8'), path.name


def test_help_templates_are_included_in_the_scan():
    help_pages = [path for path in _html_templates() if path.parent.name == 'help']
    assert len(help_pages) == 10, help_pages


def test_templates_have_unique_ids_and_local_assets():
    for path in _html_templates():
        text = path.read_text(encoding='utf-8')
        ids = re.findall(r'\bid="([^"]+)"', text)
        assert len(ids) == len(set(ids)), path.name
        for asset in re.findall(r'(?:src|href)="([^"]+)"', text):
            if not asset.startswith('/static/'):
                continue  # external/vendor CDN assets are not local files
            relative = asset.split('?', 1)[0].replace('/static/', '', 1)
            assert (STATIC / relative).exists(), (path.name, asset)


# Server-authored strings (SymPy LaTeX, rule explanations, task results) can carry
# user-controlled characters: sympify accepts Symbol('<img src=x onerror=...>').
# Whenever such a value is written with innerHTML it must be escapeHtml()-wrapped
# (or set via textContent), otherwise the browser executes the injected tags.
UNTRUSTED_INNERHTML_VALUES = [
    ('script.js', 'only.latex'),
    ('script.js', 'it.latex'),
    ('script.js', 'it.reason'),
    ('script.js', 'task.result'),
    ('method_tree.js', 'decision.parent_latex'),
    ('method_tree.js', 'decision.preview_latex'),
    ('method_tree.js', 'st.latex'),
    ('rule_select.js', 'st.latex'),
    ('rule_select.js', 'state.current_expr_latex'),
    ('rule_select.js', 'r.latex_preview'),
    ('matrix_lab.js', 'data.steps'),
    ('matrix_lab.js', 'data.result'),
    ('matrix_analysis.js', 'data.result'),
    ('matrix_analysis.js', 'data.rank'),
    ('matrix_analysis.js', 'data.determinant'),
    ('matrix_analysis.js', 'data.eigenvalues'),
    ('matrix_analysis.js', 'data.eigenvectors'),
    ('matrix_analysis.js', 'data_copy.ref'),
    ('matrix_analysis.js', 'data.process'),
    ('matrix_analysis.js', 'data.U'),
    ('matrix_analysis.js', 'data.S'),
    ('matrix_analysis.js', 'data.V'),
    ('blueprint_canvas.js', 'data.latex'),
    ('blueprint_canvas.js', 'node.data.latex'),
    ('blueprint_canvas.js', 'node.data.expression'),
    ('blueprint_canvas.js', 'node.data.result'),
    ('blueprint_canvas.js', 'node.data.text'),
    ('blueprint_canvas.js', 'node.title'),
    ('word_editor.js', 'data.latex'),
]

_INNERHTML_ASSIGN_RE = re.compile(r'\.innerHTML\s*(\+?=)')
# Object prefixes holding values that come from the server (API JSON payloads,
# task results, session state, step/rule/decision records, ...).  Checking the
# RHS only keeps LHS DOM references such as ``st.scroll.innerHTML`` out of scope.
_SERVER_FIELD_RE = re.compile(
    r'\b(?:data|data_copy|state|stats|task|decision|node|payload|response|it|st|r|only)'
    r'\.[A-Za-z_]\w*'
)


def _innerhtml_assignment_blocks(text):
    """Yield ``(line_no, rhs, block)`` for every innerHTML assignment.

    ``rhs`` is everything right of the assignment operator (including
    continuation lines of a multi-line template / string-concat statement);
    ``block`` is the whole statement, used to look for the escaping guard.
    """
    lines = text.splitlines()
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(('//', '*', '/*')):
            continue
        match = _INNERHTML_ASSIGN_RE.search(line)
        if match is None:
            continue
        block_lines = [line]
        cursor = index
        while cursor + 1 < len(lines) and cursor + 1 - index < 60:
            body = '\n'.join(block_lines)
            last = block_lines[-1].rstrip()
            if (body.count('`') % 2 == 0
                    and body.count('(') <= body.count(')')
                    and body.count('[') <= body.count(']')
                    and (';' in last or last.endswith('`'))):
                break
            cursor += 1
            block_lines.append(lines[cursor])
        rhs = '\n'.join([line[match.end():]] + block_lines[1:])
        yield index + 1, rhs, '\n'.join(block_lines)


def _unescaped_innerhtml_insertions(text):
    """Diagnostics for innerHTML assignments whose RHS references a server field
    without ``escapeHtml`` / ``textContent`` in the same statement block.

    Template interpolation ``${...}`` is covered transitively: a ``${data.x}``
    or ``${task.result}`` interpolation puts a server field on the RHS.  Purely
    local UI interpolation (``${icon}``, ``${matrixLatex}``, ...) is ignored.
    """
    problems = []
    for line_no, rhs, block in _innerhtml_assignment_blocks(text):
        if not _SERVER_FIELD_RE.search(rhs):
            continue
        if 'escapeHtml' in block or 'textContent' in block:
            continue
        problems.append(f'line {line_no}: {rhs.strip()[:120]}')
    return problems


def test_all_innerhtml_insertions_escape_server_values():
    """Full scan of static/*.js (not a fixed whitelist): every innerHTML
    assignment touching a server field must be escaped, so a newly added
    ``data.foo`` insertion point cannot slip through unnoticed."""
    files = _js_files()
    assert files
    for path in files:
        problems = _unescaped_innerhtml_insertions(path.read_text(encoding='utf-8'))
        assert not problems, f'{path.name}: {problems}'


def test_innerhtml_scanner_detects_new_unescaped_insertion_points():
    # Metamorphic check: prove the scanner above is not a vacuous assertion.
    assert _unescaped_innerhtml_insertions('el.innerHTML = data.injected;')
    assert _unescaped_innerhtml_insertions('el.innerHTML = `<b>${data.injected}</b>`;')
    assert _unescaped_innerhtml_insertions('el.innerHTML = "<b>" + task.result + "</b>";')
    assert not _unescaped_innerhtml_insertions(
        'el.innerHTML = `<b>${escapeHtml(data.injected)}</b>`;')
    assert not _unescaped_innerhtml_insertions('el.textContent = data.injected;')
    assert not _unescaped_innerhtml_insertions('el.innerHTML = `<div>${icon}</div>`;')


@pytest.mark.parametrize('filename,needle', UNTRUSTED_INNERHTML_VALUES)
def test_server_values_are_escaped_before_innerhtml(filename, needle):
    """Keep the explicit field list as an extra, concrete check."""
    text = (STATIC / filename).read_text(encoding='utf-8')
    # The field must still exist (else this case would pass vacuously), and at
    # least one occurrence must actually be passed through escapeHtml.
    assert needle in text, (filename, needle)
    assert any(needle in line and 'escapeHtml' in line
               for line in text.splitlines()), (filename, needle)
    for line in text.splitlines():
        if needle not in line or 'innerHTML' not in line:
            continue
        assert 'escapeHtml' in line or 'textContent' in line, (filename, needle, line.strip())
