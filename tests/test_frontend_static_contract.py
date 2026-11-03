"""Static frontend contract tests.

These do not need a browser: they catch missing include files, removed layout
landmarks, and JavaScript syntax errors before a page is opened.
"""

import pathlib
import re
import shutil
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
TEMPLATES = ROOT / 'templates'
STATIC = ROOT / 'static'


def test_javascript_syntax_if_node_is_available():
    node = shutil.which('node')
    if node is None:
        pytest.skip('node is not installed')
    for path in sorted(STATIC.glob('*.js')):
        result = subprocess.run(
            [node, '--check', str(path)],
            capture_output=True, text=True, check=False,
        )
        assert result.returncode == 0, f'{path.name}: {result.stderr}'


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
    for path in TEMPLATES.glob('*.html'):
        assert 'class="keyboard-header"' not in path.read_text(encoding='utf-8'), path.name


def test_templates_have_unique_ids_and_local_assets():
    for path in TEMPLATES.glob('*.html'):
        text = path.read_text(encoding='utf-8')
        ids = re.findall(r'\bid="([^"]+)"', text)
        assert len(ids) == len(set(ids)), path.name
        for asset in re.findall(r'(?:src|href)="([^"]+)"', text):
            if not asset.startswith('/static/'):
                continue
            relative = asset.split('?', 1)[0].replace('/static/', '', 1)
            assert (STATIC / relative).exists(), (path.name, asset)
