"""Main routes module for the HighDream application.

This module defines the primary blueprint and routes for the web application,
including the home page, matrix panel, analysis and lab features.
"""

from flask import Blueprint, render_template, request

from utils.matrix_latex import latex_to_nested

main = Blueprint('main', __name__)
"""Blueprint: The main application blueprint instance.

    Registers all primary routes for the application including:
    - Home page
    - Matrix panel
    - Matrix analysis
    - Matrix lab
"""


@main.route('/')
def index():
    """Render the home page.

    Returns:
        str: Rendered HTML template for the index page.
    """
    return render_template('index.html')


@main.route('/help')
def help_docs():
    """Render the help page.

    Returns:
        str: Rendered HTML template for the help page.
    """
    return render_template('help.html')


@main.route('/help/base_calculator')
def help_base_calculator():
    """Render the help page for the base calculator.

    Returns:
        str: Rendered HTML template for the help page for the base calculator.
    """
    return render_template('help/base_calculator.html')


@main.route('/help/base_step_generator')
def help_base_step_generator():
    """Render the help page for the base step generator.

    Returns:
        str: Rendered HTML template for the help page for the base step generator.
    """
    return render_template('help/base_step_generator.html')


@main.route('/help/rule_registry')
def help_rule_registry():
    """Render the help page for the rule registry.

    Returns:
        str: Rendered HTML template for the help page for the rule registry.
    """
    return render_template('help/rule_registry.html')


@main.route('/help/common_matrix_calculator')
def help_common_matrix_calculator():
    """Render the help page for the common matrix calculator.

    Returns:
        str: Rendered HTML template for the help page for the common matrix calculator.
    """
    return render_template('help/common_matrix_calculator.html')


@main.route('/help/matrix_step_generator')
def help_matrix_step_generator():
    """Render the help page for the matrix step generator.

    Returns:
        str: Rendered HTML template for the help page for the matrix step generator.
    """
    return render_template('help/matrix_step_generator.html')


@main.route('/help/base_manual_step_solver')
def help_base_manual_step_solver():
    """Render the help page for the base manual step solver.

    Returns:
        str: Rendered HTML template for the help page for the base manual step solver.
    """
    return render_template('help/base_manual_step_solver.html')


@main.route('/help/diff_manual_step_solver')
def help_diff_manual_step_solver():
    """Render the help page for the differentiation manual step solver.

    Returns:
        str: Rendered HTML template for the help page for the diff manual step solver.
    """
    return render_template('help/diff_manual_step_solver.html')


@main.route('/help/integral_manual_step_solver')
def help_integral_manual_step_solver():
    """Render the help page for the integral manual step solver.

    Returns:
        str: Rendered HTML template for the help page for the integral manual step solver.
    """
    return render_template('help/integral_manual_step_solver.html')


@main.route('/help/limit_manual_step_solver')
def help_limit_manual_step_solver():
    """Render the help page for the limit manual step solver.

    Returns:
        str: Rendered HTML template for the help page for the limit manual step solver.
    """
    return render_template('help/limit_manual_step_solver.html')


@main.route('/help/method_tree_enumerator')
def help_method_tree_enumerator():
    """Render the help page for the method tree enumerator.

    Returns:
        str: Rendered HTML template for the help page for the method tree enumerator.
    """
    return render_template('help/method_tree_enumerator.html')


@main.route('/matrix_panel')
def matrix_panel():
    """Render the matrix panel page.

    Returns:
        str: Rendered HTML template for the matrix panel interface.
    """
    return render_template('matrix_panel.html')


def _matrix_expression(raw):
    """Normalize a URL/form matrix expression; return ``(expr, error)``."""
    if raw is None or str(raw).strip() == '':
        return None, None
    try:
        expr = latex_to_nested(str(raw))
        # Shared parser: rejects empty / ∞ / NaN matrices consistently with the API.
        from core.common_matrix_calculator import CommonMatrixCalculator  # pylint: disable=import-outside-toplevel
        CommonMatrixCalculator.parse_matrix_input(expr)
        return expr, None
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return None, f'无法解析矩阵输入: {exc}'


@main.route('/matrix_analysis', methods=['GET', 'POST'])
def matrix_analysis():
    """Handle matrix analysis requests with support for both GET and POST methods.

    For POST requests, retrieve expression data from form data.
    For GET requests, retrieve expression data from query parameters.

    Returns:
        str: Rendered HTML template for matrix analysis with expression context.
    """
    raw = request.form.get(
        'expression') if request.method == 'POST' else request.args.get('expression')
    expr, error = _matrix_expression(raw)
    return render_template('matrix_analysis.html', expression=expr, matrix_error=error)


@main.route('/matrix_lab', methods=['GET', 'POST'])
def matrix_lab():
    """Handle matrix laboratory requests with support for both GET and POST methods.

    For POST requests, retrieve expression data from form data.
    For GET requests, retrieve expression data from query parameters.

    Returns:
        str: Rendered HTML template for the matrix lab with expression context.
    """
    raw = request.form.get(
        'expression') if request.method == 'POST' else request.args.get('expression')
    expr, error = _matrix_expression(raw)
    return render_template('matrix_lab.html', expression=expr, matrix_error=error)



@main.route('/blueprint_canvas')
def blueprint_canvas():
    """Render the blueprint canvas page.

    Returns:
        str: Rendered HTML template for the blueprint canvas interface.
    """
    return render_template('blueprint_canvas.html')


@main.route('/word_editor')
def word_editor():
    """Render the word editor page for symbolic computation document editing.

    Returns:
        str: Rendered HTML template for the word editor interface.
    """
    return render_template('word_editor.html')


@main.route('/method_tree')
def method_tree():
    """Render the exhaustive method-tree blueprint page.

    Query-string parameters (domain, expression, variable, point, direction)
    are passed through verbatim — the client-side JS reads them and
    pre-fills the form / auto-starts enumeration.
    """
    return render_template('method_tree.html')


@main.route('/rule_select')
def rule_select():
    """Render the manual rule-selection page.

    Returns:
        str: Rendered HTML template for the rule selection interface.
    """
    return render_template('rule_select.html')
