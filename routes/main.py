"""Main routes module for the HighDream application.

This module defines the primary blueprint and routes for the web application,
including the home page, matrix panel, analysis and lab features.
"""

from flask import Blueprint, render_template, request

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


@main.route('/matrix_panel')
def matrix_panel():
    """Render the matrix panel page.

    Returns:
        str: Rendered HTML template for the matrix panel interface.
    """
    return render_template('matrix_panel.html')


@main.route('/matrix_analysis', methods=['GET', 'POST'])
def matrix_analysis():
    """Handle matrix analysis requests with support for both GET and POST methods.

    For POST requests, retrieve expression data from form data.
    For GET requests, retrieve expression data from query parameters.

    Returns:
        str: Rendered HTML template for matrix analysis with expression context.
    """
    expr = request.form.get(
        'expression') if request.method == 'POST' else request.args.get('expression')
    return render_template('matrix_analysis.html', expression=expr)


@main.route('/matrix_lab', methods=['GET', 'POST'])
def matrix_lab():
    """Handle matrix laboratory requests with support for both GET and POST methods.

    For POST requests, retrieve expression data from form data.
    For GET requests, retrieve expression data from query parameters.

    Returns:
        str: Rendered HTML template for the matrix lab with expression context.
    """
    expr = request.form.get(
        'expression') if request.method == 'POST' else request.args.get('expression')
    return render_template('matrix_lab.html', expression=expr)


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
