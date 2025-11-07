from flask import Blueprint, render_template, request

main = Blueprint('main', __name__)


@main.route('/')
def index():
    return render_template('index.html')


@main.route('/matrix_panel')
def matrix_panel():
    return render_template('matrix_panel.html')


@main.route('/matrix_analysis', methods=['GET', 'POST'])
def matrix_analysis():
    expr = request.form.get(
        'expression') if request.method == 'POST' else request.args.get('expression')
    return render_template('matrix_analysis.html', expression=expr)


@main.route('/matrix_lab', methods=['GET', 'POST'])
def matrix_lab():
    expr = request.form.get(
        'expression') if request.method == 'POST' else request.args.get('expression')
    return render_template('matrix_lab.html', expression=expr)
