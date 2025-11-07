from flask import Blueprint, request, jsonify, send_from_directory
from domains import MatrixAnalyzer, MatrixCalculator, ProcessManager
from config import TREES_DIR
from utils.latex_formatter import str_to_latex
from .compute_service import start_compute

api = Blueprint('api', __name__)


@api.route('/parse', methods=['POST'])
def parse_input():
    data = request.json or {}
    expression = data.get('expression', '')
    operation_type = data.get('operation_type', '')
    try:
        latex_output = str_to_latex(
            expression, operation_type)
        return jsonify({'success': True, 'result': latex_output})
    except Exception as e:
        return jsonify({'success': False, 'error': f'{e}'})


@api.route('/compute', methods=['POST'])
def compute():
    data = request.json or {}
    op_type = data.get('operation_type', '')
    success, result = start_compute(op_type, data)
    if success:
        if op_type == 'expr':
            return jsonify({
                'success': True,
                'result': result[0],
                'tree_svg_url': result[1],
            })
        return jsonify({'success': True, 'result': result})
    return jsonify({'success': False, 'error': result})


@api.route('/matrix_analyze', methods=['POST'])
def matrix_analyze():
    data = request.json or {}
    expr = data.get('expression', '')
    return jsonify(MatrixAnalyzer().analyze(expr))


@api.route('/matrix_cal', methods=['POST'])
def matrix_cal():
    data = request.json or {}
    expr = data.get('expression', '')
    ops = data.get('operations', '').split('\n')
    record = data.get('record_all_steps') == 'True'
    calc = MatrixCalculator()
    calc.calculate(expr, ops, record)
    return jsonify({'steps': calc.get_steps_latex()})


@api.route('/get_process', methods=['POST'])
def get_process():
    data = request.json or {}
    ptype = data.get('type', '')
    expr = data.get('expression', '')
    return jsonify(ProcessManager.get_process(ptype, expr))


@api.route('/static/trees/<path:filename>')
def serve_tree(filename):
    return send_from_directory(TREES_DIR, filename)
