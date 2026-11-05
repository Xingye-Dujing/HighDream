"""Reusable matrix display operations (trace, subspaces, pseudoinverse, ...).

These are the non-calculator matrix views used by the universal dispatcher;
keeping them out of the route service makes them independently testable and
usable by other callers (for example a future CLI) without importing Flask.
"""

from sympy import Symbol, expand, eye, factor, latex, simplify

from core.common_matrix_calculator import CommonMatrixCalculator


def matrix_of(expression):
    """Parse a matrix expression into a SymPy Matrix."""
    return CommonMatrixCalculator.parse_matrix_input(expression)


def matrix_trace(expression):
    """tr(A) = sum of the main diagonal, shown as its own step."""
    try:
        mat = matrix_of(expression)
        total = mat.trace()
        diag = [mat[i, i] for i in range(min(mat.rows, mat.cols))]
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return False, f'矩阵解析失败: {exc}'
    diag_latex = ' + '.join(latex(d) for d in diag) if diag else '0'
    lines = ['& ' + latex(mat),
             r'&= ' + diag_latex + r' \quad \text{对角线元素之和}',
             r'&= \boxed{' + latex(total) + r'}']
    return True, r'\begin{align}' + r' \\ '.join(lines) + r'\end{align}'


def matrix_subspace(expression, kind):
    """Row/column/null space via reduced row echelon form, shown step by step."""
    try:
        mat = matrix_of(expression)
        rref_mat, pivots = mat.rref()
        title = {'nullspace': '零空间 N(A)', 'colspace': '列空间 C(A)', 'rowspace': '行空间 R(A)'}[kind]
        pivot_latex = ', '.join(str(p + 1) for p in pivots) if pivots else '无'
        lines = ['& ' + latex(mat),
                 r'&= ' + latex(rref_mat) + r' \quad \text{行最简形, 主元在第 } ' + pivot_latex + r' \text{ 列}']
        if kind == 'nullspace':
            basis = mat.nullspace()
        elif kind == 'rowspace':
            basis = [rref_mat.row(i).T for i in range(rref_mat.rows)
                     if any(not rref_mat[i, j].is_zero for j in range(rref_mat.cols))]
        else:  # colspace: pivot columns of the ORIGINAL matrix
            basis = [mat.col(p) for p in pivots]
        for vec in basis:
            lines.append(r'&\rightarrow\ ' + latex(vec))
        lines.append(r'&\quad \text{' + title + ' 的一组基如上' +
                     (f', 维数 = {len(basis)}' if basis else ', 维数 = 0') + '}')
        return True, r'\begin{align}' + r' \\ '.join(lines) + r'\end{align}'
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return False, f'子空间计算失败: {exc}'


def matrix_pinv(expression):
    """Moore-Penrose pseudoinverse, via full-rank formulas when applicable."""
    try:
        mat = matrix_of(expression)
        rank, rows, cols = mat.rank(), mat.rows, mat.cols
        lines = ['& ' + latex(mat)]
        if rank == cols and cols <= rows:  # full column rank
            ata = simplify(mat.T * mat)
            lines.append(r'&\Rightarrow A^{T}A = ' + latex(ata)
                         + r' \quad \text{列满秩: } A^{+}=(A^{T}A)^{-1}A^{T}')
            pinv_mat = simplify(ata.inv() * mat.T)
        elif rank == rows and rows <= cols:  # full row rank
            aat = simplify(mat * mat.T)
            lines.append(r'&\Rightarrow AA^{T} = ' + latex(aat)
                         + r' \quad \text{行满秩: } A^{+}=A^{T}(AA^{T})^{-1}')
            pinv_mat = simplify(mat.T * aat.inv())
        else:
            pinv_mat = mat.pinv()
            lines.append(r'& \quad \text{秩亏, 由 Moore-Penrose 定义(SVD/降阶)求伪逆}')
        lines.append(r'&= \boxed{' + latex(pinv_mat) + '}')
        return True, r'\begin{align}' + r' \\ '.join(lines) + r'\end{align}'
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return False, f'伪逆计算失败: {exc}'


def matrix_jordan(expression):
    """Jordan normal form A = P J P^{-1}: eigenvalues, the Jordan form J and P."""
    try:
        mat = matrix_of(expression)
        p_mat, j_mat = mat.jordan_form()
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return False, f'Jordan 标准形失败(可能含无法解析的符号特征值): {exc}'
    j_mat = simplify(j_mat)
    lines = [
        '& ' + latex(mat) + r' \quad \text{矩阵 } A',
        r'&= P J P^{-1} \quad \text{Jordan 分解}',
        r'& J = ' + latex(j_mat) + r' \quad \text{(对角块为 Jordan 块 } \lambda I + N\text{)}',
        r'& P = ' + latex(simplify(p_mat)) + r' \quad \text{(列向量为特征/广义特征向量)}',
    ]
    return True, r'\begin{align}' + r' \\ '.join(lines) + r'\end{align}'


def matrix_definiteness(expression):
    """Classify a real symmetric matrix by its eigenvalues.

    Sylvester's leading-minor criterion only decides strict definiteness; a
    semidefinite matrix (e.g. the zero matrix, or diag(1, 0)) has a zero minor and
    used to be reported as "indefinite", which is mathematically wrong. The
    eigenvalues decide all five cases; the minors stay in the shown derivation.
    """
    try:
        mat = matrix_of(expression)
        if mat.rows != mat.cols:
            return False, '正定/负定判定要求方阵'
        if mat != mat.T:
            return False, '正定/负定判定要求实对称矩阵'
        minors = [simplify(mat[:k, :k].det()) for k in range(1, mat.rows + 1)]
        eigvals = list(mat.eigenvals().keys())
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return False, f'矩阵解析失败: {exc}'
    lines = ['& ' + latex(mat) + r' \quad \text{实对称矩阵 } A']
    for k, delta in enumerate(minors, start=1):
        lines.append(r'& \Delta_{' + str(k) + r'} = ' + latex(delta) + r'\quad \text{顺序主子式}')

    signs = []
    for val in eigvals:
        if val.is_positive:
            signs.append(1)
        elif val.is_negative:
            signs.append(-1)
        elif val.is_zero:
            signs.append(0)
        else:
            signs = None
            break

    if signs is None:
        verdict = '无法判定 (含未定符号的特征值)'
    elif all(s == 0 for s in signs):
        verdict = '零矩阵: 既半正定又半负定'
    elif all(s > 0 for s in signs):
        verdict = '正定 (所有特征值大于 0)'
    elif all(s >= 0 for s in signs):
        verdict = '半正定 (特征值非负, 存在 0 特征值)'
    elif all(s < 0 for s in signs):
        verdict = '负定 (所有特征值小于 0)'
    elif all(s <= 0 for s in signs):
        verdict = '半负定 (特征值非正, 存在 0 特征值)'
    else:
        verdict = '不定 (特征值有正有负)'
    if signs is not None:
        lines.append(r'& \text{特征值: } ' + ', '.join(latex(v) for v in eigvals))
    lines.append(r'& \Rightarrow \boxed{\text{' + verdict + r'}}')
    return True, r'\begin{align}' + r' \\ '.join(lines) + r'\end{align}'


def matrix_charpoly(expression):
    """Characteristic polynomial det(A - lambda I), expanded and factored."""
    try:
        mat = matrix_of(expression)
        if mat.rows != mat.cols:
            return False, '特征多项式要求方阵'
        lam = Symbol('lambda')
        char = (mat - lam * eye(mat.rows)).det()
        expanded = expand(char)
        factored = factor(expanded)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return False, f'特征多项式失败: {exc}'
    lines = [
        '& ' + latex(mat) + r' \quad \text{方阵 } A',
        r'& p(\lambda)=\det(A-\lambda I) = ' + latex(expanded) + r'\quad \text{按行列式展开}',
        r'&= ' + latex(factored) + r'\quad \text{因式分解}, \text{ 令 } p(\lambda)=0 \text{ 得特征值}',
    ]
    return True, r'\begin{align}' + r' \\ '.join(lines) + r'\end{align}'
