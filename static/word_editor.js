let paragraphCounter = 0;
let activeParagraphId = null;
let isDarkTheme = localStorage.getItem('darkTheme') === 'true';

// SymPy auto-completion items
const sympyCompletions = {
  // Core functions
  'diff': { type: 'function', doc: 'Derivative', docCn: '求导', signature: 'diff(expr, x)' },
  'integrate': { type: 'function', doc: 'Integral', docCn: '积分', signature: 'integrate(expr, x)' },
  'limit': { type: 'function', doc: 'Limit', docCn: '求极限', signature: 'limit(expr, x, x0)' },
  'series': { type: 'function', doc: 'Series expansion', docCn: '泰勒级数展开', signature: 'series(expr, x, x0, n)' },
  'summation': { type: 'function', doc: 'Summation', docCn: '求和', signature: 'summation(expr, (n, a, b))' },
  'product': { type: 'function', doc: 'Product', docCn: '求积', signature: 'product(expr, (n, a, b))' },
  // Simplification
  'simplify': { type: 'function', doc: 'Simplify expression', docCn: '简化表达式', signature: 'simplify(expr)' },
  'expand': { type: 'function', doc: 'Expand expression', docCn: '展开表达式', signature: 'expand(expr)' },
  'factor': { type: 'function', doc: 'Factor expression', docCn: '因式分解', signature: 'factor(expr)' },
  'collect': { type: 'function', doc: 'Collect terms', docCn: '收集同类项', signature: 'collect(expr, x)' },
  'apart': { type: 'function', doc: 'Partial fraction', docCn: '部分分式分解', signature: 'apart(expr, x)' },
  'together': { type: 'function', doc: 'Combine fractions', docCn: '合并分数', signature: 'together(expr)' },
  'cancel': { type: 'function', doc: 'Cancel common factors', docCn: '约分化简', signature: 'cancel(expr)' },
  // Solvers
  'solve': { type: 'function', doc: 'Solve equation', docCn: '解方程', signature: 'solve(expr, x)' },
  'solveset': { type: 'function', doc: 'Solve equation set', docCn: '解方程集', signature: 'solveset(expr, x)' },
  'linsolve': { type: 'function', doc: 'Solve linear system', docCn: '解线性方程组', signature: 'linsolve([eq1, eq2], [x, y])' },
  'nonlinsolve': { type: 'function', doc: 'Solve nonlinear system', docCn: '解非线性方程组', signature: 'nonlinsolve([eq1, eq2], [x, y])' },
  'dsolve': { type: 'function', doc: 'Solve differential equation', docCn: '解微分方程', signature: 'dsolve(eq, f(x))' },
  // Matrix operations
  'Matrix': { type: 'class', doc: 'Matrix class', docCn: '矩阵类', signature: 'Matrix([[1, 2], [3, 4]])' },
  'eye': { type: 'function', doc: 'Identity matrix', docCn: '单位矩阵', signature: 'eye(n)' },
  'zeros': { type: 'function', doc: 'Zero matrix', docCn: '零矩阵', signature: 'zeros(m, n)' },
  'ones': { type: 'function', doc: 'Ones matrix', docCn: '全1矩阵', signature: 'ones(m, n)' },
  'diag': { type: 'function', doc: 'Diagonal matrix', docCn: '对角矩阵', signature: 'diag([1, 2, 3])' },
  'det': { type: 'function', doc: 'Determinant', docCn: '行列式', signature: 'det(M)' },
  'inv': { type: 'function', doc: 'Matrix inverse', docCn: '矩阵逆', signature: 'inv(M)' },
  'transpose': { type: 'function', doc: 'Matrix transpose', docCn: '矩阵转置', signature: 'transpose(M)' },
  'rank': { type: 'function', doc: 'Matrix rank', docCn: '矩阵秩', signature: 'rank(M)' },
  'eigenvals': { type: 'function', doc: 'Eigenvalues', docCn: '特征值', signature: 'eigenvals(M)' },
  'eigenvects': { type: 'function', doc: 'Eigenvectors', docCn: '特征向量', signature: 'eigenvects(M)' },
  'diagonalize': { type: 'function', doc: 'Diagonalization', docCn: '对角化', signature: 'diagonalize(M)' },
  'rref': { type: 'function', doc: 'Reduced row echelon', docCn: '行最简形', signature: 'rref(M)' },
  'lu': { type: 'function', doc: 'LU decomposition', docCn: 'LU分解', signature: 'lu(M)' },
  'qr': { type: 'function', doc: 'QR decomposition', docCn: 'QR分解', signature: 'qr(M)' },
  'svd': { type: 'function', doc: 'SVD decomposition', docCn: '奇异值分解', signature: 'svd(M)' },
  // Calculus
  'Derivative': { type: 'class', doc: 'Derivative class', docCn: '导数类', signature: 'Derivative(expr, x)' },
  'Integral': { type: 'class', doc: 'Integral class', docCn: '积分类', signature: 'Integral(expr, x)' },
  'Limit': { type: 'class', doc: 'Limit class', docCn: '极限类', signature: 'Limit(expr, x, x0)' },
  // Constants
  'pi': { type: 'constant', doc: 'π constant', docCn: '圆周率π', signature: 'pi' },
  'E': { type: 'constant', doc: 'Euler\'s number', docCn: '自然常数e', signature: 'E' },
  'oo': { type: 'constant', doc: 'Infinity', docCn: '无穷大', signature: 'oo' },
  'I': { type: 'constant', doc: 'Imaginary unit', docCn: '虚数单位i', signature: 'I' },
  'nan': { type: 'constant', doc: 'Not a number', docCn: '非数字', signature: 'nan' },
  // Symbol creation
  'symbols': { type: 'function', doc: 'Create symbols', docCn: '创建符号', signature: 'symbols("x y z")' },
  'Symbol': { type: 'class', doc: 'Symbol class', docCn: '符号类', signature: 'Symbol("x")' },
  'var': { type: 'function', doc: 'Create variable', docCn: '创建变量', signature: 'var("x y z")' },
  // Trigonometric functions
  'sin': { type: 'function', doc: 'Sine', docCn: '正弦', signature: 'sin(x)' },
  'cos': { type: 'function', doc: 'Cosine', docCn: '余弦', signature: 'cos(x)' },
  'tan': { type: 'function', doc: 'Tangent', docCn: '正切', signature: 'tan(x)' },
  'cot': { type: 'function', doc: 'Cotangent', docCn: '余切', signature: 'cot(x)' },
  'sec': { type: 'function', doc: 'Secant', docCn: '正割', signature: 'sec(x)' },
  'csc': { type: 'function', doc: 'Cosecant', docCn: '余割', signature: 'csc(x)' },
  'asin': { type: 'function', doc: 'Arc sine', docCn: '反正弦', signature: 'asin(x)' },
  'acos': { type: 'function', doc: 'Arc cosine', docCn: '反余弦', signature: 'acos(x)' },
  'atan': { type: 'function', doc: 'Arc tangent', docCn: '反正切', signature: 'atan(x)' },
  'atan2': { type: 'function', doc: 'Arc tangent 2', docCn: '四象限反正切', signature: 'atan2(y, x)' },
  // Hyperbolic functions
  'sinh': { type: 'function', doc: 'Hyperbolic sine', docCn: '双曲正弦', signature: 'sinh(x)' },
  'cosh': { type: 'function', doc: 'Hyperbolic cosine', docCn: '双曲余弦', signature: 'cosh(x)' },
  'tanh': { type: 'function', doc: 'Hyperbolic tangent', docCn: '双曲正切', signature: 'tanh(x)' },
  'asinh': { type: 'function', doc: 'Arc hyperbolic sine', docCn: '反双曲正弦', signature: 'asinh(x)' },
  'acosh': { type: 'function', doc: 'Arc hyperbolic cosine', docCn: '反双曲余弦', signature: 'acosh(x)' },
  'atanh': { type: 'function', doc: 'Arc hyperbolic tangent', docCn: '反双曲正切', signature: 'atanh(x)' },
  // Exponential and logarithmic
  'exp': { type: 'function', doc: 'Exponential', docCn: '指数函数', signature: 'exp(x)' },
  'log': { type: 'function', doc: 'Natural logarithm', docCn: '自然对数', signature: 'log(x)' },
  'ln': { type: 'function', doc: 'Natural logarithm', docCn: '自然对数', signature: 'ln(x)' },
  'sqrt': { type: 'function', doc: 'Square root', docCn: '平方根', signature: 'sqrt(x)' },
  'cbrt': { type: 'function', doc: 'Cube root', docCn: '立方根', signature: 'cbrt(x)' },
  'Rational': { type: 'class', doc: 'Rational number', docCn: '有理数', signature: 'Rational(1, 2)' },
  'Integer': { type: 'class', doc: 'Integer', docCn: '整数', signature: 'Integer(n)' },
  'Float': { type: 'class', doc: 'Float', docCn: '浮点数', signature: 'Float(3.14)' },
  // Polynomials
  'Poly': { type: 'class', doc: 'Polynomial class', docCn: '多项式类', signature: 'Poly(expr, x)' },
  'degree': { type: 'function', doc: 'Polynomial degree', docCn: '多项式次数', signature: 'degree(poly)' },
  'coeffs': { type: 'function', doc: 'Polynomial coefficients', docCn: '多项式系数', signature: 'coeffs(poly)' },
  'roots': { type: 'function', doc: 'Polynomial roots', docCn: '多项式根', signature: 'roots(poly)' },
  // Combinatorics
  'factorial': { type: 'function', doc: 'Factorial', docCn: '阶乘', signature: 'factorial(n)' },
  'binomial': { type: 'function', doc: 'Binomial coefficient', docCn: '二项式系数', signature: 'binomial(n, k)' },
  'gamma': { type: 'function', doc: 'Gamma function', docCn: '伽马函数', signature: 'gamma(x)' },
  // Substitution
  'subs': { type: 'function', doc: 'Substitution', docCn: '代入替换', signature: 'expr.subs(x, 1)' },
  // Evaluation
  'evalf': { type: 'function', doc: 'Evaluate to float', docCn: '数值求值', signature: 'expr.evalf()' },
  'N': { type: 'function', doc: 'Evaluate to float', docCn: '数值求值', signature: 'N(expr)' },
  // Assumptions
  'ask': { type: 'function', doc: 'Query assumptions', docCn: '查询假设', signature: 'ask(Q.positive(x))' },
  'Q': { type: 'class', doc: 'Assumptions query', docCn: '假设查询器', signature: 'Q.positive, Q.real, Q.integer' },
  // Printing
  'pprint': { type: 'function', doc: 'Pretty print', docCn: '美化打印', signature: 'pprint(expr)' },
  'latex': { type: 'function', doc: 'LaTeX output', docCn: 'LaTeX输出', signature: 'latex(expr)' },
  'pretty': { type: 'function', doc: 'Pretty string', docCn: '美化字符串', signature: 'pretty(expr)' },
  // Sets
  'Interval': { type: 'class', doc: 'Interval', docCn: '区间', signature: 'Interval(0, 1)' },
  'Union': { type: 'function', doc: 'Set union', docCn: '集合并集', signature: 'Union(set1, set2)' },
  'Intersection': { type: 'function', doc: 'Set intersection', docCn: '集合交集', signature: 'Intersection(set1, set2)' },
  'Complement': { type: 'function', doc: 'Set complement', docCn: '集合补集', signature: 'Complement(set1, set2)' },
  // Logic
  'And': { type: 'function', doc: 'Logical AND', docCn: '逻辑与', signature: 'And(x > 0, y < 1)' },
  'Or': { type: 'function', doc: 'Logical OR', docCn: '逻辑或', signature: 'Or(x < 0, x > 1)' },
  'Not': { type: 'function', doc: 'Logical NOT', docCn: '逻辑非', signature: 'Not(x == 0)' },
  'Xor': { type: 'function', doc: 'Logical XOR', docCn: '逻辑异或', signature: 'Xor(x, y)' },
  // Relations
  'Eq': { type: 'function', doc: 'Equality', docCn: '等于', signature: 'Eq(x, y)' },
  'Ne': { type: 'function', doc: 'Inequality', docCn: '不等于', signature: 'Ne(x, y)' },
  'Lt': { type: 'function', doc: 'Less than', docCn: '小于', signature: 'Lt(x, y)' },
  'Le': { type: 'function', doc: 'Less or equal', docCn: '小于等于', signature: 'Le(x, y)' },
  'Gt': { type: 'function', doc: 'Greater than', docCn: '大于', signature: 'Gt(x, y)' },
  'Ge': { type: 'function', doc: 'Greater or equal', docCn: '大于等于', signature: 'Ge(x, y)' },
  // Vector operations
  'MatrixSymbol': { type: 'class', doc: 'Matrix symbol', docCn: '矩阵符号', signature: 'MatrixSymbol("M", n, m)' },
  'dot': { type: 'function', doc: 'Dot product', docCn: '点积', signature: 'dot(v1, v2)' },
  'cross': { type: 'function', doc: 'Cross product', docCn: '叉积', signature: 'cross(v1, v2)' },
  'norm': { type: 'function', doc: 'Vector norm', docCn: '向量范数', signature: 'norm(v)' },
  // Complex numbers
  're': { type: 'function', doc: 'Real part', docCn: '实部', signature: 're(z)' },
  'im': { type: 'function', doc: 'Imaginary part', docCn: '虚部', signature: 'im(z)' },
  'arg': { type: 'function', doc: 'Argument', docCn: '辐角', signature: 'arg(z)' },
  'conjugate': { type: 'function', doc: 'Complex conjugate', docCn: '共轭复数', signature: 'conjugate(z)' },
  'abs': { type: 'function', doc: 'Absolute value', docCn: '绝对值', signature: 'abs(x)' },
  // Piecewise
  'Piecewise': { type: 'class', doc: 'Piecewise function', docCn: '分段函数', signature: 'Piecewise((expr1, cond1), (expr2, cond2))' },
  'condense': { type: 'function', doc: 'Condense piecewise', docCn: '压缩分段', signature: 'condense(expr)' },
  // Other
  'floor': { type: 'function', doc: 'Floor function', docCn: '向下取整', signature: 'floor(x)' },
  'ceiling': { type: 'function', doc: 'Ceiling function', docCn: '向上取整', signature: 'ceiling(x)' },
  'sign': { type: 'function', doc: 'Sign function', docCn: '符号函数', signature: 'sign(x)' },
  'Max': { type: 'function', doc: 'Maximum', docCn: '最大值', signature: 'Max(x, y, z)' },
  'Min': { type: 'function', doc: 'Minimum', docCn: '最小值', signature: 'Min(x, y, z)' },
  'Sum': { type: 'class', doc: 'Sum class', docCn: '求和类', signature: 'Sum(expr, (n, a, b))' },
  'Product': { type: 'class', doc: 'Product class', docCn: '求积类', signature: 'Product(expr, (n, a, b))' },
};

document.addEventListener("DOMContentLoaded", function () {
  applyTheme(isDarkTheme);
  addNewParagraph('text');
  bindEvents();
  animateMathBackground();
});

function bindEvents() {
  // Toolbar button events
  document.getElementById('addTextBtn').addEventListener('click', () => addNewParagraph('text'));
  document.getElementById('addSymPyBtn').addEventListener('click', () => addNewParagraph('sympy'));
  document.getElementById('addLatexBtn').addEventListener('click', () => addNewParagraph('latex'));

  document.getElementById('boldBtn').addEventListener('click', () => formatText('bold'));
  document.getElementById('italicBtn').addEventListener('click', () => formatText('italic'));

  document.getElementById('integralBtn').addEventListener('click', () => insertSymbol('\\int_{}^{}'));
  document.getElementById('sumBtn').addEventListener('click', () => insertSymbol('\\sum_{}^{}'));
  document.getElementById('limitBtn').addEventListener('click', () => insertSymbol('\\lim_{}'));
  document.getElementById('diffBtn').addEventListener('click', () => insertSymbol('\\frac{d}{dx}'));

  document.getElementById('sympyDiffBtn').addEventListener('click', () => insertSymPyTemplate('diff'));
  document.getElementById('sympyIntegralBtn').addEventListener('click', () => insertSymPyTemplate('integral'));
  document.getElementById('sympyLimitBtn').addEventListener('click', () => insertSymPyTemplate('limit'));
  document.getElementById('sympySimplifyBtn').addEventListener('click', () => insertSymPyTemplate('simplify'));

  document.getElementById('fracBtn').addEventListener('click', () => insertLatex('\\frac{}{}'));
  document.getElementById('sqrtBtn').addEventListener('click', () => insertLatex('\\sqrt{}'));
  document.getElementById('powerBtn').addEventListener('click', () => insertLatex('^{}'));
  document.getElementById('matrixBtn').addEventListener('click', () => insertLatex('\\begin{pmatrix} & \\\\ & \\end{pmatrix}'));

  document.getElementById('runAllBtn').addEventListener('click', runAllSymPyCode);
  document.getElementById('moveUpBtn').addEventListener('click', () => {
    const activeParagraph = document.querySelector('.paragraph.active');
    if (activeParagraph) moveParagraph(activeParagraph, 'up');
  });
  document.getElementById('moveDownBtn').addEventListener('click', () => {
    const activeParagraph = document.querySelector('.paragraph.active');
    if (activeParagraph) moveParagraph(activeParagraph, 'down');
  });
  document.getElementById('deleteBtn').addEventListener('click', deleteActiveParagraph);
  document.getElementById('toggleMinimalBtn').addEventListener('click', () => {
    toggleAllMinimalMode();
  });
  document.getElementById('saveJsonBtn').addEventListener('click', saveAsJson);
  document.getElementById('loadBtn').addEventListener('click', () => document.getElementById('fileInput').click());

  document.getElementById('fileInput').addEventListener('change', loadDocument);
  document.getElementById('toggleThemeBtn').addEventListener('click', toggleTheme);

  document.getElementById('formulaEditorBtn').addEventListener('click', toggleFormulaEditor);
  document.getElementById('closeFormulaEditorBtn').addEventListener('click', closeFormulaEditor);

  // Formula editor category buttons
  document.querySelectorAll('.formula-editor-body .formula-category-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      selectedFormulaCategory = btn.dataset.category;
      renderFormulaSymbols();
    });
  });

  // Click on other areas of the document to deactivate
  document.addEventListener('click', function (e) {
    if (!e.target.closest('.paragraph') && !e.target.closest('#toolbar') && !e.target.closest('#header') && !e.target.closest('#formulaEditorPanel')) {
      document.querySelectorAll('.paragraph').forEach((p) => {
        p.classList.remove('active');
      });
      activeParagraphId = null;
    }
  });

  // Keyboard shortcuts
  document.addEventListener('keydown', function (e) {
    const activeParagraph = document.querySelector('.paragraph.active');
    if (!activeParagraph) return;

    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      if (activeParagraph.dataset.paragraphType === 'sympy') {
        runSymPyCode(activeParagraph);
        e.preventDefault();
      }
    }

    if ((e.key === 'A' || e.key === 'a') && e.ctrlKey && e.shiftKey) {
      addParagraphAbove(activeParagraph, 'text');
      e.preventDefault();
    }

    if ((e.key === 'S' || e.key === 's') && e.ctrlKey && e.shiftKey) {
      addParagraphAbove(activeParagraph, 'sympy');
      e.preventDefault();
    }

    if ((e.key === 'L' || e.key === 'l') && e.ctrlKey && e.shiftKey) {
      addParagraphAbove(activeParagraph, 'latex');
      e.preventDefault();
    }

    if ((e.key === 'B' || e.key === 'b') && e.ctrlKey && !e.shiftKey) {
      addParagraphBelow(activeParagraph, 'text');
      e.preventDefault();
    }

    if ((e.key === 'B' || e.key === 'b') && e.ctrlKey && e.shiftKey) {
      addParagraphBelow(activeParagraph, 'sympy');
      e.preventDefault();
    }

    if ((e.key === 'B' || e.key === 'b') && e.ctrlKey && e.altKey) {
      addParagraphBelow(activeParagraph, 'latex');
      e.preventDefault();
    }

    if (e.key === 'ArrowUp' && (e.ctrlKey || e.metaKey)) {
      moveParagraph(activeParagraph, 'up');
      e.preventDefault();
    }

    if (e.key === 'ArrowDown' && (e.ctrlKey || e.metaKey)) {
      moveParagraph(activeParagraph, 'down');
      e.preventDefault();
    }

    if ((e.key === 'D' || e.key === 'd') && e.ctrlKey && !e.shiftKey) {
      deleteParagraph(activeParagraph);
      e.preventDefault();
    }

    if ((e.key === 'M' || e.key === 'm') && e.ctrlKey && !e.shiftKey) {
      toggleMinimalMode(activeParagraph);
      e.preventDefault();
    }
  });
}

function addNewParagraph(type = 'text') {
  const contentArea = document.getElementById('content-area');
  const newParagraph = createParagraphElement(paragraphCounter, type);
  contentArea.appendChild(newParagraph);
  setupParagraphEvents(newParagraph);
  setActiveParagraph(newParagraph);
  paragraphCounter++;

  newParagraph.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function createParagraphElement(id, type) {
  const paragraph = document.createElement('div');
  paragraph.className = `paragraph ${type}-paragraph`;
  paragraph.dataset.paragraphId = id;
  paragraph.dataset.paragraphType = type;

  // Paragraph type indicator
  const indicator = document.createElement('div');
  indicator.className = 'paragraph-type-indicator';
  indicator.innerHTML = type === 'text' ? 'T' : type === 'sympy' ? 'S' : 'L';
  paragraph.appendChild(indicator);

  // Minimal mode toggle button
  const minimalToggle = document.createElement('button');
  minimalToggle.className = 'minimal-mode-toggle';
  minimalToggle.title = '切换最简模式';
  minimalToggle.innerHTML = '<i class="fas fa-eye-slash"></i>';
  paragraph.appendChild(minimalToggle);

  if (type === 'text') {
    paragraph.innerHTML += `
      <textarea placeholder="输入文本..." class="text-input"></textarea>
      <div class="text-render"></div>
    `;
  } else if (type === 'sympy') {
    paragraph.innerHTML += `
      <div class="code-header">
        <span class="code-label">SymPy 代码</span>
        <div class="code-actions">
          <button class="toggle-code-btn" title="显示/隐藏代码">
            <i class="fas fa-eye"></i>
          </button>
          <button class="run-button" title="运行代码">
            <i class="fas fa-play"></i> 运行
          </button>
        </div>
      </div>
      <textarea placeholder="输入 SymPy 代码..." class="code-input"></textarea>
      <div class="code-output"></div>
    `;
  } else if (type === 'latex') {
    paragraph.innerHTML += `
      <div class="latex-header">
        <span class="latex-label">LaTeX 公式</span>
        <div class="latex-actions">
          <button class="toggle-latex-btn" title="显示/隐藏代码">
            <i class="fas fa-eye"></i>
          </button>
        </div>
      </div>
      <textarea placeholder="输入 LaTeX 公式..." class="latex-input"></textarea>
      <div class="latex-render"></div>
    `;
  }

  return paragraph;
}

function setupParagraphEvents(paragraph) {
  const type = paragraph.dataset.paragraphType;

  // Click to activate paragraph
  paragraph.addEventListener('click', function (e) {
    if (!e.target.closest('button')) {
      setActiveParagraph(paragraph);
    }
  });

  // Minimal mode toggle button
  const minimalToggle = paragraph.querySelector('.minimal-mode-toggle');
  minimalToggle.addEventListener('click', function (e) {
    e.stopPropagation();
    toggleMinimalMode(paragraph);
  });

  if (type === 'text') {
    const textarea = paragraph.querySelector('.text-input');

    textarea.addEventListener('input', function () {
      renderMarkdown(paragraph);
    });

    textarea.addEventListener('focus', function () {
      setActiveParagraph(paragraph);
    });
  } else if (type === 'sympy') {
    const textarea = paragraph.querySelector('.code-input');
    const toggleBtn = paragraph.querySelector('.toggle-code-btn');
    const runBtn = paragraph.querySelector('.run-button');

    // Setup autocomplete for SymPy code input
    setupAutocomplete(textarea);

    textarea.addEventListener('focus', function () {
      setActiveParagraph(paragraph);
    });

    toggleBtn.addEventListener('click', function (e) {
      e.stopPropagation();
      const input = paragraph.querySelector('.code-input');
      input.classList.toggle('collapsed');
      const icon = toggleBtn.querySelector('i');
      icon.className = input.classList.contains('collapsed') ? 'fas fa-eye-slash' : 'fas fa-eye';
    });

    runBtn.addEventListener('click', function (e) {
      e.stopPropagation();
      runSymPyCode(paragraph);
    });
  } else if (type === 'latex') {
    const textarea = paragraph.querySelector('.latex-input');
    const toggleBtn = paragraph.querySelector('.toggle-latex-btn');

    textarea.addEventListener('input', function () {
      debounce(() => {
        renderLatex(paragraph);
      }, 500)();
    });

    textarea.addEventListener('focus', function () {
      setActiveParagraph(paragraph);
    });

    toggleBtn.addEventListener('click', function (e) {
      e.stopPropagation();
      const input = paragraph.querySelector('.latex-input');
      input.classList.toggle('collapsed');
      const icon = toggleBtn.querySelector('i');
      icon.className = input.classList.contains('collapsed') ? 'fas fa-eye-slash' : 'fas fa-eye';
    });
  }
}

function setActiveParagraph(paragraph) {
  document.querySelectorAll('.paragraph').forEach((p) => {
    p.classList.remove('active');
  });
  paragraph.classList.add('active');
  activeParagraphId = paragraph.dataset.paragraphId;

  const textarea = paragraph.querySelector('textarea');
  if (textarea) {
    textarea.focus();
    paragraph.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }
}

function renderMarkdown(paragraph) {
  const textarea = paragraph.querySelector('.text-input');
  const render = paragraph.querySelector('.text-render');

  if (!textarea || !render) return;

  const content = textarea.value;

  let html = content
    // Headers
    .replace(/^###### (.*$)/gim, '<h6>$1</h6>')
    .replace(/^##### (.*$)/gim, '<h5>$1</h5>')
    .replace(/^#### (.*$)/gim, '<h4>$1</h4>')
    .replace(/^### (.*$)/gim, '<h3>$1</h3>')
    .replace(/^## (.*$)/gim, '<h2>$1</h2>')
    .replace(/^# (.*$)/gim, '<h1>$1</h1>')
    // Bold
    .replace(/__(.*?)__/gim, '<strong>$1</strong>')
    .replace(/\*\*(.*?)\*\*/gim, '<strong>$1</strong>')
    // Italic
    .replace(/_(.*?)_/gim, '<em>$1</em>')
    .replace(/\*(.*?)\*/gim, '<em>$1</em>')
    // Underline
    .replace(/~~(.*?)~~/gim, '<u>$1</u>')
    // Code
    .replace(/`(.*?)`/gim, '<code>$1</code>')
    // Line breaks
    .replace(/\n/gim, '<br>')
    // Links
    .replace(/\[([^\]]+)\]\(([^)]+)\)/gim, '<a href="$2" target="_blank">$1</a>')
    // Unordered lists
    .replace(/^\s*[-*] (.*$)/gim, '<li>$1</li>')
    // Ordered lists
    .replace(/^\s*\d+\. (.*$)/gim, '<li>$1</li>')
    // Handle list wrapping
    .replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>')
    // Quotes
    .replace(/^> (.*$)/gim, '<blockquote>$1</blockquote>')
    // Strike through
    .replace(/~~(.*?)~~/gim, '<del>$1</del>');

  // Handle multiple consecutive line breaks (paragraphs)
  html = html.replace(/(<br>\s*){2,}/gim, '</p><p>');
  html = '<p>' + html + '</p>';

  render.innerHTML = html || '<span style="color: #999;">空段落</span>';
}

function runSymPyCode(paragraph) {
  const textarea = paragraph.querySelector('.code-input');
  const output = paragraph.querySelector('.code-output');
  const code = textarea.value.trim();

  if (!code) {
    showNotification('请输入 SymPy 代码', 'error');
    return;
  }

  output.innerHTML = '<div class="computing"><div class="loading-spinner"></div><div class="loading-text">计算中...</div></div>';

  fetch('/api/sympy_execute', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ code: code }),
  })
    .then((response) => response.json())
    .then((data) => {
      if (data.success) {
        // If there is LaTeX result, prioritize displaying LaTeX
        if (data.latex) {
          output.innerHTML = `$$${data.latex}$$<div style="font-size: 12px; color: #999; margin-top: 8px;">${data.result}</div>`;
        } else {
          output.innerHTML = data.result;
        }

        if (window.MathJax) {
          MathJax.typesetPromise([output]).then(() => {
            output.classList.add('success-flash');
            setTimeout(() => {
              output.classList.remove('success-flash');
            }, 500);
          });
        } else {
          output.classList.add('success-flash');
          setTimeout(() => {
            output.classList.remove('success-flash');
          }, 500);
        }
      } else {
        output.innerHTML = `<div class="error">${data.error}</div>`;
      }
    })
    .catch((error) => {
      output.innerHTML = `<div class="error">错误: ${error.message}</div>`;
    });
}

function renderLatex(paragraph) {
  const textarea = paragraph.querySelector('.latex-input');
  const render = paragraph.querySelector('.latex-render');
  const latex = textarea.value.trim();

  if (!latex) {
    render.innerHTML = '';
    return;
  }

  render.innerHTML = `$$${latex}$$`;

  if (window.MathJax) {
    MathJax.typesetPromise([render]).catch((err) => {
      render.innerHTML = `<div class="error">LaTeX 渲染错误: ${err.message}</div>`;
    });
  }
}

function runAllSymPyCode() {
  document.querySelectorAll('.sympy-paragraph').forEach((paragraph) => {
    const textarea = paragraph.querySelector('.code-input');
    if (textarea.value.trim()) {
      runSymPyCode(paragraph);
    }
  });
}

function deleteActiveParagraph() {
  const activeParagraph = document.querySelector('.paragraph.active');
  if (activeParagraph) {
    deleteParagraph(activeParagraph);
  } else {
    showNotification('请先选择要删除的段落', 'error');
  }
}

function formatText(format) {
  const activeParagraph = document.querySelector('.paragraph.active');
  if (activeParagraph && activeParagraph.dataset.paragraphType === 'text') {
    const textarea = activeParagraph.querySelector('.text-input');
    const start = textarea.selectionStart;
    const end = textarea.selectionEnd;
    const selectedText = textarea.value.substring(start, end);

    let wrapper = '';
    switch (format) {
      case 'bold':
        wrapper = '**';
        break;
      case 'italic':
        wrapper = '*';
        break;
    }

    textarea.value = textarea.value.substring(0, start) + wrapper + selectedText + wrapper + textarea.value.substring(end);
    textarea.focus();
    textarea.dispatchEvent(new Event('input'));
  } else {
    showNotification('请先选择一个文本段落', 'error');
  }
}

function insertSymbol(symbol) {
  const activeParagraph = document.querySelector('.paragraph.active');
  if (activeParagraph) {
    const input = activeParagraph.querySelector('.latex-input');
    if (input) {
      insertAtCursor(input, symbol);
      input.focus();
      input.dispatchEvent(new Event('input'));
    } else {
      showNotification('请先选择一个 LaTeX 段落', 'error');
    }
  }
}

function insertLatex(latex) {
  const activeParagraph = document.querySelector('.paragraph.active');
  if (activeParagraph) {
    const input = activeParagraph.querySelector('.latex-input');
    if (input) {
      insertAtCursor(input, latex);
      input.focus();
      input.dispatchEvent(new Event('input'));
    } else {
      showNotification('请先选择一个 LaTeX 段落', 'error');
    }
  }
}

function insertSymPyTemplate(type) {
  const activeParagraph = document.querySelector('.paragraph.active');
  if (activeParagraph) {
    const input = activeParagraph.querySelector('.code-input');
    if (input) {
      let template = '';
      switch (type) {
        case 'diff':
          template = 'diff(expr, x)';
          break;
        case 'integral':
          template = 'integrate(expr, x)';
          break;
        case 'limit':
          template = 'limit(expr, x, 0)';
          break;
        case 'simplify':
          template = 'simplify(expr)';
          break;
      }
      insertAtCursor(input, template);
      input.focus();
      input.dispatchEvent(new Event('input'));
    } else {
      showNotification('请先选择一个 SymPy 段落', 'error');
    }
  }
}

// Insert text at cursor position
function insertAtCursor(input, text) {
  const start = input.selectionStart;
  const end = input.selectionEnd;
  const value = input.value;

  input.value = value.substring(0, start) + text + value.substring(end);
  input.selectionStart = input.selectionEnd = start + text.length;
}

function debounce(func, wait) {
  let timeout;
  return function executedFunction(...args) {
    const later = () => {
      clearTimeout(timeout);
      func(...args);
    };
    clearTimeout(timeout);
    timeout = setTimeout(later, wait);
  };
}

function saveAsJson() {
  const paragraphs = [];
  document.querySelectorAll('.paragraph').forEach((p) => {
    const type = p.dataset.paragraphType;
    const data = {
      type: type,
      content: '',
    };

    if (type === 'text') {
      data.content = p.querySelector('.text-input').value;
    } else if (type === 'sympy') {
      data.content = p.querySelector('.code-input').value;
    } else if (type === 'latex') {
      data.content = p.querySelector('.latex-input').value;
    }

    paragraphs.push(data);
  });

  const json = JSON.stringify(paragraphs, null, 2);
  const blob = new Blob([json], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'document.json';
  a.click();
  URL.revokeObjectURL(url);

  showNotification('文档已保存为 JSON', 'success');
}

function loadDocument(e) {
  const file = e.target.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = function (event) {
    const content = event.target.result;

    if (file.name.endsWith('.json')) {
      try {
        const paragraphs = JSON.parse(content);
        const contentArea = document.getElementById('content-area');
        contentArea.innerHTML = '';
        paragraphCounter = 0;

        paragraphs.forEach((p) => {
          const newParagraph = createParagraphElement(paragraphCounter, p.type);
          contentArea.appendChild(newParagraph);
          setupParagraphEvents(newParagraph);

          if (p.type === 'text') {
            newParagraph.querySelector('.text-input').value = p.content;
            renderMarkdown(newParagraph);
          } else if (p.type === 'sympy') {
            newParagraph.querySelector('.code-input').value = p.content;
          } else if (p.type === 'latex') {
            newParagraph.querySelector('.latex-input').value = p.content;
            renderLatex(newParagraph);
          }

          paragraphCounter++;
        });

        showNotification('文档加载成功', 'success');
      } catch (error) {
        showNotification('JSON 解析错误', 'error');
      }
    }
  };

  reader.readAsText(file);
  e.target.value = '';
}

function toggleTheme() {
  isDarkTheme = !isDarkTheme;
  localStorage.setItem('darkTheme', isDarkTheme);
  applyTheme(isDarkTheme);
}

function applyTheme(isDark) {
  if (isDark) {
    document.documentElement.style.setProperty('--jp-layout-color0', '#1e1e1e');
    document.documentElement.style.setProperty('--jp-layout-color1', '#2d2d2d');
    document.documentElement.style.setProperty('--jp-layout-color2', '#3d3d3d');
    document.documentElement.style.setProperty('--jp-ui-font-color0', '#e0e0e0');
    document.documentElement.style.setProperty('--jp-ui-font-color1', '#b0b0b0');
    document.documentElement.style.setProperty('--jp-border-color0', '#3d3d3d');
    document.documentElement.style.setProperty('--jp-border-color1', '#4d4d4d');
    document.querySelector('.a4-page').style.backgroundColor = '#1e1e1e';
    document.querySelector('.a4-page').style.color = '#e0e0e0';
  } else {
    document.documentElement.style.setProperty('--jp-layout-color0', 'white');
    document.documentElement.style.setProperty('--jp-layout-color1', '#f7f7f7');
    document.documentElement.style.setProperty('--jp-layout-color2', '#e7e7e7');
    document.documentElement.style.setProperty('--jp-ui-font-color0', 'rgba(0, 0, 0, 0.87)');
    document.documentElement.style.setProperty('--jp-ui-font-color1', 'rgba(0, 0, 0, 0.54)');
    document.documentElement.style.setProperty('--jp-border-color0', '#e0e0e0');
    document.documentElement.style.setProperty('--jp-border-color1', '#bdbdbd');
    document.querySelector('.a4-page').style.backgroundColor = 'white';
    document.querySelector('.a4-page').style.color = 'rgba(0, 0, 0, 0.87)';
  }
}

function showNotification(message, type = 'success') {
  const notification = document.getElementById('notification');
  const notificationText = document.getElementById('notification-text');
  const icon = notification.querySelector('i');

  notification.className = `notification notification-${type}`;
  notificationText.textContent = message;

  if (type === 'success') {
    icon.className = 'fas fa-check-circle';
  } else {
    icon.className = 'fas fa-exclamation-circle';
  }

  notification.classList.remove('notification-hide');
  notification.classList.add('notification-show');

  setTimeout(() => {
    notification.classList.remove('notification-show');
    notification.classList.add('notification-hide');
  }, 3000);
}

function animateMathBackground() {
  const elements = document.querySelectorAll('.math-bg-element');
  elements.forEach((el) => {
    const delay = Math.random() * 2;
    const duration = 15 + Math.random() * 10;
    el.style.animation = `float ${duration}s ease-in-out ${delay}s infinite`;
  });
}

function addParagraphAbove(paragraph, type) {
  const contentArea = document.getElementById('content-area');
  const newParagraph = createParagraphElement(paragraphCounter, type);
  contentArea.insertBefore(newParagraph, paragraph);
  setupParagraphEvents(newParagraph);
  setActiveParagraph(newParagraph);
  paragraphCounter++;
}

function addParagraphBelow(paragraph, type) {
  const contentArea = document.getElementById('content-area');
  const newParagraph = createParagraphElement(paragraphCounter, type);

  if (paragraph.nextSibling) {
    contentArea.insertBefore(newParagraph, paragraph.nextSibling);
  } else {
    contentArea.appendChild(newParagraph);
  }

  setupParagraphEvents(newParagraph);
  setActiveParagraph(newParagraph);
  paragraphCounter++;
}

function moveParagraph(paragraph, direction) {
  const contentArea = document.getElementById('content-area');

  if (direction === 'up') {
    const prevSibling = paragraph.previousElementSibling;
    if (prevSibling) {
      // Add upward movement animation
      paragraph.classList.add('moving-up');

      setTimeout(() => {
        contentArea.insertBefore(paragraph, prevSibling);

        setTimeout(() => {
          paragraph.classList.remove('moving-up');
        }, 300);

      }, 50);
    }
  } else if (direction === 'down') {
    const nextSibling = paragraph.nextElementSibling;
    if (nextSibling) {
      // Add downward movement animation
      paragraph.classList.add('moving-down');

      setTimeout(() => {
        contentArea.insertBefore(nextSibling, paragraph);

        setTimeout(() => {
          paragraph.classList.remove('moving-down');
        }, 300);

      }, 50);
    }
  }
}

function toggleMinimalMode(paragraph) {
  paragraph.classList.toggle('minimal-mode');
  const isMinimal = paragraph.classList.contains('minimal-mode');
  const toggleBtn = paragraph.querySelector('.minimal-mode-toggle');
  const icon = toggleBtn.querySelector('i');

  if (isMinimal) {
    icon.className = 'fas fa-eye';
  } else {
    icon.className = 'fas fa-eye-slash';
  }
}

function deleteParagraph(paragraph) {
  const contentArea = document.getElementById('content-area');
  if (contentArea.children.length > 1) {
    const prevSibling = paragraph.previousElementSibling;

    paragraph.remove();

    if (prevSibling) {
      setActiveParagraph(prevSibling);
    } else {
      const firstParagraph = contentArea.querySelector('.paragraph');
      if (firstParagraph) {
        setActiveParagraph(firstParagraph);
      }
    }

  } else {
    showNotification('至少保留一个段落', 'error');
  }
}

function toggleAllMinimalMode() {
  const paragraphs = document.querySelectorAll('.paragraph');
  const allMinimal = Array.from(paragraphs).every(p => p.classList.contains('minimal-mode'));

  paragraphs.forEach(paragraph => {
    if (allMinimal) {
      // If all are in minimal mode, exit minimal mode
      paragraph.classList.remove('minimal-mode');
      const toggleBtn = paragraph.querySelector('.minimal-mode-toggle');
      const icon = toggleBtn.querySelector('i');
      icon.className = 'fas fa-eye-slash';
    } else {
      // Otherwise enter minimal mode for all
      paragraph.classList.add('minimal-mode');
      const toggleBtn = paragraph.querySelector('.minimal-mode-toggle');
      const icon = toggleBtn.querySelector('i');
      icon.className = 'fas fa-eye';
    }
  });
}

// Auto-complete functionality
let currentAutocompleteIndex = 0;
let currentSuggestions = [];

function setupAutocomplete(textarea) {
  // Wrap textarea in autocomplete container
  const wrapper = textarea.parentElement;
  if (!wrapper.classList.contains('autocomplete')) {
    wrapper.classList.add('autocomplete');
  }

  // Create suggestions container if it doesn't exist
  let suggestionsContainer = wrapper.querySelector('.autocomplete-suggestions');
  if (!suggestionsContainer) {
    suggestionsContainer = document.createElement('div');
    suggestionsContainer.className = 'autocomplete-suggestions';
    wrapper.appendChild(suggestionsContainer);
  }

  // Remove existing event listeners by cloning
  const newTextarea = textarea.cloneNode(true);
  textarea.parentNode.replaceChild(newTextarea, textarea);

  // Add input event listener
  newTextarea.addEventListener('input', function (e) {
    handleAutocompleteInput(newTextarea, suggestionsContainer);
  });

  // Add keyboard event listener
  newTextarea.addEventListener('keydown', function (e) {
    handleAutocompleteKeydown(e, newTextarea, suggestionsContainer);
  });

  // Hide suggestions when clicking outside
  document.addEventListener('click', function (e) {
    if (!wrapper.contains(e.target)) {
      hideSuggestions(suggestionsContainer);
    }
  });

  return newTextarea;
}

function handleAutocompleteInput(textarea, suggestionsContainer) {
  const value = textarea.value;
  const cursorPos = textarea.selectionStart;

  // Find the current word being typed
  const beforeCursor = value.substring(0, cursorPos);
  const wordMatch = beforeCursor.match(/([a-zA-Z_][a-zA-Z0-9_]*)$/);

  if (!wordMatch) {
    hideSuggestions(suggestionsContainer);
    return;
  }

  const currentWord = wordMatch[1];
  const wordStartPos = cursorPos - currentWord.length;

  // Get matching suggestions
  const suggestions = getMatchingSuggestions(currentWord);

  if (suggestions.length === 0 || currentWord.length < 1) {
    hideSuggestions(suggestionsContainer);
    return;
  }

  currentSuggestions = suggestions;
  currentAutocompleteIndex = 0;  // Default select first suggestion

  showSuggestions(suggestionsContainer, suggestions, currentWord, textarea, wordStartPos);
}

function getMatchingSuggestions(prefix) {
  const suggestions = [];
  const lowerPrefix = prefix.toLowerCase();

  for (const [name, info] of Object.entries(sympyCompletions)) {
    if (name.toLowerCase().startsWith(lowerPrefix)) {
      suggestions.push({ name, ...info });
    }
  }

  // Sort by name length (shorter matches first)
  suggestions.sort((a, b) => a.name.length - b.name.length);
  // Limit to 10 suggestions
  return suggestions.slice(0, 10);
}

function showSuggestions(container, suggestions, currentWord, textarea, wordStartPos) {
  container.innerHTML = '';
  container.classList.add('visible');

  suggestions.forEach((suggestion, index) => {
    const div = document.createElement('div');

    div.className = 'autocomplete-suggestion';

    // Highlight the matching part
    const matchIndex = suggestion.name.toLowerCase().indexOf(currentWord.toLowerCase());
    const beforeMatch = suggestion.name.substring(0, matchIndex);
    const match = suggestion.name.substring(matchIndex, matchIndex + currentWord.length);
    const afterMatch = suggestion.name.substring(matchIndex + currentWord.length);

    div.innerHTML = `

      <div class="autocomplete-suggestion-content">
        <span class="autocomplete-suggestion-name">
          ${beforeMatch}<span class="suggestion-match">${match}</span>${afterMatch}
        </span>
        <span class="autocomplete-suggestion-signature">${suggestion.signature}</span>
      </div>

      <div class="autocomplete-suggestion-meta">
        <span class="autocomplete-suggestion-doc-cn">${suggestion.docCn || suggestion.doc}</span>
        <span class="suggestion-type ${suggestion.type}">${suggestion.type}</span>
      </div>

    `;

    div.addEventListener('click', function () {
      applySuggestion(textarea, suggestion.name, wordStartPos, textarea.selectionEnd);
      hideSuggestions(container);
      textarea.focus();
    });
    container.appendChild(div);
  });


  // Default select first suggestion
  if (suggestions.length > 0) {
    const firstItem = container.querySelector('.autocomplete-suggestion');
    if (firstItem) {
      firstItem.classList.add('active');
    }
  }
}

function hideSuggestions(container) {
  container.classList.remove('visible');
  currentSuggestions = [];
  currentAutocompleteIndex = -1;
}

function handleAutocompleteKeydown(e, textarea, suggestionsContainer) {
  const suggestionItems = suggestionsContainer.querySelectorAll('.autocomplete-suggestion');

  if (!suggestionsContainer.classList.contains('visible')) {
    return;
  }

  switch (e.key) {
    case 'Escape':
      e.preventDefault();
      hideSuggestions(suggestionsContainer);
      break;
    case 'ArrowDown':
      e.preventDefault();
      currentAutocompleteIndex = Math.min(currentAutocompleteIndex + 1, suggestionItems.length - 1);
      updateActiveSuggestion(suggestionItems);
      break;

    case 'ArrowUp':
      e.preventDefault();
      currentAutocompleteIndex = Math.max(currentAutocompleteIndex - 1, -1);
      updateActiveSuggestion(suggestionItems);
      break;

    case 'Enter':
    case 'Tab':
      if (currentAutocompleteIndex >= 0 && currentSuggestions[currentAutocompleteIndex]) {
        e.preventDefault();
        const value = textarea.value;
        const cursorPos = textarea.selectionStart;
        const beforeCursor = value.substring(0, cursorPos);
        const wordMatch = beforeCursor.match(/([a-zA-Z_][a-zA-Z0-9_]*)$/);
        const wordStartPos = cursorPos - wordMatch[1].length;

        applySuggestion(textarea, currentSuggestions[currentAutocompleteIndex].name, wordStartPos, cursorPos);
        hideSuggestions(suggestionsContainer);
      } else if (e.key === 'Tab') {
        // If no suggestion is selected, just hide and let tab work normally
        e.preventDefault();
        hideSuggestions(suggestionsContainer);
      }
      break;
  }
}

function updateActiveSuggestion(items) {
  items.forEach((item, index) => {
    if (index === currentAutocompleteIndex) {
      item.classList.add('active');
      item.scrollIntoView({ block: 'nearest' });
    } else {
      item.classList.remove('active');
    }
  });
}

function applySuggestion(textarea, suggestionName, startPos, endPos) {
  const value = textarea.value;
  const newValue = value.substring(0, startPos) + suggestionName + value.substring(endPos);
  textarea.value = newValue;

  // Move cursor after the inserted suggestion
  const newCursorPos = startPos + suggestionName.length;
  textarea.setSelectionRange(newCursorPos, newCursorPos);

  // Trigger input event
  textarea.dispatchEvent(new Event('input'));
}

// Formula Editor Data
const formulaSymbolCategories = {
  operators: [
    { label: '+', latex: '+' },
    { label: '-', latex: '-' },
    { label: '×', latex: '\\times' },
    { label: '÷', latex: '\\div' },
    { label: '±', latex: '\\pm' },
    { label: '∓', latex: '\\mp' },
    { label: '⊕', latex: '\\oplus' },
    { label: '⊗', latex: '\\otimes' },
    { label: '∧', latex: '\\land' },
    { label: '∨', latex: '\\lor' },
    { label: '¬', latex: '\\neg' },
    { label: '∩', latex: '\\cap' },
    { label: '∪', latex: '\\cup' },
    { label: '⊂', latex: '\\subset' },
    { label: '⊃', latex: '\\supset' },
    { label: '∈', latex: '\\in' },
  ],
  relations: [
    { label: '=', latex: '=' },
    { label: '≠', latex: '\\neq' },
    { label: '≈', latex: '\\approx' },
    { label: '≡', latex: '\\equiv' },
    { label: '<', latex: '<' },
    { label: '>', latex: '>' },
    { label: '≤', latex: '\\leq' },
    { label: '≥', latex: '\\geq' },
    { label: '≪', latex: '\\ll' },
    { label: '≫', latex: '\\gg' },
    { label: '⊂', latex: '\\subset' },
    { label: '⊆', latex: '\\subseteq' },
    { label: '⊃', latex: '\\supset' },
    { label: '⊇', latex: '\\supseteq' },
    { label: '∈', latex: '\\in' },
    { label: '∉', latex: '\\notin' },
  ],
  greek: [
    { label: 'α', latex: '\\alpha' },
    { label: 'β', latex: '\\beta' },
    { label: 'γ', latex: '\\gamma' },
    { label: 'δ', latex: '\\delta' },
    { label: 'ε', latex: '\\epsilon' },
    { label: 'θ', latex: '\\theta' },
    { label: 'λ', latex: '\\lambda' },
    { label: 'μ', latex: '\\mu' },
    { label: 'π', latex: '\\pi' },
    { label: 'σ', latex: '\\sigma' },
    { label: 'φ', latex: '\\phi' },
    { label: 'ω', latex: '\\omega' },
    { label: 'Δ', latex: '\\Delta' },
    { label: 'Σ', latex: '\\Sigma' },
    { label: 'Ω', latex: '\\Omega' },
    { label: 'Γ', latex: '\\Gamma' },
  ],
  functions: [
    { label: 'sin', latex: '\\sin' },
    { label: 'cos', latex: '\\cos' },
    { label: 'tan', latex: '\\tan' },
    { label: 'cot', latex: '\\cot' },
    { label: 'sec', latex: '\\sec' },
    { label: 'csc', latex: '\\csc' },
    { label: 'ln', latex: '\\ln' },
    { label: 'log', latex: '\\log' },
    { label: 'exp', latex: '\\exp' },
    { label: '√', latex: '\\sqrt{x}' },
    { label: '⌊⌋', latex: '\\lfloor x \\rfloor' },
    { label: '⌈⌉', latex: '\\lceil x \\rceil' },
    { label: 'lim', latex: '\\lim' },
    { label: '∑', latex: '\\sum' },
    { label: '∏', latex: '\\prod' },
    { label: '∫', latex: '\\int' },
  ],
  matrix: [
    { label: '( )', latex: '\\begin{pmatrix} a & b \\\\ c & d \\end{pmatrix}' },
    { label: '[ ]', latex: '\\begin{bmatrix} a & b \\\\ c & d \\end{bmatrix}' },
    { label: '| |', latex: '\\begin{vmatrix} a & b \\\\ c & d \\end{vmatrix}' },
    { label: '|| ||', latex: '\\begin{Vmatrix} a & b \\\\ c & d \\end{Vmatrix}' },
    { label: '2×2', latex: '\\begin{pmatrix} a & b \\\\ c & d \\end{pmatrix}' },
    { label: '3×3', latex: '\\begin{pmatrix} a & b & c \\\\ d & e & f \\\\ g & h & i \\end{pmatrix}' },
    { label: '列向量', latex: '\\begin{pmatrix} a \\\\ b \\\\ c \\end{pmatrix}' },
    { label: '行向量', latex: '\\begin{pmatrix} a & b & c \\end{pmatrix}' },
    { label: '零矩阵', latex: '\\mathbf{0}' },
    { label: '单位矩阵', latex: '\\mathbf{I}' },
    { label: '转置', latex: 'A^{T}' },
    { label: '逆矩阵', latex: 'A^{-1}' },
    { label: '伴随', latex: 'A^{*}' },
    { label: '共轭', latex: '\\overline{A}' },
    { label: '迹', latex: '\\operatorname{tr}(A)' },
    { label: '秩', latex: '\\operatorname{rank}(A)' },
  ]
};

let isFormulaEditorVisible = false;
let selectedFormulaCategory = 'operators';
let formulaEditorPosition = { x: 0, y: 0 };
let isDragging = false;
let dragOffset = { x: 0, y: 0 };

// Formula Editor Functions
function toggleFormulaEditor() {
  isFormulaEditorVisible = !isFormulaEditorVisible;
  const panel = document.getElementById('formulaEditorPanel');

  if (isFormulaEditorVisible) {
    panel.style.display = 'flex';
    renderFormulaSymbols();
    makeFormulaEditorDraggable();
  } else {
    panel.style.display = 'none';
  }
}

function closeFormulaEditor() {
  isFormulaEditorVisible = false;
  document.getElementById('formulaEditorPanel').style.display = 'none';
}

function renderFormulaSymbols() {
  const grid = document.getElementById('formulaSymbolGrid');
  const symbols = formulaSymbolCategories[selectedFormulaCategory];

  grid.innerHTML = '';

  symbols.forEach(symbol => {
    const btn = document.createElement('button');
    btn.className = 'latex-symbol-btn';
    btn.title = symbol.label;
    btn.innerHTML = `$$${symbol.latex}$$`;

    btn.addEventListener('click', () => {
      insertFormulaSymbol(symbol.latex);
    });

    grid.appendChild(btn);
  });

  // Render MathJax
  if (window.MathJax) {
    MathJax.typesetPromise([grid]).catch((err) => {
      console.error('MathJax rendering error:', err);
    });
  }

  // Update category buttons
  document.querySelectorAll('.formula-editor-body .formula-category-btn').forEach(btn => {
    btn.classList.remove('active');
    if (btn.dataset.category === selectedFormulaCategory) {
      btn.classList.add('active');
    }
  });
}

function makeFormulaEditorDraggable() {
  const panel = document.getElementById('formulaEditorPanel');
  const header = panel.querySelector('.formula-editor-header');

  header.addEventListener('mousedown', (e) => {
    // Don't start drag if clicking on close button
    if (e.target.closest('.formula-editor-close-btn')) return;

    isDragging = true;
    dragOffset.x = e.clientX - panel.offsetLeft;
    dragOffset.y = e.clientY - panel.offsetTop;
    e.preventDefault();
  });

  document.addEventListener('mousemove', (e) => {
    if (!isDragging) return;

    let newX = e.clientX - dragOffset.x;
    let newY = e.clientY - dragOffset.y;

    // Keep panel within viewport
    const maxX = window.innerWidth - panel.offsetWidth;
    const maxY = window.innerHeight - panel.offsetHeight;

    newX = Math.max(0, Math.min(newX, maxX));
    newY = Math.max(0, Math.min(newY, maxY));

    panel.style.left = newX + 'px';
    panel.style.top = newY + 'px';
    panel.style.right = 'auto';
  });

  document.addEventListener('mouseup', () => {
    isDragging = false;
    header.style.cursor = 'move';
  });
}

function insertFormulaSymbol(latex) {
  const activeParagraph = document.querySelector('.paragraph.active');

  if (!activeParagraph) {
    showNotification('请先选择一个LaTeX段落', 'error');
    return;
  }

  if (activeParagraph.dataset.paragraphType !== 'latex') {
    showNotification('请选择一个LaTeX段落', 'error');
    return;
  }

  const textarea = activeParagraph.querySelector('.latex-input');
  if (textarea) {
    insertAtCursor(textarea, latex);
    textarea.focus();
    textarea.dispatchEvent(new Event('input'));
  }
}