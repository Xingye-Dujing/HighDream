// Global variables
let cellCounter = 0;
const APPLIED_OPS = ["tangent", "linearization", "critical_points", "newton", "arc_length", "volume_disk", "surface_area", "implicit_diff", "parametric_diff", "polar_area", "polar_slope", "taylor", "series_convergence", "improper_integral", "ode", "numerical_integration", "mean_value", "radius_of_convergence", "taylor_error", "complex_simplify", "complex_polar", "complex_roots", "volume_shell", "de_moivre", "iterated_integral", "power_series"];

// Operations whose input is structured text (matrices / vectors), not a rendered
// math expression: these keep the plain <textarea> instead of the MathLive editor.
// Ops whose input is plain multi-line text: equation lists (linear-system) and
// sympy-syntax symbolic matrix expressions (matrix_symbolic), which don't fit a
// rendered matrix editor.
const MATH_TEXT_OPS = new Set(["linear-system", "matrix_symbolic"]);

// Single-matrix linear-algebra ops: use the WYSIWYG editor and get a matrix skeleton.
const MATRIX_SINGLE_OPS = new Set(["ref", "det", "invert", "rank", "LU", "diag", "eigen", "singular", "svd", "schur", "orthogonal", "dependence"]);

// Two-input matrix/vector ops: show a second WYSIWYG field; backend splits on newline.
const MULTI_INPUT_OPS = new Set(["operations", "linear-solver", "projection", "transform-1", "transform-2"]);

// Per-operation LaTeX hint shown as the editor placeholder ("预设题的形式").
const OP_PLACEHOLDER = {
  auto: "\\frac{d}{d\\square}\\square\\ \\ \\int\\square\\, d\\square\\ \\ \\det\\begin{bmatrix}\\square&\\square\\\\\\square&\\square\\end{bmatrix}\\ \\ \\cdots",
  calculus: "\\frac{d}{d\\square}\\left[\\square\\right]\\ \\text{或}\\ \\int_{\\square}^{\\square}\\square\\, d\\square\\ \\text{或}\\ \\lim_{\\square\\to\\square}\\square",
  expr: "\\text{表达式: }\\frac{1}{x}+\\sin x",
  diff: "\\frac{d}{d\\square}\\left[\\square\\right]",
  integral: "\\int \\square\\, d\\square",
  limit: "\\lim_{\\square \\to \\square}\\ \\square",
};

// A 2x2 fillable matrix seeded into single-matrix ops when the editor is empty.
const MATRIX_SKELETON = "\\begin{bmatrix}{}&{}\\\\{}&{}\\end{bmatrix}";

// Natural variable / summation index per operation (avoids the series index
// colliding with the power-series argument, which broke convergence checks).
const VAR_DEFAULT = {
  series_convergence: "n", radius_of_convergence: "n", power_series: "n",
  parametric_diff: "t", polar_area: "theta", polar_slope: "theta",
  complex_roots: "z", complex_simplify: "z", complex_polar: "z", de_moivre: "z",
};

function isTextOp(op) {
  return MATH_TEXT_OPS.has(op);
}

// Canonical value store for a cell is the hidden <textarea.math-input>; the
// visible MathLive <math-field> mirrors it. Two-input ops keep "a\nb" in the
// canonical textarea and split it across two editors.
function getMathSource(cell) { return cell.querySelector("textarea.math-input"); }
function getMathField(cell) { return cell.querySelector("math-field.math-editor"); }
function getMathSourceB(cell) { return cell.querySelector("textarea.math-input-b"); }
function getMathFieldB(cell) { return cell.querySelector("math-field.math-editor-b"); }
function getOp(cell) { return cell.querySelector(".operation-type").value; }

// MathLive renders a keyboard-toggle and insert-menu button inside each field's
// shadow DOM; hide them so the only insertion path is our curated palette.
function hideMlToggles(field) {
  if (!field) return;
  const apply = () => {
    const sr = field.shadowRoot;
    if (!sr || sr.querySelector("#hl-hide-toggles")) return;
    const s = document.createElement("style");
    s.id = "hl-hide-toggles";
    s.textContent = ".ML__virtual-keyboard-toggle,.ML__menu-toggle{display:none!important}";
    sr.appendChild(s);
  };
  apply();
  requestAnimationFrame(apply);
}

// Read the live expression, committing the visible editor(s) into the canonical
// store first so every existing path (run/save/steps) stays correct.
function readExpr(cell) {
  const op = getOp(cell);
  const src = getMathSource(cell);
  if (!src) return "";
  if (MULTI_INPUT_OPS.has(op)) {
    const a = getMathField(cell);
    const b = getMathFieldB(cell);
    const va = (a && a.style.display !== "none") ? a.value : src.value.split("\n")[0];
    const vb = (b && b.style.display !== "none") ? b.value : (src.value.split("\n")[1] || "");
    src.value = va + "\n" + vb;
    return src.value.trim();
  }
  const field = getMathField(cell);
  if (field && field.style.display !== "none") {
    src.value = field.value;
  }
  return src.value.trim();
}

// Show the right input(s) for the operation: two editors (multi), one editor
// (scalar/single-matrix), or the plain textarea (text ops).
function applyInputMode(cell) {
  const op = getOp(cell);
  const field = getMathField(cell);
  const src = getMathSource(cell);
  const fieldB = getMathFieldB(cell);
  const srcB = getMathSourceB(cell);
  const labelB = cell.querySelector(".input-b-label");
  if (!field || !src) return;
  const multi = MULTI_INPUT_OPS.has(op);
  if (fieldB) fieldB.style.display = multi ? "" : "none";
  if (srcB) srcB.style.display = "none";
  if (labelB) labelB.style.display = multi ? "" : "none";

  if (isTextOp(op)) {
    field.style.display = "none";
    src.style.display = "";
  } else {
    const parts = (src.value || "").split("\n");
    field.value = parts[0] || "";
    if (multi && fieldB) fieldB.value = parts[1] || "";
    if (!src.value && MATRIX_SINGLE_OPS.has(op)) {
      field.value = MATRIX_SKELETON;
      src.value = MATRIX_SKELETON;
    }
    if (!src.value && multi && fieldB) {
      field.value = MATRIX_SKELETON;
      fieldB.value = MATRIX_SKELETON;
      src.value = MATRIX_SKELETON + "\n" + MATRIX_SKELETON;
    }
    field.style.display = "";
    src.style.display = "none";
    // Hide MathLive's built-in field toolbar (keyboard toggle + insert menu) so the
    // only way to insert is our curated, supported-token palette.
    field.toolbar = "hide";
    if (fieldB) fieldB.toolbar = "hide";
    hideMlToggles(field);
    if (fieldB) hideMlToggles(fieldB);
    field.setAttribute("placeholder", OP_PLACEHOLDER[op] || "\\square");
  }
  const varInput = cell.querySelector(".var-input input");
  if (varInput) varInput.value = VAR_DEFAULT[op] || "x";
}

let activeCellId = null;
let isDarkTheme = localStorage.getItem('darkTheme') === 'true';
let isKeyboardVisible = false;
let kernelStatus = 'idle';

let foreverVisible = true;

// Store polling timers
const pollTimers = new Map();

// Initialize after page loads
document.addEventListener("DOMContentLoaded", function () {
  applyTheme(isDarkTheme);
  addNewCell();
  bindEvents();
  animateMathBackground();
});

// Add control buttons to all math-output areas
function addMathOutputControls() {
  const mathOutputs = document.querySelectorAll('.math-output');

  mathOutputs.forEach(output => {
    // Check if control buttons have already been added
    if (!output.querySelector('.math-controls')) {
      // Create control button container
      const controls = document.createElement('div');
      controls.className = 'math-controls';

      // Create collapse button
      const collapseBtn = document.createElement('button');
      collapseBtn.className = 'math-control-btn';
      collapseBtn.title = '折叠/展开';
      collapseBtn.innerHTML = '<i class="fas fa-chevron-up"></i>';

      // Create fullscreen button
      const fullscreenBtn = document.createElement('button');
      fullscreenBtn.className = 'math-control-btn';
      fullscreenBtn.title = '全屏显示';
      fullscreenBtn.innerHTML = '<i class="fas fa-expand"></i>';

      // Add buttons to container
      controls.appendChild(collapseBtn);
      controls.appendChild(fullscreenBtn);

      // Add container to math-output
      output.appendChild(controls);

      // Add collapse functionality
      collapseBtn.addEventListener('click', function () {
        output.classList.toggle('collapsed');

        // Update icon
        if (output.classList.contains('collapsed')) {
          collapseBtn.innerHTML = '<i class="fas fa-chevron-down"></i>';
        } else {
          collapseBtn.innerHTML = '<i class="fas fa-chevron-up"></i>';
        }

        // Re-render MathJax
        if (window.MathJax) {
          MathJax.typesetPromise([output]);
        }
      });

      // Add fullscreen functionality
      fullscreenBtn.addEventListener('click', function () {
        openFullscreen(output);
      });
    }
  });
}

// Open fullscreen display
function openFullscreen(mathOutput) {
  const overlay = document.querySelector('.math-fullscreen-overlay');
  const content = overlay.querySelector('.math-fullscreen-body');

  // Copy content to fullscreen display area, but keep original content
  const originalContent = mathOutput.cloneNode(true);

  // Remove control buttons
  const controls = originalContent.querySelector('.math-controls');
  if (controls) {
    controls.remove();
  }

  content.innerHTML = originalContent.innerHTML;

  // Show fullscreen overlay
  overlay.classList.add('active');

  // Re-render MathJax
  if (window.MathJax) {
    MathJax.typesetPromise([content]);
  }
}

// Math background animation function
function animateMathBackground() {
  const elements = document.querySelectorAll('.math-bg-element');
  elements.forEach(el => {
    const delay = Math.random() * 2;
    const duration = 15 + Math.random() * 10;
    el.style.animation = `float ${duration}s ease-in-out ${delay}s infinite`;
  });
}

// Toggle visibility
function toggleCodeCellInputs() {
  foreverVisible = !foreverVisible;
  setVisibility(foreverVisible);
}

// Set visibility for all Cells
function setVisibility(visible) {
  document.querySelectorAll(".code-cell").forEach(cell => {
    setSingleVisibility(cell, visible);
  });
}

// Set visibility for a single Cell
function setSingleVisibility(cell, visible) {
  const mathControls = cell.querySelector(".math-controls");
  const mjxContainer = cell.querySelector(".cell-output mjx-container")
  if (visible) {
    cell.querySelector(".cell-output").style.background = "var(--jp-cell-output-background)";
    cell.querySelector(".cell-output").style.border = "1px dashed var(--jp-border-color1)";
    cell.querySelector(".cell-content").style.display = "block";
    cell.querySelector(".add-cell-button").style.display = "flex";
    if (mathControls) mathControls.style.display = "flex";
    if (mjxContainer) mjxContainer.style.textAlign = "center";
  } else {
    cell.querySelector(".cell-output").style.background = "transparent";
    cell.querySelector(".cell-output").style.border = "none";
    cell.querySelector(".cell-content").style.display = "none";
    cell.querySelector(".add-cell-button").style.display = "none";
    if (mathControls) mathControls.style.display = "none";
    if (mjxContainer) mjxContainer.style.textAlign = "left";
  }
}

// Bind events
function bindEvents() {
  document.getElementById("addCellBtn").addEventListener("click", () => addNewCell());
  document.getElementById("addMdCellBtn").addEventListener("click", () => addNewCell("markdown"));
  document.getElementById("toggleCodeCellInputs").addEventListener("click", toggleCodeCellInputs);
  document.getElementById("runAllBtn").addEventListener("click", runAllCells);
  document.getElementById("moveCellUpBtn").addEventListener("click", moveActiveCellUp);
  document.getElementById("moveCellDownBtn").addEventListener("click", moveActiveCellDown);
  document.getElementById("deleteCellBtn").addEventListener("click", deleteActiveCell);
  document.getElementById("toggleKeyboardBtn").addEventListener("click", toggleKeyboard);
  document.getElementById("closeKeyboardBtn").addEventListener("click", toggleKeyboard);
    const _themeBtn = document.getElementById("toggleThemeBtn");
  if (_themeBtn) _themeBtn.addEventListener("click", toggleTheme);
  document.getElementById("saveNotebookBtn").addEventListener("click", saveNotebook);
  document.getElementById("loadNotebookBtn").addEventListener("click", loadNotebook);

  // Cancel active cell when clicking on other areas of the document
  document.addEventListener("click", function (e) {
    if (!e.target.closest(".cell") && !e.target.closest("#math-keyboard") && !e.target.closest("#hd-top-menu") && !e.target.closest("#toolbar")) {
      document.querySelectorAll(".cell").forEach((cell) => {
        cell.classList.remove("cell-active");
        if (cell.dataset.cellType === "code" && !foreverVisible) {
          setSingleVisibility(cell, false);
        } else if (cell.dataset.cellType === "markdown") {
          const input = cell.querySelector(".markdown-input");
          const render = cell.querySelector(".markdown-render");
          renderMarkdown(cell);
          if (input && render) {
            input.style.display = "none";
            render.style.display = "block";
          }
        }
      });
    }
  });

  // Keyboard shortcuts
  document.addEventListener("keydown", function (e) {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
      const activeCell = document.querySelector(".cell-active");
      if (activeCell) runCell(activeCell);
      e.preventDefault();
    }

    if (e.key === "a" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      const activeCell = document.querySelector(".cell-active");
      if (activeCell) addCellAbove(activeCell);
    }

    if (e.key === "b" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      const activeCell = document.querySelector(".cell-active");
      if (activeCell) addCellBelow(activeCell);
    }

    if (e.key === "ArrowUp" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      const activeCell = document.querySelector(".cell-active");
      if (activeCell) moveActiveCellUp();
    }

    if (e.key === "ArrowDown" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      const activeCell = document.querySelector(".cell-active");
      if (activeCell) moveActiveCellDown();
    }

    if (e.key === "d" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      const activeCell = document.querySelector(".cell-active");
      if (activeCell) deleteActiveCell();
    }

    if (e.key === "k" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      toggleKeyboard();
    }
  });

  // Bind virtual keyboard / template palette buttons
  document.querySelectorAll(".keyboard-key").forEach((key) => {
    key.addEventListener("click", function () {
      const latex = this.getAttribute("data-latex");
      const activeCell = document.querySelector(".cell-active");
      if (!activeCell) return;
      const field = getMathField(activeCell);
      const textarea = getMathSource(activeCell);
      if (field && field.style.display !== "none") {
        // Insert a LaTeX template with editable \square placeholders into the editor.
        field.focus();
        if (typeof field.insert === "function") {
          field.insert(latex);
        } else {
          field.value = (field.value || "") + latex;
        }
        if (textarea) textarea.value = field.value;
      } else if (textarea) {
        const cursorPos = textarea.selectionStart || textarea.value.length;
        textarea.value = textarea.value.substring(0, cursorPos) + latex + textarea.value.substring(cursorPos);
        textarea.focus();
      }
    });
  });

  // Fullscreen related events
  const closeBtn = document.querySelector('.math-fullscreen-close');
  if (closeBtn) {
    closeBtn.addEventListener('click', function () {
      const overlay = document.querySelector('.math-fullscreen-overlay');
      overlay.classList.remove('active');
    });
  }

  // Click overlay background to close fullscreen
  const overlay = document.querySelector('.math-fullscreen-overlay');
  if (overlay) {
    overlay.addEventListener('click', function (e) {
      if (e.target === overlay) {
        overlay.classList.remove('active');
      }
    });
  }

  // ESC key to close fullscreen
  document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape') {
      const overlay = document.querySelector('.math-fullscreen-overlay');
      if (overlay && overlay.classList.contains('active')) {
        overlay.classList.remove('active');
      }
    }
  });
}

// Set cell events
function setupCellEvents(cell) {
  const operationSelector = cell.querySelector(".operation-type");
  const mathField = cell.querySelector("math-field");
  const runButton = cell.querySelector(".run-cell-button");
  const keyboardButton = cell.querySelector(".keyboard-button");
  const addCellButton = cell.querySelector(".add-cell-button");

  const cellType = cell.dataset.cellType;

  if (cellType === "code") {
    // The MathLive editor(s) mirror their LaTeX into the hidden textarea, which
    // stays the canonical value every other path reads (readExpr commits them).
    const mathFieldB = cell.querySelector("math-field.math-editor-b");
    if (mathField) {
      mathField.addEventListener("change", function () { readExpr(cell); });
      mathField.addEventListener("click", function () { setActiveCell(cell); });
    }
    if (mathFieldB) {
      mathFieldB.addEventListener("change", function () { readExpr(cell); });
      mathFieldB.addEventListener("click", function () { setActiveCell(cell); });
    }
    applyInputMode(cell);

    // Operation type change event
    operationSelector.addEventListener("change", function () {
      // Switch editor/textarea visibility and set the per-op template hint.
      applyInputMode(cell);


      // If diff, integral, limit is selected, show variable input box
      /*
      const varInput = cell.querySelector(".var-input");
      if (this.value === "diff" || this.value === "integral" || this.value === "limit") {
        if (varInput) varInput.style.display = "flex";
      } else {
        if (varInput) varInput.style.display = "none";
      }
      */

      // If ref is selected, show target form input box for matrix
      const refControls = cell.querySelector(".ref-controls");
      if (this.value === "ref") {
        if (refControls) refControls.style.display = "flex";
      } else {
        if (refControls) refControls.style.display = "none";
      }

      const operationsControls = cell.querySelector(".operations-controls");
      if (this.value === "operations") {
        if (operationsControls) operationsControls.style.display = "flex";
      } else {
        if (operationsControls) operationsControls.style.display = "none";
      }

      const transform1Controls = cell.querySelector(".transform-1-controls");
      if (this.value === "transform-1") {
        if (transform1Controls) transform1Controls.style.display = "flex";
      } else {
        if (transform1Controls) transform1Controls.style.display = "none";
      }

      const transform2Controls = cell.querySelector(".transform-2-controls");
      if (this.value === "transform-2") {
        if (transform2Controls) transform2Controls.style.display = "flex";
      } else {
        if (transform2Controls) transform2Controls.style.display = "none";
      }

      const detControls = cell.querySelector(".det-controls");
      if (this.value === "det") {
        if (detControls) detControls.style.display = "flex";
      } else {
        if (detControls) detControls.style.display = "none";
      }

      // If limit is selected, show additional options
      const limitControls = cell.querySelector(".limit-controls");
      if (this.value === "limit") {
        if (limitControls) limitControls.style.display = "flex";
      } else {
        if (limitControls) limitControls.style.display = "none";
      }

      // If limit is selected, show additional options
      const integralControls = cell.querySelector(".integral-controls");
      if (this.value === "integral") {
        if (integralControls) integralControls.style.display = "flex";
      } else {
        if (integralControls) integralControls.style.display = "none";
      }

      // 高等数学(应用/级数/方程)类: 显示统一参数输入
      const appliedControls = cell.querySelector(".applied-controls");
      const isApplied = APPLIED_OPS.indexOf(this.value) !== -1;
      if (appliedControls) appliedControls.style.display = isApplied ? "flex" : "none";

      // 累次/二重积分: 额外第二变量与其上下限
      const iteratedControls = cell.querySelector(".iterated-controls");
      if (iteratedControls) iteratedControls.style.display = (this.value === "iterated_integral") ? "flex" : "none";

      // If expr is selected, show additional options
      const expressionControls = cell.querySelector(".expression-controls");
      if (this.value === "expr") {
        if (expressionControls) expressionControls.style.display = "flex";
      } else {
        if (expressionControls) expressionControls.style.display = "none";
      }
    });

    // Run cell
    runButton.addEventListener("click", function () {
      runCell(cell);
    });

    // Virtual keyboard button
    keyboardButton.addEventListener("click", function () {
      toggleKeyboard();
    });
  }

  const cellPrompt = cell.querySelector(".cell-prompt");
  // Click event - set active cell
  cellPrompt.addEventListener("click", function () {
    setActiveCell(cell);
  });

  // Add cell button
  addCellButton.addEventListener("click", function () {
    addCellBelow(cell);
  });

  // Textbox click event - set active cell
  const sourceArea = getMathSource(cell);
  if (sourceArea) {
    sourceArea.addEventListener("click", function () {
      setActiveCell(cell);
    });
  }
}

// Debounce function
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

// Set active cell
function setActiveCell(cell) {
  document.querySelectorAll(".cell").forEach((c) => {
    c.classList.remove("cell-active");
    if (c.dataset.cellType === "code" && !foreverVisible) {
      setSingleVisibility(c, false);
    } else if (c.dataset.cellType === "markdown") {
      // For markdown cells, hide input box and show rendered content when inactive
      const input = c.querySelector(".markdown-input");
      const render = c.querySelector(".markdown-render");
      renderMarkdown(c);
      if (input && render) {
        input.style.display = "none";
        render.style.display = "block";
      }
    }
  });
  cell.classList.add("cell-active");
  activeCellId = cell.dataset.cellId;
  if (cell.dataset.cellType === "code" && !foreverVisible) {
    setSingleVisibility(cell, true);
  } else if (cell.dataset.cellType === "markdown") {
    // For markdown cells, show input box and hide rendered content when active
    const input = cell.querySelector(".markdown-input");
    const rendered = cell.querySelector(".markdown-render");
    if (input && rendered) {
      input.style.display = "block";
      rendered.style.display = "none";

      // Focus on input box
      input.focus();
      cell.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  } else {
    const field = cell.querySelector("math-field");
    const textarea = cell.querySelector("textarea");
    const target = (field && field.style.display !== "none") ? field : textarea;
    if (target) {
      target.focus();
      cell.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  }
}

// Add new cell
function addNewCell(type = "code") {
  const notebook = document.getElementById("notebook");
  const newCell = createCellElement(cellCounter, type);
  notebook.appendChild(newCell);
  setupCellEvents(newCell);
  setActiveCell(newCell);
  cellCounter++;

  newCell.style.opacity = "0";
  newCell.style.transform = "translateY(20px)";

  setTimeout(() => {
    newCell.style.transition = "opacity 0.3s ease, transform 0.3s ease";
    newCell.style.opacity = "1";
    newCell.style.transform = "translateY(0)";
  }, 10);
}

// Create cell element
function createCellElement(id, type) {
  const cell = document.createElement("div");
  cell.className = `cell ${type}-cell`;
  cell.dataset.cellId = id;
  cell.dataset.cellType = type;
  const radioButtonGroupName = `include-option-${id}`;

  if (type === "code") {
    cell.innerHTML = `
      <div class="cell-prompt">Calculator:</div>
      <div class="cell-content-wrapper">
        <div class="cell-content">
          <div class="cell-input">
            <div class="cell-controls">
              <button class="run-cell-button" title="运行单元格">
                <i class="fas fa-play"></i>
              </button>
              <select class="operation-type">
                <optgroup label="万能输入">
                  <option value="auto" selected>自动识别(微积分+线性代数)</option>
                </optgroup>
                <optgroup label="基本处理">
                <option value="expr">等价表达式</option>
                </optgroup>
                <optgroup label="微积分">
                  <option value="calculus">微积分(自动识别算子)</option>
                  <option value="diff">求导/微分</option>
                  <option value="integral">积分</option>
                  <option value="limit">极限</option>
                </optgroup>
                <optgroup label="线性代数">
                  <option value="operations">基本运算</option>
                  <option value="matrix_symbolic">符号矩阵化简</option>
                  <option value="invert">求逆</option>
                  <option value="ref">行阶梯/行最简</option>
                  <option value="transform-1">基/坐标变换</option>
                  <option value="transform-2">线性变换变基</option>
                  <option value="det">行列式</option>
                  <option value="dependence">线性相关/无关</option>
                  <option value="projection">向量投影</option>
                  <option value="rank">秩</option>
                  <option value="LU">LU 分解</option>
                  <option value="orthogonal">QR 分解</option>
                  <option value="diag">对角化分解</option>
                  <option value="eigen">特征值/向量</option>
                  <option value="singular">奇异值</option>
                  <option value="svd">SVD 分解</option>
                  <option value="schur">Schur 分解</option>
                  <option value="linear-system">方程组矩阵转换</option>
                  <option value="linear-solver">方程组求解</option>
                </optgroup>
                <optgroup label="高等数学(应用/级数/方程)">
                  <option value="tangent">切线方程</option>
                  <option value="linearization">线性化</option>
                  <option value="critical_points">临界点/极值</option>
                  <option value="newton">牛顿法</option>
                  <option value="arc_length">弧长</option>
                  <option value="volume_disk">旋转体体积(圆盘)</option>
                  <option value="surface_area">旋转曲面面积</option>
                  <option value="volume_shell">旋转体体积(柱壳法)</option>
                  <option value="iterated_integral">累次/二重积分(矩形区域)</option>
                  <option value="implicit_diff">隐函数求导</option>
                  <option value="parametric_diff">参数方程求导</option>
                  <option value="polar_area">极坐标面积</option>
                  <option value="polar_slope">极坐标斜率</option>
                  <option value="taylor">泰勒/麦克劳林展开</option>
                  <option value="series_convergence">级数敛散性</option>
                  <option value="improper_integral">反常积分</option>
                  <option value="ode">微分方程求解</option>
                  <option value="numerical_integration">数值积分(梯形/辛普森)</option>
                  <option value="mean_value">中值定理/罗尔定理</option>
                  <option value="radius_of_convergence">幂级数收敛半径</option>
                  <option value="power_series">幂级数收敛域+逐项求导积分</option>
                  <option value="taylor_error">泰勒余项误差估计(拉格朗日)</option>
                  <option value="complex_simplify">复数化简(a+bi)</option>
                  <option value="complex_polar">复数极坐标/指数形式</option>
                  <option value="complex_roots">复数方程求根</option>
                  <option value="de_moivre">复数高次幂(棣莫弗)</option>
                </optgroup>
              </select>
              <!-- 变量输入框 -->
              <!--
              <span class="var-input" style="display: flex;">
                <span class="var-label">变量:</span>
                <input type="text" value="x" size="2" class="var-input-field">
              </span>
              -->

              <!-- RefCalculator 专有 -->
              <div class="ref-controls controls" style="display: none; gap:8px; margin-left:8px; align-items:center;">
                <span class="mtef-label label">目标形式:</label>
                <select class="ref-sort sort">
                  <option value="ref">行阶梯</option>
                  <option value="rref">行最简</option>
                </select>
              </div>

              <!-- BasicOperations 专有 -->
              <div class="operations-controls controls" style="display: none; gap:8px; margin-left:8px; align-items:center;">
                <span class="operations-label label">符号:</label>
                <select class="operations-sort sort">
                  <option value="+">加</option>
                  <option value="-">减</option>
                  <option value="T">转置</option>
                  <option value="*">矩阵相乘</option>
                  <option value="scalar_mul">标量乘矩阵</option>
                  <option value="dot">点积</option>
                  <option value="cross">叉积</option>
                </select>
              </div>

              <!-- BaseTransform 专有 -->
              <div class="transform-1-controls controls" style="display: none; gap:8px; margin-left:8px; align-items:center;">
                <span class="transform-1-label label">类型:</label>
                <select class="transform-1-sort sort">
                  <option value="basis_change">一般基变换</option>
                  <option value="coord_transform">坐标一般基变换</option>
                  <option value="std_to_basis">坐标标准基到给定基变换</option>
                  <option value="basis_to_std">坐标给定基到标准基变换</option>
                </select>
              </div>

              <!-- LinearTransform 专有 -->
              <div class="transform-2-controls controls" style="display: none; gap:8px; margin-left:8px; align-items:center;">
                <span class="transform-2-label label">类型:</label>
                <select class="transform-2-sort sort">
                  <option value="find_matrix">标准基到给定基</option>
                  <option value="basis_change">一般基变换</option>
                </select>
              </div>

              <!-- DetCal 专有 -->
              <div class="det-controls controls" style="display: none; gap:8px; margin-left:8px; align-items:center;">
                <span class="det-label label">计算器类型选择:</label>
                <select class="det-sort sort">
                  <option value="1">1</option>
                  <option value="2">2</option>
                </select>
              </div>

              <!-- LimitCalculator 专有 -->
              <div class="limit-controls" style="display: none; gap:8px; margin-left:8px; align-items:center;">
                <span class="limit-label">方向:</label>
                <select class="limit-sort">
                  <option value="+">右极限</option>
                  <option value="-">左极限</option>
                  <option value="both">双极限/极限</option>
                </select>
                <label style="font-size:12px;">趋于:</label>
                <input class="point" value="0" style="width:50px;">
                <label style="font-size:12px;">最多使用洛必达的次数</label>
                <input class="max-lhopital-count" value="5" style="width:50px;">
              </div>

              <!-- IntegralCalculator 专有 -->
              <div class="integral-controls" style="display: none; gap:8px; margin-left:8px; align-items:center;">
               <label class="include-toggle">
                <input type="radio" name="${radioButtonGroupName}" value="include" checked>
                <span>包含特殊匹配器</span>
               </label>
               <label class="include-toggle">
                <input type="radio" name="${radioButtonGroupName}" value="none">
                <span>不包含特殊匹配器</span>
               </label>
               <label style="font-size:12px; margin-left:8px;">上限:</label>
               <input class="integral-upper" placeholder="空=不定积分" style="width:80px;">
               <label style="font-size:12px;">下限:</label>
               <input class="integral-lower" placeholder="默认0" style="width:70px;">
              </div>

              <!-- ExpressionParser 专有 -->
              <div class="expression-controls" style="display: none; gap:8px; margin-left:8px; align-items:center;">
                <span class="expr-label">排序:</label>
                <select class="expr-sort">
                  <option value="complexity">复杂度</option>
                  <option value="length">长度</option>
                  <option value="none">不排序</option>
                </select>
                <label style="font-size:12px;">深度:</label>
                <input type="number" class="expr-depth" value="3" min="1" max="10" style="width:50px;">
                <!--
                <span class="gene-label">生成推导树:</label>
                <select class="gene-sort">
                  <option value="false">否</option>
                  <option value="true">是</option>
                </select>
                -->
              </div>

              <!-- 高等数学(应用/级数/方程) 专有 -->
              <div class="applied-controls" style="display: none; gap:8px; margin-left:8px; align-items:center; flex-wrap:wrap;">
                <label style="font-size:12px;">点/初值:</label>
                <input class="applied-point" value="0" style="width:60px;">
                <label style="font-size:12px;">下限:</label>
                <input class="applied-lower" value="0" style="width:60px;">
                <label style="font-size:12px;">上限:</label>
                <input class="applied-upper" value="1" style="width:60px;">
                <label style="font-size:12px;">阶数/次数:</label>
                <input class="applied-order" value="6" style="width:50px;">
              </div>

              <!-- 累次/二重积分: 第二个变量与其上下限 (第一变量的上下限复用"下限/上限") -->
              <div class="iterated-controls" style="display: none; gap:8px; margin-left:8px; align-items:center; flex-wrap:wrap;">
                <label style="font-size:12px;">变量2:</label>
                <input class="iterated-var2" value="y" style="width:50px;">
                <label style="font-size:12px;">下限2:</label>
                <input class="iterated-lower2" value="0" style="width:60px;">
                <label style="font-size:12px;">上限2:</label>
                <input class="iterated-upper2" value="1" style="width:60px;">
              </div>

              <span class="var-input" style="display:inline-flex; align-items:center; gap:4px; margin-left:8px;">
                <label style="font-size:12px;">变量/指标:</label>
                <input value="x" style="width:50px;" title="求导/积分/极限变量, 级数指标等">
              </span>

              <label class="cache-toggle" title="使用缓存加速计算">                <input type="checkbox" class="use-cache-checkbox" checked>
                <span>缓存</span>
              </label>
              <label class="cache-toggle" title="将计算结果存入缓存">
                <input type="checkbox" class="record-cache-checkbox" checked>
                <span>记录</span>
              </label>

              <button class="keyboard-button" title="打开虚拟键盘">
                <i class="fas fa-keyboard"></i>
              </button>
            </div>
            <math-field class="math-editor" virtual-keyboard-mode="manual" toolbar="hide" style="display:none;"></math-field>
            <textarea placeholder="输入数学表达式..." class="math-input"></textarea>
            <span class="input-b-label" style="display:none; font-size:12px; color:#888; margin-top:4px;">第二个矩阵/向量:</span>
            <math-field class="math-editor math-editor-b" virtual-keyboard-mode="manual" toolbar="hide" style="display:none;"></math-field>
            <textarea placeholder="第二个输入..." class="math-input math-input-b" style="display:none;"></textarea>
          </div>
        </div>
        <div class="cell-output-container">
          <div class="cell-output"></div>
        </div>
      </div>
      <div class="add-cell-button" title="在下方添加单元格">
        <i class="fas fa-plus"></i>
      </div>
    `;
  } else if (type === "markdown") {
    cell.innerHTML = `
      <div class="cell-prompt">Markdown:</div>
      <div class="cell-content-wrapper">
        <div class="cell-content">
          <div class="cell-input">
            <textarea placeholder="输入 Markdown 文本..." class="markdown-input" style="display: none;"></textarea>
            <div class="markdown-render"></div>
          </div>
        </div>
      </div>
      <div class="add-cell-button" title="在下方添加单元格">
        <i class="fas fa-plus"></i>
      </div>
    `;
  }

  return cell;
}

// Add cell above
function addCellAbove(cell) {
  const notebook = document.getElementById("notebook");
  const newCell = createCellElement(cellCounter, cell.dataset.cellType);
  notebook.insertBefore(newCell, cell);
  setupCellEvents(newCell);
  setActiveCell(newCell);
  cellCounter++;

  newCell.style.opacity = "0";
  newCell.style.transform = "translateY(-20px)";

  setTimeout(() => {
    newCell.style.transition = "opacity 0.3s ease, transform 0.3s ease";
    newCell.style.opacity = "1";
    newCell.style.transform = "translateY(0)";
  }, 10);
}

// Add cell below
function addCellBelow(cell) {
  const notebook = document.getElementById("notebook");
  const newCell = createCellElement(cellCounter, cell.dataset.cellType);

  if (cell.nextSibling) {
    notebook.insertBefore(newCell, cell.nextSibling);
  } else {
    notebook.appendChild(newCell);
  }

  setupCellEvents(newCell);
  setActiveCell(newCell);
  cellCounter++;

  newCell.style.opacity = "0";
  newCell.style.transform = "translateY(20px)";

  setTimeout(() => {
    newCell.style.transition = "opacity 0.3s ease, transform 0.3s ease";
    newCell.style.opacity = "1";
    newCell.style.transform = "translateY(0)";
  }, 10);
}

// Run all cells
function runAllCells() {
  updateKernelStatus('busy');

  document.querySelectorAll('.cell[data-cell-type="code"]').forEach((cell) => {
    const expression = readExpr(cell);
    if (expression) runCell(cell);
  });

  setTimeout(() => {
    updateKernelStatus('idle');
  }, 500);
}

// Update kernel status
function updateKernelStatus(status) {
  kernelStatus = status;
  const statusElement = document.getElementById("kernel-status");
  const icon = statusElement.querySelector("i");

  switch (status) {
    case 'idle':
      icon.style.color = "var(--jp-kernel-idle-color)";
      statusElement.querySelector("span").textContent = "内核就绪";
      break;
    case 'busy':
      icon.style.color = "var(--jp-kernel-busy-color)";
      statusElement.querySelector("span").textContent = "计算中...";
      break;
    case 'error':
      icon.style.color = "var(--jp-kernel-error-color)";
      statusElement.querySelector("span").textContent = "内核错误";
      break;
  }
}

// Run single cell
function runCell(cell) {
  const operationType = cell.querySelector(".operation-type").value;
  const expression = readExpr(cell);
  const variable = cell.querySelector(".var-input input")
    ? cell.querySelector(".var-input input").value.trim() || "x"
    : "x";

  if (!expression) {
    showNotification("请输入表达式", "error");
    return;
  }

  let outputArea = cell.querySelector(".cell-output");
  if (!outputArea) {
    const outputDiv = document.createElement("div");
    outputDiv.className = "cell-output";
    cell.querySelector(".cell-output-container").appendChild(outputDiv);
    outputArea = outputDiv;
  }

  outputArea.innerHTML = '<div class="loading">计算中...</div>';
  updateKernelStatus('busy');

  const runButton = cell.querySelector(".run-cell-button");
  runButton.classList.add("running");

  // Pass data to backend
  const payload = {
    operation_type: operationType,
    expression: expression,
    variable: variable,
  };

  if (operationType === "limit") {
    const point = cell.querySelector(".point").value || 0;
    const max_lhopital_count = cell.querySelector(".max-lhopital-count").value || 5;
    const direction = cell.querySelector(".limit-sort").value || '+';
    payload.point = point;
    payload.direction = direction;
    payload.max_lhopital_count = max_lhopital_count;
  }

  if (operationType === "integral") {
    const radioButtonGroupName = `include-option-${cell.dataset.cellId}`;
    const selectedRadio = cell.querySelector(`input[name="${radioButtonGroupName}"]:checked`).value === "include";
    payload.include_special_matchers = selectedRadio ? 'True' : 'False';

    // 定积分: 仅当填写上限时按定积分处理; 下限留空默认 0
    const upperInput = cell.querySelector(".integral-upper");
    const lowerInput = cell.querySelector(".integral-lower");
    const upper = upperInput ? upperInput.value.trim() : "";
    const lower = lowerInput ? lowerInput.value.trim() : "";
    if (upper !== "") {
      payload.upper = upper;
      payload.lower = lower;
    }
  }

  if (APPLIED_OPS.indexOf(operationType) !== -1) {
    const ap = cell.querySelector(".applied-point");
    const al = cell.querySelector(".applied-lower");
    const au = cell.querySelector(".applied-upper");
    const ao = cell.querySelector(".applied-order");
    if (ap) payload.point = ap.value.trim();
    if (al) payload.lower = al.value.trim();
    if (au) payload.upper = au.value.trim();
    if (ao) payload.order = ao.value.trim();
  }

  if (operationType === "iterated_integral") {
    const v2 = cell.querySelector(".iterated-var2");
    const l2 = cell.querySelector(".iterated-lower2");
    const u2 = cell.querySelector(".iterated-upper2");
    if (v2) payload.variable2 = v2.value.trim() || "y";
    if (l2) payload.lower2 = l2.value.trim();
    if (u2) payload.upper2 = u2.value.trim();
  }

  if (operationType === "ref") {
    const target_form = cell.querySelector(".ref-sort").value;
    payload.target_form = target_form;
  }

  if (operationType === "operations") {
    const operations = cell.querySelector(".operations-sort").value;
    payload.operations = operations;
  }

  if (operationType === "transform-1") {
    const transform_type = cell.querySelector(".transform-1-sort").value;
    payload.type = transform_type;
  }

  if (operationType === "transform-2") {
    const transform_type = cell.querySelector(".transform-2-sort").value;
    payload.type = transform_type;
  }

  if (operationType === "det") {
    const det_type = cell.querySelector(".det-sort").value;
    payload.type = det_type;
  }

  // Cache control
  const useCacheCheckbox = cell.querySelector(".use-cache-checkbox");
  const recordCacheCheckbox = cell.querySelector(".record-cache-checkbox");
  payload.use_cache = useCacheCheckbox ? (useCacheCheckbox.checked ? 'True' : 'False') : 'False';
  payload.record_cache = recordCacheCheckbox ? (recordCacheCheckbox.checked ? 'True' : 'False') : 'False';

  // If solving for equivalent expressions, add sorting and depth parameters
  if (operationType === 'expr') {
    const sort = cell.querySelector(".expr-sort").value || "complexity";
    const depth = cell.querySelector(".expr-depth").value || "3";
    // const is_draw_tree = cell.querySelector(".gene-sort").value;
    payload.max_depth = depth;
    payload.sort_strategy = sort;
    // payload.is_draw_tree = is_draw_tree;
  }

  // Display computing status
  outputArea.innerHTML = `<div class="computing">
    <div class="loading-spinner"></div>
    <div class="loading-text">正在计算中，请稍候...</div>
  </div>`;

  fetch("/api/compute", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  })
    .then((response) => response.json())
    .then((data) => {
      if (data.success && data.task_id) {
        // Start polling task status
        subscribeTask(data.task_id, outputArea, runButton, operationType);
      } else {
        runButton.classList.remove("running");
        updateKernelStatus('idle');
        outputArea.innerHTML = `<div class="error"> 错误: ${data.error || '未知错误'}</> `;
        updateKernelStatus('error');
      }
    })
    .catch((error) => {
      runButton.classList.remove("running");
      outputArea.innerHTML = `<div class="error"> 请求错误: ${error.message}</> `;
      updateKernelStatus('error');
    });
}

// Render the equivalent-expression list returned by ExpressionParser.
function renderExpressionResult(outputArea, data) {
  // data.result: [{latex, reason}, ...]
  const expressions = data.result || [];

  let html = `<div class="math-output">`;
  if (expressions.length === 0) {
    html += `<div class="error"> 未找到等价表达式</> `;
  } else {
    html += `<div class="expr-list"> <h3>等价表达式</h3>`;
    expressions.forEach((it, idx) => {
      html += `<div class="expr-item"> <div class="expr-rank">${idx + 1}.&nbsp;&nbsp;${it.reason}<div><div class="expr-content">$$${it.latex}$$</div></div>`;
    });
    html += `</div>`;
  }

  // if (treeUrl) {
  //   html += `<div class="expr-tree"><h3>推导树</h3>
  //           <img src="${treeUrl}" alt="expression tree"
  //               style="max-width:100%; border:1px solid var(--jp-border-color0); border-radius:8px;"
  //               onerror="this.onerror=null; this.parentElement.innerHTML='&lt;h3&gt;推导树&lt;/h3&gt;&lt;div class=&quot;note&quot;&gt;图片加载失败，请自行到 static/trees 文件夹下查看&lt;/div&gt;'"/>
  //         </div>`;
  // } else {
  //   html += `<div class="expr-tree"><h3>推导树</h3><div class="note">没有可用的推导树(后端未生成).</div></div>`;
  // }

  html += `</> `;
  outputArea.innerHTML = html;

  // Add control buttons
  addMathOutputControls();

  if (typeof MathJax !== "undefined") {
    MathJax.typesetPromise([outputArea]);
  }
}

// Render Markdown content
function renderMarkdown(cell) {
  const markdownInput = cell.querySelector(".markdown-input");
  const markdownRendered = cell.querySelector(".markdown-render");

  if (!markdownInput || !markdownRendered) return;

  const content = markdownInput.value;

  // Simple Markdown rendering implementation
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
    // Italic
    .replace(/_(.*?)_/gim, '<em>$1</em>')
    // Code
    .replace(/`(.*?)`/gim, '<code>$1</code>')
    // Line breaks
    .replace(/\n/gim, '<br>')
    // Links
    .replace(/\[([^\]]+)\]\(([^)]+)\)/gim, '<a href="$2" target="_blank">$1</a>')
    // Unordered lists
    .replace(/^\s*[-*] (.*$)/gim, '<li>$1</li>')
    // Handle list wrapping
    .replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>')
    // Quotes
    .replace(/^> (.*$)/gim, '<blockquote>$1</blockquote>');

  // Handle multiple consecutive line breaks (paragraphs)
  html = html.replace(/(<br>\s*){2,}/gim, '</p><p>');
  html = '<p>' + html + '</p>';

  markdownRendered.innerHTML = html;
}

// Show notification
function showNotification(message, type = "success") {
  const notification = document.createElement("div");
  notification.className = `notification notification-${type} `;
  notification.innerHTML = `
      <div class="notification-content">
      <i class="fas ${type === 'success' ? 'fa-check-circle' : 'fa-exclamation-circle'}"></i>
      <span>${message}</span>
    </>
      `;

  document.body.appendChild(notification);

  setTimeout(() => {
    notification.classList.add("notification-show");
  }, 10);

  setTimeout(() => {
    notification.classList.remove("notification-show");
    notification.classList.add("notification-hide");
    setTimeout(() => {
      if (document.body.contains(notification)) {
        document.body.removeChild(notification);
      }
    }, 300);
  }, 3000);
}

function toggleKeyboard() {
  const keyboard = document.getElementById("math-keyboard");
  const helpfulArea = document.getElementById("helpful-area");

  if (isKeyboardVisible) {
    // Hide keyboard
    keyboard.classList.remove("keyboard-visible");
    isKeyboardVisible = false;
    helpfulArea.style.display = "none";
  } else {
    // Show keyboard
    keyboard.classList.add("keyboard-visible");
    isKeyboardVisible = true;
    // Used to occupy the bottom of the page, when the keyboard is open, extend the page downward to prevent the keyboard from blocking the cell
    helpfulArea.style.display = "flex";
  }
}

// Apply theme settings
function applyTheme(isDark) {
  // The whole product is themed from one place: /static/hd-theme.css reads
  // the body.theme-dark class and re-points every design token.
  document.body.classList.toggle("theme-dark", !!isDark);
  const themeBtn = document.getElementById("toggleThemeBtn");
  if (themeBtn) {
    themeBtn.innerHTML = isDark ? '<i class="fas fa-sun"></i>' : '<i class="fas fa-moon"></i>';
  }
  isDarkTheme = !!isDark;
  localStorage.setItem("darkTheme", isDarkTheme);
}

// Toggle theme
function toggleTheme() {
  applyTheme(!isDarkTheme);
}

// Move active cell up
function moveActiveCellUp() {
  const activeCell = document.querySelector(".cell-active");
  if (activeCell && activeCell.previousElementSibling) {
    activeCell.style.transform = "translateY(-10px)";
    activeCell.style.opacity = "0.5";

    setTimeout(() => {
      activeCell.parentNode.insertBefore(activeCell, activeCell.previousElementSibling);

      setTimeout(() => {
        activeCell.style.transform = "translateY(0)";
        activeCell.style.opacity = "1";
      }, 10);
    }, 150);
  }
}

// Move active cell down
function moveActiveCellDown() {
  const activeCell = document.querySelector(".cell-active");
  if (activeCell && activeCell.nextElementSibling) {
    activeCell.style.transform = "translateY(10px)";
    activeCell.style.opacity = "0.5";

    setTimeout(() => {
      activeCell.parentNode.insertBefore(activeCell.nextElementSibling, activeCell);

      setTimeout(() => {
        activeCell.style.transform = "translateY(0)";
        activeCell.style.opacity = "1";
      }, 10);
    }, 150);
  }
}

// Delete active cell
function deleteActiveCell() {
  const activeCell = document.querySelector(".cell-active");
  if (activeCell && document.querySelectorAll(".cell").length > 1) {
    activeCell.style.transform = "scale(0.8)";
    activeCell.style.opacity = "0";

    setTimeout(() => {
      const previousCell = activeCell.previousElementSibling;
      const nextCell = activeCell.nextElementSibling;
      activeCell.parentNode.removeChild(activeCell);
      if (previousCell) {
        setActiveCell(previousCell);
      } else if (nextCell) {
        setActiveCell(nextCell);
      }
    }, 300);
  }
}

// Save notebook with improved data capture
function saveNotebook() {
  const cells = [];
  document.querySelectorAll(".cell").forEach((cell) => {
    const type = cell.dataset.cellType;
    const textarea = cell.querySelector("textarea");
    const content = textarea ? textarea.value : "";

    if (type === "code") {
      const operationType = cell.querySelector(".operation-type").value;
      const variable = cell.querySelector(".var-input input")
        ? cell.querySelector(".var-input input").value
        : "x";

      // Prepare cell data object
      const cellData = {
        type: type,
        content: content,
        operationType: operationType,
        variable: variable,
      };

      // Save additional parameters based on operation type
      if (operationType === "limit") {
        cellData.point = cell.querySelector(".point").value || "0";
        cellData.direction = cell.querySelector(".limit-sort").value || "+";
        cellData.maxLhopitalCount = cell.querySelector(".max-lhopital-count").value || "5";
      }

      if (operationType === "integral") {
        const radioButtonGroupName = `include-option-${cell.dataset.cellId}`;
        const selectedRadio = cell.querySelector(`input[name="${radioButtonGroupName}"]:checked`);
        cellData.includeSpecialMatchers = selectedRadio ? selectedRadio.value : "include";
        const upperInput = cell.querySelector(".integral-upper");
        const lowerInput = cell.querySelector(".integral-lower");
        cellData.integralUpper = upperInput ? upperInput.value : "";
        cellData.integralLower = lowerInput ? lowerInput.value : "";
      }

      if (operationType === "ref") {
        cellData.targetForm = cell.querySelector(".ref-sort").value || "ref";
      }

      if (operationType === "operations") {
        cellData.operationSymbol = cell.querySelector(".operations-sort").value || "+";
      }

      if (operationType === "transform-1") {
        cellData.transform1Type = cell.querySelector(".transform-1-sort").value || "basis_change";
      }

      if (operationType === "transform-2") {
        cellData.transform2Type = cell.querySelector(".transform-2-sort").value || "find_matrix";
      }

      if (operationType === "det") {
        cellData.detType = cell.querySelector(".det-sort").value || "1";
      }

      if (operationType === "expr") {
        cellData.exprSort = cell.querySelector(".expr-sort").value || "complexity";
        cellData.exprDepth = cell.querySelector(".expr-depth").value || "3";
        // cellData.geneSort = cell.querySelector(".gene-sort").value || "false";
      }

      cells.push(cellData);
    } else {
      cells.push({
        type: type,
        content: content,
      });
    }
  });

  const notebookData = JSON.stringify(cells, null, 2);
  const blob = new Blob([notebookData], { type: "application/json" });
  const url = URL.createObjectURL(blob);

  const a = document.createElement("a");
  a.href = url;
  a.download = "highdream-notebook.json";
  a.click();

  URL.revokeObjectURL(url);
  showNotification("笔记本已保存", "success");
}

// Load notebook with improved data restoration
function loadNotebook() {
  const input = document.createElement("input");
  input.type = "file";
  input.accept = ".json";

  input.onchange = function (e) {
    const file = e.target.files[0];
    const reader = new FileReader();

    reader.onload = function (e) {
      try {
        const cells = JSON.parse(e.target.result);
        const notebook = document.getElementById("notebook");
        notebook.innerHTML = "";

        cells.forEach((cellData, index) => {
          const cell = createCellElement(index, cellData.type);
          setupCellEvents(cell);

          if (cellData.type === "code") {
            const textarea = cell.querySelector("textarea");
            const operationTypeSelect = cell.querySelector(".operation-type");
            const varInputField = cell.querySelector(".var-input input");

            if (textarea) textarea.value = cellData.content || "";
            if (operationTypeSelect) operationTypeSelect.value = cellData.operationType || "diff";
            if (varInputField) varInputField.value = cellData.variable || "x";

            // Restore additional parameters based on operation type
            if (cellData.operationType === "limit") {
              const pointInput = cell.querySelector(".point");
              const directionSelect = cell.querySelector(".limit-sort");
              const maxLhopitalInput = cell.querySelector(".max-lhopital-count");

              if (pointInput) pointInput.value = cellData.point || "0";
              if (directionSelect) directionSelect.value = cellData.direction || "+";
              if (maxLhopitalInput) maxLhopitalInput.value = cellData.maxLhopitalCount || "5";
            }

            if (cellData.operationType === "integral") {
              const upperInput = cell.querySelector(".integral-upper");
              const lowerInput = cell.querySelector(".integral-lower");
              if (upperInput) upperInput.value = cellData.integralUpper || "";
              if (lowerInput) lowerInput.value = cellData.integralLower || "";
              const which = cellData.includeSpecialMatchers || "include";
              const radio = cell.querySelector(`.integral-controls input[type="radio"][value="${which}"]`);
              if (radio) radio.checked = true;
            }

            if (cellData.operationType === "ref") {
              const refSort = cell.querySelector(".ref-sort");
              if (refSort) refSort.value = cellData.targetForm || "ref";
            }

            if (cellData.operationType === "operations") {
              const operationsSort = cell.querySelector(".operations-sort");
              if (operationsSort) operationsSort.value = cellData.operationSymbol || "+";
            }

            if (cellData.operationType === "transform-1") {
              const transform1Sort = cell.querySelector(".transform-1-sort");
              if (transform1Sort) transform1Sort.value = cellData.transform1Type || "basis_change";
            }

            if (cellData.operationType === "transform-2") {
              const transform2Sort = cell.querySelector(".transform-2-sort");
              if (transform2Sort) transform2Sort.value = cellData.transform2Type || "find_matrix";
            }

            if (cellData.operationType === "det") {
              const detSort = cell.querySelector(".det-sort");
              if (detSort) detSort.value = cellData.detType || "1";
            }

            if (cellData.operationType === "expr") {
              const exprSort = cell.querySelector(".expr-sort");
              const exprDepth = cell.querySelector(".expr-depth");
              // const geneSort = cell.querySelector(".gene-sort");

              if (exprSort) exprSort.value = cellData.exprSort || "complexity";
              if (exprDepth) exprDepth.value = cellData.exprDepth || "3";
            }

            // Trigger change event to show/hide appropriate controls
            if (operationTypeSelect) {
              operationTypeSelect.dispatchEvent(new Event('change'));
            }
          } else {
            const textarea = cell.querySelector("textarea");
            if (textarea) textarea.value = cellData.content || "";
          }

          notebook.appendChild(cell);
          if (cellData.type === "code") applyInputMode(cell);
        });

        cellCounter = cells.length;
        if (cells.length > 0) {
          setActiveCell(document.querySelector(".cell"));
        }

        showNotification("笔记本已加载", "success");
      } catch (err) {
        showNotification("加载失败: " + err.message, "error");
      }
    };

    reader.readAsText(file);
  };

  input.click();
}

// Subscribe to task status via SSE
// ---- Step-linking: open a derivation sub-expression in a fresh cell ----
function collectStepsPayload(cell) {
  const operationType = cell.querySelector(".operation-type").value;
  const expression = readExpr(cell);
  const varEl = cell.querySelector(".var-input input");
  const variable = (varEl && varEl.value.trim()) || "x";
  const payload = { operation_type: operationType, expression: expression, variable: variable };
  if (operationType === "limit") {
    const pointEl = cell.querySelector(".point");
    const dirEl = cell.querySelector(".limit-sort");
    payload.point = (pointEl && pointEl.value) || "0";
    payload.direction = (dirEl && dirEl.value) || "+";
  }
  if (operationType === "integral") {
    const upperEl = cell.querySelector(".integral-upper");
    const lowerEl = cell.querySelector(".integral-lower");
    payload.upper = (upperEl && upperEl.value.trim()) || "";
    payload.lower = (lowerEl && lowerEl.value.trim()) || "";
  }
  return payload;
}

function spawnCellFromSubtask(sub) {
  addNewCell();
  const cell = document.getElementById("notebook").lastElementChild;
  if (!cell) return;
  const opSel = cell.querySelector(".operation-type");
  if (opSel) {
    opSel.value = sub.op;
    opSel.dispatchEvent(new Event("change"));
  }
  const ta = cell.querySelector("textarea");
  if (ta) ta.value = sub.expression || "";
  const varInput = cell.querySelector(".var-input input");
  if (varInput && sub.variable) varInput.value = sub.variable;
  if (sub.op === "limit") {
    const p = cell.querySelector(".point");
    if (p && sub.point) p.value = sub.point;
    const d = cell.querySelector(".limit-sort");
    if (d && sub.direction) d.value = sub.direction;
  }
  if (sub.op === "integral") {
    const u = cell.querySelector(".integral-upper");
    if (u) u.value = sub.upper || "";
    const l = cell.querySelector(".integral-lower");
    if (l) l.value = sub.lower || "";
  }
  applyInputMode(cell);
  runCell(cell);
}

function attachStepLinks(outputArea, cell) {
  if (!cell) return;
  let payload;
  try {
    payload = collectStepsPayload(cell);
  } catch (e) {
    return;
  }
  fetch("/api/steps", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  })
    .then((r) => r.json())
    .then((data) => {
      if (!data || !data.success || !Array.isArray(data.steps)) return;
      const box = document.createElement("div");
      box.className = "step-links";
      box.style.cssText = "margin-top:8px;display:flex;flex-wrap:wrap;gap:6px;align-items:center;";
      const title = document.createElement("span");
      title.textContent = "可继续计算的子式：";
      title.style.cssText = "font-size:12px;color:#888;margin-right:4px;";
      box.appendChild(title);
      let count = 0;
      data.steps.forEach((st) => {
        (st.subtasks || []).forEach((sub) => {
          if (!sub.op) return;
          count += 1;
          const btn = document.createElement("button");
          btn.className = "step-link-btn";
          btn.type = "button";
          btn.style.cssText = "cursor:pointer;font-size:12px;padding:2px 8px;border:1px solid #ccc;border-radius:4px;background:#f7f7f7;";
          const prefix = sub.op === "integral" ? "∫" : sub.op === "diff" ? ("d/d" + (sub.variable || "x")) : "lim";
          btn.textContent = prefix + " " + sub.expression;
          btn.title = "在新单元格中计算此式";
          btn.addEventListener("click", function () { spawnCellFromSubtask(sub); });
          box.appendChild(btn);
        });
      });
      if (count > 0) outputArea.appendChild(box);
    })
    .catch(() => {});
}

function subscribeTask(taskId, outputArea, runButton, operationType) {
  const es = new EventSource(`/api/stream?task_id=${taskId}`);
  pollTimers.set(taskId, es);

  es.addEventListener('status', function (e) {
    const data = JSON.parse(e.data);
    updateComputingStatus(outputArea, data.status);
  });

  es.addEventListener('completed', function (e) {
    es.close();
    pollTimers.delete(taskId);

    runButton.classList.remove("running");
    updateKernelStatus('idle');

    const task = JSON.parse(e.data);

    if (operationType === "expr") {
      renderExpressionResult(outputArea, {
        success: true,
        result: task.result[0]
      });
    } else {
      outputArea.innerHTML = `<div class="math-output">${task.result}</div>`;

      addMathOutputControls();

      if (typeof MathJax !== "undefined") {
        MathJax.typesetPromise([outputArea]);
      }

      outputArea.classList.add("result-animation");
      setTimeout(() => {
        outputArea.classList.remove("result-animation");
      }, 1000);

      if (["diff", "integral", "limit"].includes(operationType)) {
        attachStepLinks(outputArea, outputArea.closest(".cell"));
      }
    }
  });

  es.addEventListener('failed', function (e) {
    es.close();
    pollTimers.delete(taskId);

    runButton.classList.remove("running");
    updateKernelStatus('idle');

    let msg = '任务失败';
    try {
      const data = JSON.parse(e.data);
      msg = data.error || msg;
    } catch (_) {}

    outputArea.innerHTML = `<div class="error">错误: ${msg}</div>`;
    updateKernelStatus('error');
  });

  es.onerror = function () {
    // EventSource auto-reconnects; only clean up if permanently closed
    if (es.readyState === EventSource.CLOSED) {
      pollTimers.delete(taskId);
      runButton.classList.remove("running");
      updateKernelStatus('idle');
      outputArea.innerHTML = '<div class="error">连接中断</div>';
      updateKernelStatus('error');
    }
  };
}

// Update computing status display
function updateComputingStatus(outputArea, status) {
  const statusMessages = {
    'pending': '任务已提交，等待处理...',
    'running': '正在计算中，请稍候...'
  };

  const statusIcons = {
    'pending': '<i class="fas fa-clock"></i>',
    'running': '<div class="loading-spinner"></div>'
  };

  const message = statusMessages[status] || '处理中...';
  const icon = statusIcons[status] || '<div class="loading-spinner"></div>';

  outputArea.innerHTML = `<div class="computing">
    ${icon}
    <div class="loading-text">${message}</div>
  </div>`;
}