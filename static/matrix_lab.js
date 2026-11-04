// Global variables
let cellCounter = 0;
let activeCellId = null;
let isDarkTheme = localStorage.getItem('darkTheme') === 'true';
let kernelStatus = 'idle';

let data_copy = {};
let is_record_all_steps = 'True';

// Escape untrusted text (server error echoes, JS error messages) before innerHTML.
function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

// Only allow non-executing URL schemes for markdown links (block javascript:, etc.).
function safeUrl(url) {
  const trimmed = String(url).trim();
  return /^(https?:|mailto:|#|\/)/i.test(trimmed) ? trimmed : "#";
}

// True when a keydown came from a text editor (input/textarea/select,
// contenteditable, MathLive field). Single-letter Ctrl/Cmd shortcuts must not
// hijack editing there -- Ctrl+D used to delete the whole cell mid-typing.
function isTextEditingTarget(target) {
  if (!target || !target.tagName) return false;
  const tag = target.tagName;
  if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || tag === "MATH-FIELD") {
    return true;
  }
  return !!(target.closest && target.closest('[contenteditable="true"]'));
}

// Add control buttons for all math-output areas
function addMathOutputControls() {
  const mathOutputs = document.querySelectorAll('.math-output');

  mathOutputs.forEach(output => {
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
        if (window.MathJax && typeof MathJax.typesetPromise === "function") {
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

  // Copy content to fullscreen display area, but preserve original content
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
  if (window.MathJax && typeof MathJax.typesetPromise === "function") {
    MathJax.typesetPromise([content]);
  }
}

// Initialize after page loads
document.addEventListener("DOMContentLoaded", function () {
  applyTheme(isDarkTheme);
  addNewCell();
  bindEvents();
  animateMathBackground();
});

// Mathematical background animation function
function animateMathBackground() {
  const elements = document.querySelectorAll('.math-bg-element');
  elements.forEach(el => {
    const delay = Math.random() * 2;
    const duration = 15 + Math.random() * 10;
    el.style.animation = `float ${duration}s ease-in-out ${delay}s infinite`;
  });
}

// Bind events
function bindEvents() {
  document.getElementById("addCellBtn").addEventListener("click", () => addNewCell());
  document.getElementById("addMdCellBtn").addEventListener("click", () => addNewCell("markdown"));
  document.getElementById("moveCellUpBtn").addEventListener("click", moveActiveCellUp);
  document.getElementById("moveCellDownBtn").addEventListener("click", moveActiveCellDown);
  document.getElementById("deleteCellBtn").addEventListener("click", deleteActiveCell);
  const _themeBtn = document.getElementById("toggleThemeBtn");
  if (_themeBtn) _themeBtn.addEventListener("click", toggleTheme);
  document.getElementById("saveNotebookBtn").addEventListener("click", saveNotebook);
  document.getElementById("loadNotebookBtn").addEventListener("click", loadNotebook);

  // Load / replace the initial matrix. The field is the same WYSIWYG editor as
  // the notebook's universal input; convert_form turns a filled bmatrix back
  // into the nested-list form the matrix engine consumes.
  const startForm = document.getElementById("matrix-lab-start");
  if (startForm) {
    startForm.addEventListener("submit", function (event) {
      event.preventDefault();
      const textarea = startForm.querySelector("textarea");
      const value = (textarea && textarea.value.trim()) || "";
      if (!value) {
        showNotification("请输入初始矩阵", "error");
        return;
      }
      if (/\\(placeholder|square|Box)\b/.test(value)) {
        showNotification("请先填写矩阵里的占位框", "error");
        return;
      }
      normalizeUniversal(value).then(function (expression) {
        if (!expression) return;
        window.location.href = "/matrix_lab?expression=" + encodeURIComponent(expression);
      });
    });
  }

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

  // Cancel active cell when clicking other areas of the document
  document.addEventListener("click", function (e) {
    if (!e.target.closest(".cell") && !e.target.closest("#math-keyboard") && !e.target.closest("#hd-top-menu") && !e.target.closest("#toolbar") && !e.target.closest(".hd-select-menu") && !e.target.closest(".hd-select")) {
      document.querySelectorAll(".cell").forEach((cell) => {
        cell.classList.remove("cell-active");
        if (cell.dataset.cellType === "markdown") {
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

    // The remaining shortcuts are document-level only (not while editing).
    if (isTextEditingTarget(e.target)) return;

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
  });

  // Bind virtual keyboard buttons
  document.querySelectorAll(".keyboard-key").forEach((key) => {
    key.addEventListener("click", function () {
      const latex = this.getAttribute("data-latex");
      const insert = this.getAttribute("data-insert") || latex || this.getAttribute("data-value");
      const activeCell = document.querySelector(".cell-active");
      if (!activeCell) return;
      const field = activeCell.querySelector("math-field.hd-universal-field");
      if (field) {
        field.focus();
        if (typeof field.insert === "function") field.insert(insert);
        else field.value = (field.value || "") + insert;
        return;
      }
      const textarea = activeCell.querySelector("textarea");
      if (textarea) {
        const cursorPos = textarea.selectionStart;
        const textBefore = textarea.value.substring(0, cursorPos);
        const textAfter = textarea.value.substring(cursorPos);
        textarea.value = textBefore + insert + textAfter;
        textarea.selectionStart = cursorPos + insert.length;
        textarea.selectionEnd = cursorPos + insert.length;
        textarea.focus();
        textarea.dispatchEvent(new Event("input"));
      }
    });
  });

  // Get expression
  const expressionElement = document.querySelector('.matrix-preview p');

  if (expressionElement) {
    let expressionText = expressionElement.textContent;
    data_copy['expression'] = expressionText;
  }
  renderMathPreview();
}

// Set up cell events
function setupCellEvents(cell) {
  const textarea = cell.querySelector("textarea");
  const runButton = cell.querySelector(".run-cell-button");
  const addCellButton = cell.querySelector(".add-cell-button");

  const cellType = cell.dataset.cellType;

  if (cellType === "code") {
    // Run
    runButton.addEventListener("click", function () {
      runCell(cell);
    });

    // Textbox input event - live preview (debounced: build the wrapper once)
    textarea.addEventListener("input", debounce(function () {
      renderMathPreview(cell);
    }, 500));
  } else if (cellType === "markdown") {
    const cellPrompt = cell.querySelector(".cell-prompt");
    // Textbox click event - set active cell
    cellPrompt.addEventListener("click", function () {
      setActiveCell(cell);
    });
  }

  // Add cell button
  addCellButton.addEventListener("click", function () {
    addCellBelow(cell);
  });

  // Textbox click event - set active cell
  textarea.addEventListener("click", function () {
    setActiveCell(cell);
  });
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

    // For markdown cells, hide input box and show rendered content when inactive
    if (c.dataset.cellType === "markdown") {
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

  // For markdown cells, show input box and hide rendered content when active
  if (cell.dataset.cellType === "markdown") {
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
    const textarea = cell.querySelector("textarea");
    if (textarea) {
      const field = cell.querySelector("math-field.hd-universal-field");
      if (field) field.focus();
      else textarea.focus();
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

  const radioButtonGroupName = `record-option-${id}`;

  if (type === "code") {
    cell.innerHTML = `
      <div class="cell-content-wrapper">
        <div class="cell-content">
          <div class="cell-input">
            <div class="cell-controls">
              <button class="run-cell-button" title="运行">
                <i class="fas fa-play"></i>
              </button>
              <label class="record-toggle">
               <input type="radio" name="${radioButtonGroupName}" value="all" checked>
               <span>记录中间结果</span>
              </label>
              <label class="record-toggle">
               <input type="radio" name="${radioButtonGroupName}" value="result">
               <span>仅记录最终结果</span>
              </label>
            </div>
            <textarea placeholder="输入操作，例如 rref / transpose / inverse；也可点右侧操作键" class="math-input" data-universal-input></textarea>
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
      <div class="cell-prompt">Md:</div>
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

// The WYSIWYG editor stores LaTeX. Turn it back into operator form before the
// matrix step engine sees it; unrecognised input is returned unchanged.
function normalizeUniversal(expression) {
  if (!expression) return Promise.resolve(expression);
  return fetch("/api/convert_form", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ expression: expression }),
  })
    .then((response) => response.json())
    .then((data) => (data && data.success && typeof data.result === "string" ? data.result : expression))
    .catch(() => expression);
}

// Run
function runCell(cell) {
  const runButton = cell.querySelector(".run-cell-button");
  runButton.classList.add("running");
  let outputArea = cell.querySelector(".cell-output");

  const operations = cell.querySelector("textarea").value.trim();
  if (!operations) {
    runButton.classList.remove("running");
    showNotification("请输入操作", "error");
    return;
  }

  if (/\\(placeholder|square|Box)\b/.test(operations)) {
    runButton.classList.remove("running");
    showNotification("请先填写矩阵里的占位框", "error");
    return;
  }

  if (!data_copy["expression"]) {
    runButton.classList.remove("running");
    showNotification("请先在上方载入初始矩阵", "error");
    return;
  }

  const radioButtonGroupName = `record-option-${cell.dataset.cellId}`;
  const selectedRadio = cell.querySelector(`input[name="${radioButtonGroupName}"]:checked`).value === "all";
  if (selectedRadio) {
    is_record_all_steps = 'True'
  }
  else {
    is_record_all_steps = 'False'
  }

  normalizeUniversal(operations).then(function (normalizedOps) {
    // Send data to backend
    const payload = {
      expression: data_copy['expression'],
      operations: normalizedOps,
      record_all_steps: is_record_all_steps
    };

    return fetch("/api/matrix_cal", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });
  })
    .then((response) => response.json())
    .then((data) => {
      runButton.classList.remove("running");
      if (!data || data.success === false) {
        outputArea.innerHTML = `<div class="error">${escapeHtml((data && data.error) || "计算失败")}</div>`;
        return;
      }
      outputArea.innerHTML = `<div class="math-output">${escapeHtml(data.steps || "")}</div>`;
      // Add control buttons
      addMathOutputControls();
      if (window.MathJax && typeof MathJax.typesetPromise === "function") {
        MathJax.typesetPromise([outputArea]);
      }
      outputArea.classList.add("result-animation");
      setTimeout(() => {
        outputArea.classList.remove("result-animation");
      }, 1000);
    })
    .catch((error) => {
      runButton.classList.remove("running");
      outputArea.innerHTML = `<div class="error">请求错误: ${escapeHtml(error.message)}</div>`;
    });
}

// Render the live editor's LaTeX directly (placeholders become \square).
function renderMathPreview(cell) {
  if (!cell) return;
  const field = cell.querySelector("math-field.hd-universal-field");
  const textarea = cell.querySelector("textarea");
  if (!field && !textarea) return;
  const content = (field ? field.value : textarea.value).trim();
  const previewArea = cell.querySelector(".math-preview");
  if (!previewArea) return;

  if (!content) {
    previewArea.innerHTML = "";
    return;
  }

  const latex = content
    .replace(/\\placeholder\s*\{\s*\}/g, "\\square")
    .replace(/\\placeholder\b/g, "\\square")
    .replace(/\{\s*\}/g, "{\\square}");
  previewArea.innerHTML = `$$${escapeHtml(latex)}$$`;
  if (window.MathJax && typeof MathJax.typesetPromise === "function") {
    MathJax.typesetPromise([previewArea]);
  }
}

// Show notification
function showNotification(message, type = "success") {
  const notification = document.createElement("div");
  notification.className = `notification notification-${type}`;
  notification.innerHTML = `
    <div class="notification-content">
      <i class="fas ${type === 'success' ? 'fa-check-circle' : 'fa-exclamation-circle'}"></i>
      <span>${escapeHtml(message)}</span>
    </div>
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

// Save notebook
function saveNotebook() {
  const cells = [];
  document.querySelectorAll(".cell").forEach((cell) => {
    const type = cell.dataset.cellType;
    const textarea = cell.querySelector("textarea");
    const content = textarea ? textarea.value : "";

    cells.push({
      type: type,
      content: content,
    });
  });

  const notebookData = JSON.stringify(cells, null, 2);
  const blob = new Blob([notebookData], { type: "application/json" });
  const url = URL.createObjectURL(blob);

  const a = document.createElement("a");
  a.href = url;
  a.download = "matrix_lab.json";
  a.click();

  URL.revokeObjectURL(url);
  showNotification("笔记本已保存", "success");
}

// Load notebook
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

        cellCounter = 0; // Reset cell counter

        cells.forEach((cellData, _index) => {
          const cell = createCellElement(cellCounter, cellData.type);

          const textarea = cell.querySelector("textarea");
          if (textarea) textarea.value = cellData.content || "";

          notebook.appendChild(cell);
          setupCellEvents(cell);
          cellCounter++;
        });

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

function renderMathPreview() {
  const expressionElement = document.querySelector('.matrix-preview p');
  const previewArea = document.querySelector(".matrix-preview .render");
  if (!expressionElement || !previewArea) return;
  let content = expressionElement.textContent;
  const operationType = 'matrix';

  if (!content) {
    previewArea.innerHTML = "";
    return;
  }

  fetch("/api/parse", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      expression: content,
      operation_type: operationType,
    }),
  })
    .then((response) => response.json())
    .then((data) => {
      if (data.success) {
        previewArea.innerHTML = `$$${escapeHtml(data.result)}$$`;
        if (window.MathJax && typeof MathJax.typesetPromise === "function") {
          MathJax.typesetPromise([previewArea]);
        }
      } else {
        previewArea.innerHTML = `<span class="preview-error">解析错误: ${escapeHtml(data.error)}</span>`;
      }
    })
    .catch((error) => {
      previewArea.innerHTML = `<span class="preview-error">请求错误: ${escapeHtml(error.message)}</span>`;
    });
}

// Render Markdown content
function renderMarkdown(cell) {
  const markdownInput = cell.querySelector(".markdown-input");
  const markdownRendered = cell.querySelector(".markdown-render");

  if (!markdownInput || !markdownRendered) return;

  // Escape first: all raw HTML from the cell is neutralized before we re-add markup.
  const content = escapeHtml(markdownInput.value);

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
    // Line break
    .replace(/\n/gim, '<br>')
    // Link (URL scheme allowlisted to block javascript:)
    .replace(/\[([^\]]+)\]\(([^)]+)\)/gim, (m, text, url) => '<a href="' + safeUrl(url) + '" target="_blank" rel="noopener noreferrer">' + text + '</a>')
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