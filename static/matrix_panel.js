// Global variables
let cellCounter = 0;
let activeCellId = null;
let isDarkTheme = localStorage.getItem('darkTheme') === 'true';
let isKeyboardVisible = false;
let kernelStatus = 'idle';

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
  document.getElementById("toggleKeyboardBtn").addEventListener("click", toggleKeyboard);
  document.getElementById("closeKeyboardBtn").addEventListener("click", toggleKeyboard);
  document.getElementById("toggleThemeBtn").addEventListener("click", toggleTheme);
  document.getElementById("saveNotebookBtn").addEventListener("click", saveNotebook);
  document.getElementById("loadNotebookBtn").addEventListener("click", loadNotebook);

  // Cancel active cell when clicking elsewhere in the document
  document.addEventListener("click", function (e) {
    if (!e.target.closest(".cell") && !e.target.closest("#math-keyboard") && !e.target.closest("#header") && !e.target.closest("#toolbar")) {
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

  // Bind virtual keyboard buttons
  document.querySelectorAll(".keyboard-key").forEach((key) => {
    key.addEventListener("click", function () {
      const value = this.getAttribute("data-value");
      const activeCell = document.querySelector(".cell-active");
      if (activeCell) {
        const textarea = activeCell.querySelector("textarea");
        if (textarea) {
          const cursorPos = textarea.selectionStart;
          const textBefore = textarea.value.substring(0, cursorPos);
          const textAfter = textarea.value.substring(cursorPos);

          if (value.includes("{}")) {
            const parts = value.split("{}");
            textarea.value = textBefore + parts[0] + textAfter;
            textarea.selectionStart = cursorPos + parts[0].length;
            textarea.selectionEnd = cursorPos + parts[0].length;
          } else {
            textarea.value = textBefore + value + textAfter;
            textarea.selectionStart = cursorPos + value.length;
            textarea.selectionEnd = cursorPos + value.length;
          }

          textarea.focus();
          textarea.dispatchEvent(new Event("input"));
        }
      }
    });
  });
}

// Set up cell events
function setupCellEvents(cell) {
  const textarea = cell.querySelector("textarea");
  const runButton = cell.querySelector(".run-cell-button");
  const opretationButton = cell.querySelector(".operation-button");
  const keyboardButton = cell.querySelector(".keyboard-button");
  const addCellButton = cell.querySelector(".add-cell-button");

  const cellType = cell.dataset.cellType;

  if (cellType === "code") {
    // Analyze
    runButton.addEventListener("click", function () {
      runCell(cell);
    });

    // Operation
    opretationButton.addEventListener("click", function () {
      runOpretation(cell);
    });

    // Textbox input event - real-time preview
    textarea.addEventListener("input", function () {
      debounce(() => {
        renderMathPreview(cell);
      }, 500)();
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

  // For markdown cells, show input box and hide rendered content when activated
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
      textarea.focus();
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

  if (type === "code") {
    cell.innerHTML = `
      <div class="cell-prompt">Calculator:</div>
      <div class="cell-content-wrapper">
        <div class="cell-content">
          <div class="cell-input">
            <div class="cell-controls">
              <button class="run-cell-button" title="分析">
                <i class="fas fa-search"></i>
              </button>
              <button class="operation-button" title="操作">
                <i class="fas fa-calculator"></i>
              </button>
              <button class="keyboard-button" title="虚拟键盘">
                <i class="fas fa-keyboard"></i>
              </button>
            </div>
            <textarea placeholder="输入数学表达式..." class="math-input"></textarea>
            <div class="math-preview"></div>
          </div>
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

// Analyze matrix
function runCell(cell) {
  const expression = cell.querySelector("textarea").value.trim();

  if (!expression) {
    showNotification("请输入表达式", "error");
    return;
  }

  // Pass data to backend
  const payload = {
    expression: expression,
  };

  // Open a new page with POST method and pass data
  const form = document.createElement('form');
  form.method = 'POST';
  form.action = '/matrix_analysis';
  form.target = '_blank';

  Object.keys(payload).forEach(key => {
    const input = document.createElement('input');
    input.type = 'hidden';
    input.name = key;
    input.value = payload[key];
    form.appendChild(input);
  });

  document.body.appendChild(form);
  form.submit();
  document.body.removeChild(form);
}

// Operate matrix
function runOpretation(cell) {
  const expression = cell.querySelector("textarea").value.trim();

  if (!expression) {
    showNotification("请输入表达式", "error");
    return;
  }

  // Pass data to backend
  const payload = {
    expression: expression,
  };

  // Open a new page with POST method and pass data
  const form = document.createElement('form');
  form.method = 'POST';
  form.action = '/matrix_lab';
  form.target = '_blank';

  Object.keys(payload).forEach(key => {
    const input = document.createElement('input');
    input.type = 'hidden';
    input.name = key;
    input.value = payload[key];
    form.appendChild(input);
  });

  document.body.appendChild(form);
  form.submit();
  document.body.removeChild(form);
}

// Render math preview
function renderMathPreview(cell) {
  const textarea = cell.querySelector("textarea");
  const content = textarea.value.trim();
  const previewArea = cell.querySelector(".math-preview");
  const operationType = 'matrix';

  if (!content) {
    previewArea.innerHTML = "";
    return;
  }

  fetch("/parse", {
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
        previewArea.innerHTML = `$$${data.result} $$`;
        if (typeof MathJax !== "undefined") {
          MathJax.typesetPromise([previewArea]);
        }
      } else {
        previewArea.innerHTML = `<span span class="preview-error" > 解析错误: ${data.error}</span > `;
      }
    })
    .catch((error) => {
      previewArea.innerHTML = `<span span class="preview-error" > Request Error: ${error.message}</span > `;
    });
}

// Show notification
function showNotification(message, type = "success") {
  const notification = document.createElement("div");
  notification.className = `notification notification-${type} `;
  notification.innerHTML = `
      <div div class="notification-content" >
      <i class="fas ${type === 'success' ? 'fa-check-circle' : 'fa-exclamation-circle'}"></i>
      <span>${message}</span>
    </div >
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
  const themeBtn = document.getElementById("toggleThemeBtn");
  if (isDark) {
    document.documentElement.style.setProperty("--jp-ui-font-color0", "rgba(255, 255, 255, 0.87)");
    document.documentElement.style.setProperty("--jp-ui-font-color1", "rgba(255, 255, 255, 0.6)");
    document.documentElement.style.setProperty("--jp-layout-color0", "#1e1e1e");
    document.documentElement.style.setProperty("--jp-layout-color1", "#2d2d30");
    document.documentElement.style.setProperty("--jp-layout-color2", "#252526");
    document.documentElement.style.setProperty("--jp-layout-color3", "#3e3e42");
    document.documentElement.style.setProperty("--jp-border-color0", "#454545");
    document.documentElement.style.setProperty("--jp-border-color1", "#606060");
    document.documentElement.style.setProperty("--jp-border-color2", "#808080");
    themeBtn.innerHTML = '<i class="fas fa-sun"></i>';
    isDarkTheme = true;
    document.querySelector('meta[name="theme-color"]').setAttribute('content', '#1e1e1e');
  } else {
    document.documentElement.style.setProperty("--jp-ui-font-color0", "rgba(0, 0, 0, 0.87)");
    document.documentElement.style.setProperty("--jp-ui-font-color1", "rgba(0, 0, 0, 0.54)");
    document.documentElement.style.setProperty("--jp-layout-color0", "white");
    document.documentElement.style.setProperty("--jp-layout-color1", "#f7f7f7");
    document.documentElement.style.setProperty("--jp-layout-color2", "#e7e7e7");
    document.documentElement.style.setProperty("--jp-layout-color3", "#d7d7d7");
    document.documentElement.style.setProperty("--jp-border-color0", "#e0e0e0");
    document.documentElement.style.setProperty("--jp-border-color1", "#bdbdbd");
    document.documentElement.style.setProperty("--jp-border-color2", "#9e9e9e");
    themeBtn.innerHTML = '<i class="fas fa-moon"></i>';
    isDarkTheme = false;
  }

  localStorage.setItem('darkTheme', isDarkTheme);
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
  a.download = "matrix_panel.json";
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
    if (!file) return; // Handle case where user cancels file selection

    const reader = new FileReader();

    reader.onload = function (e) {
      try {
        const cells = JSON.parse(e.target.result);
        const notebook = document.getElementById("notebook");
        notebook.innerHTML = "";

        cells.forEach((cellData, index) => {
          const cell = createCellElement(index, cellData.type);

          // Unified handling for both code and markdown cells
          const textarea = cell.querySelector("textarea");
          if (textarea) textarea.value = cellData.content || "";

          notebook.appendChild(cell);
          setupCellEvents(cell);
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

    reader.onerror = function () {
      showNotification("文件读取失败", "error");
    };

    reader.readAsText(file);
  };

  input.click();
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
    // Line break
    .replace(/\n/gim, '<br>')
    // Link
    .replace(/\[([^\]]+)\]\(([^)]+)\)/gim, '<a href="$2" target="_blank">$1</a>')
    // Unordered list
    .replace(/^\s*[-*] (.*$)/gim, '<li>$1</li>')
    // Handle list wrapping
    .replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>')
    // Quote
    .replace(/^> (.*$)/gim, '<blockquote>$1</blockquote>');

  // Handle multiple consecutive line breaks (paragraphs)
  html = html.replace(/(<br>\s*){2,}/gim, '</p><p>');
  html = '<p>' + html + '</p>';

  markdownRendered.innerHTML = html;
}