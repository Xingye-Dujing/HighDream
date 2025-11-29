// 全局变量
let cellCounter = 0;
let activeCellId = null;
let isDarkTheme = localStorage.getItem('darkTheme') === 'true';
let isKeyboardVisible = false;
let kernelStatus = 'idle';

let data_copy = {};
let is_record_all_steps = 'True';

// 为所有 math-output 区域添加控制按钮
function addMathOutputControls() {
  const mathOutputs = document.querySelectorAll('.math-output');

  mathOutputs.forEach(output => {
    // 检查是否已经添加了控制按钮
    if (!output.querySelector('.math-controls')) {
      // 创建控制按钮容器
      const controls = document.createElement('div');
      controls.className = 'math-controls';

      // 创建折叠按钮
      const collapseBtn = document.createElement('button');
      collapseBtn.className = 'math-control-btn';
      collapseBtn.title = '折叠/展开';
      collapseBtn.innerHTML = '<i class="fas fa-chevron-up"></i>';

      // 创建全屏按钮
      const fullscreenBtn = document.createElement('button');
      fullscreenBtn.className = 'math-control-btn';
      fullscreenBtn.title = '全屏显示';
      fullscreenBtn.innerHTML = '<i class="fas fa-expand"></i>';

      // 添加按钮到容器
      controls.appendChild(collapseBtn);
      controls.appendChild(fullscreenBtn);

      // 添加容器到math-output
      output.appendChild(controls);

      // 添加折叠功能
      collapseBtn.addEventListener('click', function () {
        output.classList.toggle('collapsed');

        // 更新图标
        if (output.classList.contains('collapsed')) {
          collapseBtn.innerHTML = '<i class="fas fa-chevron-down"></i>';
        } else {
          collapseBtn.innerHTML = '<i class="fas fa-chevron-up"></i>';
        }

        // 重新渲染 MathJax
        if (window.MathJax) {
          MathJax.typesetPromise([output]);
        }
      });

      // 添加全屏功能
      fullscreenBtn.addEventListener('click', function () {
        openFullscreen(output);
      });
    }
  });
}

// 打开全屏显示
function openFullscreen(mathOutput) {
  const overlay = document.querySelector('.math-fullscreen-overlay');
  const content = overlay.querySelector('.math-fullscreen-body');

  // 复制内容到全屏显示区域，但保留原始内容
  const originalContent = mathOutput.cloneNode(true);

  // 移除控制按钮
  const controls = originalContent.querySelector('.math-controls');
  if (controls) {
    controls.remove();
  }

  content.innerHTML = originalContent.innerHTML;

  // 显示全屏遮罩
  overlay.classList.add('active');

  // 重新渲染 MathJax
  if (window.MathJax) {
    MathJax.typesetPromise([content]);
  }
}

// 页面加载完成后初始化
document.addEventListener("DOMContentLoaded", function () {
  applyTheme(isDarkTheme);
  addNewCell();
  bindEvents();
  animateMathBackground();
});

// 数学背景动画函数
function animateMathBackground() {
  const elements = document.querySelectorAll('.math-bg-element');
  elements.forEach(el => {
    const delay = Math.random() * 2;
    const duration = 15 + Math.random() * 10;
    el.style.animation = `float ${duration}s ease-in-out ${delay}s infinite`;
  });
}

// 绑定事件
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

  // 全屏相关事件
  const closeBtn = document.querySelector('.math-fullscreen-close');
  if (closeBtn) {
    closeBtn.addEventListener('click', function () {
      const overlay = document.querySelector('.math-fullscreen-overlay');
      overlay.classList.remove('active');
    });
  }

  // 点击遮罩背景关闭全屏
  const overlay = document.querySelector('.math-fullscreen-overlay');
  if (overlay) {
    overlay.addEventListener('click', function (e) {
      if (e.target === overlay) {
        overlay.classList.remove('active');
      }
    });
  }

  // ESC 键关闭全屏
  document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape') {
      const overlay = document.querySelector('.math-fullscreen-overlay');
      if (overlay && overlay.classList.contains('active')) {
        overlay.classList.remove('active');
      }
    }
  });

  // 点击文档其他区域时取消活动单元格
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

  // 键盘快捷键
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

  // 绑定虚拟键盘按钮
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

  // 获取表达式
  const expressionElement = document.querySelector('.matrix-preview p');

  if (expressionElement) {
    let expressionText = expressionElement.textContent;
    data_copy['expression'] = expressionText;
  }
  renderMathPreview();
}

// 设置单元格事件
function setupCellEvents(cell) {
  const textarea = cell.querySelector("textarea");
  const runButton = cell.querySelector(".run-cell-button");
  const keyboardButton = cell.querySelector(".keyboard-button");
  const addCellButton = cell.querySelector(".add-cell-button");

  const cellType = cell.dataset.cellType;

  if (cellType === "code") {
    // 运行
    runButton.addEventListener("click", function () {
      runCell(cell);
    });

    // 文本框输入事件 - 实时预览
    textarea.addEventListener("input", function () {
      debounce(() => {
        renderMathPreview(cell);
      }, 500)();
    });

    // 虚拟键盘按钮
    keyboardButton.addEventListener("click", function () {
      toggleKeyboard();
    });
  } else if (cellType === "markdown") {
    const cellPrompt = cell.querySelector(".cell-prompt");
    // 文本框点击事件 - 设置活动单元格
    cellPrompt.addEventListener("click", function () {
      setActiveCell(cell);
    });
  }

  // 添加单元格按钮
  addCellButton.addEventListener("click", function () {
    addCellBelow(cell);
  });

  // 文本框点击事件 - 设置活动单元格
  textarea.addEventListener("click", function () {
    setActiveCell(cell);
  });
}

// 防抖函数
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

// 设置活动单元格
function setActiveCell(cell) {
  document.querySelectorAll(".cell").forEach((c) => {
    c.classList.remove("cell-active");

    // 对于 markdown 单元格, 非激活时隐藏输入框显示渲染内容
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

  // 对于 markdown 单元格, 激活时显示输入框隐藏渲染内容
  if (cell.dataset.cellType === "markdown") {
    const input = cell.querySelector(".markdown-input");
    const rendered = cell.querySelector(".markdown-render");
    if (input && rendered) {
      input.style.display = "block";
      rendered.style.display = "none";

      // 聚焦到输入框
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

// 添加新单元格
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

// 创建单元格元素
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
              <button class="keyboard-button" title="虚拟键盘">
                <i class="fas fa-keyboard"></i>
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
            <textarea placeholder="输入操作表达式..." class="math-input"></textarea>
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

// 在上面添加单元格
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

// 在下面添加单元格
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

// 运行
function runCell(cell) {
  const runButton = cell.querySelector(".run-cell-button");
  runButton.classList.add("running");
  let outputArea = cell.querySelector(".cell-output");

  const operations = cell.querySelector("textarea").value.trim();
  if (!operations) {
    showNotification("请输入操作", "error");
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

  // 传递数据给后端
  const payload = {
    expression: data_copy['expression'],
    operations: operations,
    record_all_steps: is_record_all_steps
  };

  fetch("/matrix_cal", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  })
    .then((response) => response.json())
    .then((data) => {
      outputArea.innerHTML = `<div class="math-output">${data.steps}</div>`;
      // 添加控制按钮
      addMathOutputControls();
      if (typeof MathJax !== "undefined") {
        MathJax.typesetPromise([outputArea]);
      }
      outputArea.classList.add("result-animation");
      setTimeout(() => {
        outputArea.classList.remove("result-animation");
      }, 1000);
    })
    .catch((error) => {
      runButton.classList.remove("running");
      outputArea.innerHTML = `<div class="error">请求错误: ${error.message}</div>`;
    });
}

// 渲染数学预览
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
        previewArea.innerHTML = `$$${data.result}$$`;
        if (typeof MathJax !== "undefined") {
          MathJax.typesetPromise([previewArea]);
        }
      } else {
        previewArea.innerHTML = `<span class="preview-error">解析错误: ${data.error}</span>`;
      }
    })
    .catch((error) => {
      previewArea.innerHTML = `<span class="preview-error">请求错误: ${error.message}</span>`;
    });
}

// 显示通知
function showNotification(message, type = "success") {
  const notification = document.createElement("div");
  notification.className = `notification notification-${type}`;
  notification.innerHTML = `
    <div class="notification-content">
      <i class="fas ${type === 'success' ? 'fa-check-circle' : 'fa-exclamation-circle'}"></i>
      <span>${message}</span>
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

function toggleKeyboard() {
  const keyboard = document.getElementById("math-keyboard");
  const helpfulArea = document.getElementById("helpful-area");

  if (isKeyboardVisible) {
    // 隐藏键盘
    keyboard.classList.remove("keyboard-visible");
    isKeyboardVisible = false;
    helpfulArea.style.display = "none";
  } else {
    // 显示键盘
    keyboard.classList.add("keyboard-visible");
    isKeyboardVisible = true;
    // 用于占据页面底部, 在键盘打开时, 将页面向下延伸, 防止键盘挡住单元格
    helpfulArea.style.display = "flex";
  }
}

// 应用主题设置
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

// 切换主题
function toggleTheme() {
  applyTheme(!isDarkTheme);
}

// 移动活动单元格向上
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

// 移动活动单元格向下
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

// 删除活动单元格
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

// 保存笔记本
function saveNotebook() {
  const cells = [];
  document.querySelectorAll(".cell").forEach((cell) => {
    const type = cell.dataset.cellType;
    const textarea = cell.querySelector("textarea");
    const content = textarea ? textarea.value : "";

    if (type === "code") {
      cells.push({
        type: type,
        content: content,
        operationType: operationType,
        variable: variable,
      });
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
  a.download = "mathtool-notebook.json";
  a.click();

  URL.revokeObjectURL(url);
  showNotification("笔记本已保存", "success");
}

// 加载笔记本
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

          if (cellData.type === "code") {
            const textarea = cell.querySelector("textarea");

            if (textarea) textarea.value = cellData.content || "";
          } else {
            const textarea = cell.querySelector("textarea");
            if (textarea) textarea.value = cellData.content || "";
          }

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

    reader.readAsText(file);
  };

  input.click();
}

function renderMathPreview() {
  const expressionElement = document.querySelector('.matrix-preview p');
  let content = expressionElement.textContent;

  const previewArea = document.querySelector(".matrix-preview .render");
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
        previewArea.innerHTML = `$$${data.result}$$`;
        if (typeof MathJax !== "undefined") {
          MathJax.typesetPromise([previewArea]);
        }
      } else {
        previewArea.innerHTML = `<span class="preview-error">解析错误: ${data.error}</span>`;
      }
    })
    .catch((error) => {
      previewArea.innerHTML = `<span class="preview-error">请求错误: ${error.message}</span>`;
    });
}

// 渲染 Markdown 内容
function renderMarkdown(cell) {
  const markdownInput = cell.querySelector(".markdown-input");
  const markdownRendered = cell.querySelector(".markdown-render");

  if (!markdownInput || !markdownRendered) return;

  const content = markdownInput.value;

  // 简单的 Markdown 渲染实现
  let html = content
    // 标题
    .replace(/^###### (.*$)/gim, '<h6>$1</h6>')
    .replace(/^##### (.*$)/gim, '<h5>$1</h5>')
    .replace(/^#### (.*$)/gim, '<h4>$1</h4>')
    .replace(/^### (.*$)/gim, '<h3>$1</h3>')
    .replace(/^## (.*$)/gim, '<h2>$1</h2>')
    .replace(/^# (.*$)/gim, '<h1>$1</h1>')
    // 粗体
    .replace(/__(.*?)__/gim, '<strong>$1</strong>')
    // 斜体
    .replace(/_(.*?)_/gim, '<em>$1</em>')
    // 代码
    .replace(/`(.*?)`/gim, '<code>$1</code>')
    // 换行
    .replace(/\n/gim, '<br>')
    // 链接
    .replace(/\[([^\]]+)\]\(([^)]+)\)/gim, '<a href="$2" target="_blank">$1</a>')
    // 无序列表
    .replace(/^\s*[-*] (.*$)/gim, '<li>$1</li>')
    // 处理列表包装
    .replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>')
    // 引用
    .replace(/^> (.*$)/gim, '<blockquote>$1</blockquote>');

  // 处理多个连续换行(段落)
  html = html.replace(/(<br>\s*){2,}/gim, '</p><p>');
  html = '<p>' + html + '</p>';

  markdownRendered.innerHTML = html;
}