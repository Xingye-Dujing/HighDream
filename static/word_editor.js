let paragraphCounter = 0;
let activeParagraphId = null;
let isDarkTheme = localStorage.getItem('darkTheme') === 'true';
let isKeyboardVisible = false;

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

  document.getElementById('toggleKeyboardBtn').addEventListener('click', toggleKeyboard);
  document.getElementById('closeKeyboardBtn').addEventListener('click', toggleKeyboard);
  document.getElementById('toggleThemeBtn').addEventListener('click', toggleTheme);

  // Virtual keyboard events
  document.querySelectorAll('.keyboard-key').forEach((key) => {
    key.addEventListener('click', function () {
      const value = this.getAttribute('data-value');
      const activeParagraph = document.querySelector('.paragraph.active');
      if (activeParagraph) {
        const input = activeParagraph.querySelector('.code-input, .latex-input');
        if (input) {
          insertAtCursor(input, value);
          input.focus();
          input.dispatchEvent(new Event('input'));
        }
      }
    });
  });

  // Click on other areas of the document to deactivate
  document.addEventListener('click', function (e) {
    if (!e.target.closest('.paragraph') && !e.target.closest('#toolbar') && !e.target.closest('#header')) {
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

    if ((e.key === 'K' || e.key === 'k') && e.ctrlKey) {
      toggleKeyboard();
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

function toggleKeyboard() {
  isKeyboardVisible = !isKeyboardVisible;
  const keyboard = document.getElementById('math-keyboard');
  const helpfulArea = document.getElementById('helpful-area');

  if (isKeyboardVisible) {
    keyboard.classList.remove('keyboard-hidden');
    keyboard.classList.add('keyboard-visible');
    helpfulArea.style.display = 'block';
  } else {
    keyboard.classList.remove('keyboard-visible');
    keyboard.classList.add('keyboard-hidden');
    helpfulArea.style.display = 'none';
  }
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