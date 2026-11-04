// Escape untrusted text (server error echoes, JS error messages) before innerHTML.
function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

let data_copy = {};

// The universal input stores LaTeX; the analysis endpoints take operator form.
function normalizeUniversal(expression) {
  if (!expression) return Promise.resolve(expression);
  return fetch('/api/convert_form', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ expression: expression }),
  })
    .then((response) => response.json())
    .then((data) => (data && data.success && typeof data.result === 'string' ? data.result : expression))
    .catch(() => expression);
}

function setupUniversalInputForm() {
  const form = document.getElementById('matrix-analysis-form');
  if (!form) return;
  form.addEventListener('submit', function (event) {
    event.preventDefault();
    const textarea = form.querySelector('textarea');
    const value = (textarea && textarea.value.trim()) || '';
    if (/\\(placeholder|square|Box)\b/.test(value)) {
      alert('请先填写矩阵里的占位框');
      return;
    }
    normalizeUniversal(value).then(function (expression) {
      if (!expression) return;
      window.location.href = '/matrix_analysis?expression=' + encodeURIComponent(expression);
    });
  });
}

// Execute when document is loaded
document.addEventListener('DOMContentLoaded', function () {
  // Get expression
  const expressionElement = document.querySelector('.matrix-preview p');

  if (expressionElement) {
    let expressionText = expressionElement.textContent;
    data_copy['expression'] = expressionText;

    if (expressionText) {
      analyzeMatrix(expressionText);
    }
  }
  renderMathPreview();

  // Set event listeners
  setupEventListeners();
});

// Add control buttons for all math-output areas
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

// Set up event listeners
function setupEventListeners() {
  setupUniversalInputForm();
  // Run Ref
  const runRefElement = document.querySelector('.run-ref');
  if (runRefElement) {
    runRefElement.addEventListener("click", function () {
      runRef();
      runRefElement.style.display = 'none';
    });
  }

  // Run Det
  const runDetElement1 = document.querySelector('.run-det-1');
  if (runDetElement1) {
    runDetElement1.addEventListener("click", function () {
      runDet1();
      runDetElement1.style.display = 'none';
    });
  }
  const runDetElement2 = document.querySelector('.run-det-2');
  if (runDetElement2) {
    runDetElement2.addEventListener("click", function () {
      runDet2();
      runDetElement2.style.display = 'none';
    });
  }

  // Run Rank
  const runRankElement = document.querySelector('.run-rank');
  if (runRankElement) {
    runRankElement.addEventListener("click", function () {
      runRank();
      runRankElement.style.display = 'none';
    });
  }

  // Run Eigen
  const runEigenElement = document.querySelector('.run-eigen');
  if (runEigenElement) {
    runEigenElement.addEventListener("click", function () {
      runEigen();
      runEigenElement.style.display = 'none';
    });
  }

  // Run SVD
  const runSvdElement = document.querySelector('.run-svd');
  if (runSvdElement) {
    runSvdElement.addEventListener("click", function () {
      runSVD();
      runSvdElement.style.display = 'none';
    });
  }

  // Close fullscreen button
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

// Matrix analysis function
function analyzeMatrix(expression) {
  // Show loading state
  showLoadingState();

  // Build request data
  const requestData = {
    expression: expression,
    analysis_types: ['rank', 'determinant', 'eigenvalues', 'eigenvectors', 'svd']
  };

  fetch('/api/matrix_analyze', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(requestData)
  })
    .then(response => response.json())
    .then(data => {
      if (!data || data.success === false || data.error) {
        showAnalysisError((data && data.error) || '矩阵分析失败');
        return;
      }
      data_copy = data;
      data_copy['expression'] = expression;
      // Update page display analysis results
      updateAnalysisResults(data);
    })
    .catch(error => {
      console.error('分析矩阵时出错: ', error);
      showAnalysisError('请求失败: ' + error.message);
    })
    .finally(() => {
      // Hide loading state
      hideLoadingState();
    });
}

// Show a user-facing analysis error in every result slot.
function showAnalysisError(message) {
  document.querySelectorAll('.result-item .math-output').forEach(function (el) {
    el.innerHTML = `<span class="preview-error">${escapeHtml(message)}</span>`;
  });
  hideLoadingState();
}

// Show loading state
function showLoadingState() {
  // Create or show loading indicator
  let loadingIndicator = document.getElementById('loading-indicator');
  if (!loadingIndicator) {
    loadingIndicator = document.createElement('div');
    loadingIndicator.id = 'loading-indicator';
    loadingIndicator.innerHTML = '<div class="spinner"></div><p>正在分析矩阵...</p>';
    loadingIndicator.style.cssText = `
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: rgba(255, 255, 255, 0.9);
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            z-index: 1000;
            text-align: center;
        `;

    // Add simple CSS animation
    const style = document.createElement('style');
    style.textContent = `
            .spinner {
                border: 4px solid var(--border-dim);
                border-top: 4px solid var(--text);
                border-radius: 50%;
                width: 30px;
                height: 30px;
                animation: spin 1s linear infinite;
                margin: 0 auto 10px;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
        `;
    document.head.appendChild(style);
    document.body.appendChild(loadingIndicator);
  } else {
    loadingIndicator.style.display = 'block';
  }
}

// Hide loading state
function hideLoadingState() {
  const loadingIndicator = document.getElementById('loading-indicator');
  if (loadingIndicator) {
    loadingIndicator.style.display = 'none';
  }
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

// Update analysis results display
function updateAnalysisResults(data) {
  // Update rank
  if (data.rank !== undefined) {
    const rankElement = document.querySelector('.rank');
    if (rankElement) {
      rankElement.innerHTML = `\\( \\text{rank}(A) = ${escapeHtml(data.rank)} \\)`;
      // Re-add control buttons
      addMathOutputControls();
    }
  }

  // Update determinant
  if (data.determinant !== undefined) {
    const detElement = document.querySelector('.det');
    if (detElement) {
      detElement.innerHTML = `\\( \\det(A) = ${escapeHtml(data.determinant)} \\)`;
      // Re-add control buttons
      addMathOutputControls();
    }
  }

  // Update eigenvalues
  if (data.eigenvalues) {
    const eigenValuesElement = document.querySelector('.eigenvalues');
    if (eigenValuesElement && typeof data.eigenvalues == 'object') {
      // Construct matrix display based on returned data format
      let matrixLatex = '';
      data.eigenvalues.forEach((value, idx) => {
        matrixLatex += '\\lambda_{' + (idx + 1) + '} = ' + escapeHtml(value);
        matrixLatex += idx < data.eigenvalues.length - 1 ? ',\\quad' : '';
      })
      eigenValuesElement.innerHTML = `\\( ${matrixLatex} \\)`
      // Re-add control buttons
      addMathOutputControls();
    }
    else if (data.eigenvalues) {
      if (eigenValuesElement) {
        eigenValuesElement.innerHTML = `\\( ${escapeHtml(data.eigenvalues)} \\)`;
        // Re-add control buttons
        addMathOutputControls();
      }
    }
  }

  // Update eigenvectors
  if (data.eigenvectors) {
    const eigenVectorsElement = document.querySelector('.eigenvectors');
    if (eigenVectorsElement && typeof data.eigenvectors == 'object') {
      // Construct matrix display based on returned data format
      let matrixLatex = '';
      data.eigenvectors.forEach((vector, idx) => {
        matrixLatex += '\\boldsymbol{v}_{' + (idx + 1) + '} = ' + escapeHtml(vector);
        matrixLatex += idx < data.eigenvectors.length - 1 ? ',\\quad' : '';
      });
      eigenVectorsElement.innerHTML = `\\( ${matrixLatex} \\)`;
      // Re-add control buttons
      addMathOutputControls();
    }
    else if (data.eigenvectors) {
      if (eigenVectorsElement) {
        eigenVectorsElement.innerHTML = `\\( ${escapeHtml(data.eigenvectors)} \\)`;
        // Re-add control buttons
        addMathOutputControls();
      }
    }
  }

  // Re-render MathJax
  if (window.MathJax && MathJax.typesetPromise) {
    MathJax.typesetPromise();
  }
}

function runRef() {
  if (!data_copy.ref) {
    showAnalysisError('暂无行阶梯结果，请重新分析矩阵');
    return;
  }
  const refElement = document.querySelector('.ref');
  if (refElement) {
    refElement.innerHTML = '\\[ ' + escapeHtml(data_copy.ref) + ' \\]';
    addMathOutputControls();
    if (window.MathJax && typeof MathJax.typesetPromise === "function") {
      MathJax.typesetPromise([refElement]);
    }
  }
}

function showProcessResult(element, data, label) {
  if (!element) return;
  if (!data || data.success === false) {
    element.innerHTML = `<span class="preview-error">${escapeHtml((data && data.error) || label + '计算失败')}</span>`;
    return;
  }
  element.innerHTML = '\\[ ' + escapeHtml(data.process || '') + ' \\]';
  addMathOutputControls();
  if (window.MathJax && typeof MathJax.typesetPromise === "function") {
    MathJax.typesetPromise([element]);
  }
}

function runDet1() {
  fetch("/api/get_process", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      type: "det_1",
      expression: data_copy.expression,
    }),
  })
    .then((response) => response.json())
    .then((data) => {
      showProcessResult(document.querySelector('.det-steps-1'), data, '行列式');
    })
    .catch((error) => {
      showProcessResult(document.querySelector('.det-steps-1'), { error: error.message }, '行列式');
    });
}

function runDet2() {
  fetch("/api/get_process", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      type: "det_2",
      expression: data_copy.expression,
    }),
  })
    .then((response) => response.json())
    .then((data) => {
      showProcessResult(document.querySelector('.det-steps-2'), data, '行列式');
    })
    .catch((error) => {
      showProcessResult(document.querySelector('.det-steps-2'), { error: error.message }, '行列式');
    });
}

function runRank() {
  fetch("/api/get_process", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      type: "rank",
      expression: data_copy.expression,
    }),
  })
    .then((response) => response.json())
    .then((data) => {
      showProcessResult(document.querySelector('.rank-steps'), data, '秩');
    })
    .catch((error) => {
      showProcessResult(document.querySelector('.rank-steps'), { error: error.message }, '秩');
    });
}

function runEigen() {
  fetch("/api/get_process", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      type: "eigen",
      expression: data_copy.expression,
    }),
  })
    .then((response) => response.json())
    .then((data) => {
      showProcessResult(document.querySelector('.eigen-steps'), data, '特征值');
    })
    .catch((error) => {
      showProcessResult(document.querySelector('.eigen-steps'), { error: error.message }, '特征值');
    });
}

function runSVD() {
  fetch("/api/get_process", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      type: "svd",
      expression: data_copy.expression,
    }),
  })
    .then((response) => response.json())
    .then((data) => {
      if (!data || data.success === false) {
        showAnalysisError((data && data.error) || 'SVD 计算失败');
        return;
      }
      showProcessResult(document.querySelector('.svd-steps'), data, 'SVD');
      const UElement = document.querySelector('.svd.U');
      const SElement = document.querySelector('.svd.S');
      const VElement = document.querySelector('.svd.V');
      if (data.U && UElement) UElement.innerHTML = '\\[ U = ' + escapeHtml(data.U) + ' \\]';
      if (data.S && SElement) SElement.innerHTML = '\\[ \\Sigma = ' + escapeHtml(data.S) + ' \\]';
      if (data.V && VElement) VElement.innerHTML = '\\[ V = ' + escapeHtml(data.V) + ' \\]';
      addMathOutputControls();
      if (window.MathJax && typeof MathJax.typesetPromise === "function") {
        MathJax.typesetPromise([UElement, SElement, VElement].filter(Boolean));
      }
    })
    .catch((error) => {
      showAnalysisError('SVD 请求失败: ' + error.message);
    });
}
