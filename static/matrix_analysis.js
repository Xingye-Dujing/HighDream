let data_copy = {};

// 当文档加载完成后执行
document.addEventListener('DOMContentLoaded', function () {
  // 获取表达式
  const expressionElement = document.querySelector('.matrix-preview p');

  if (expressionElement) {
    let expressionText = expressionElement.textContent;
    data_copy['expression'] = expressionText;

    if (expressionText) {
      analyzeMatrix(expressionText);
    }
  }
  renderMathPreview();

  // 设置事件监听器
  setupEventListeners();
});

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

      // 添加容器到 math-output
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

        // 重新渲染MathJax
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

  // 复制内容到全屏显示区域, 但保留原始内容
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

// 设置事件监听器
function setupEventListeners() {
  // 运行 Ref
  const runRefElement = document.querySelector('.run-ref');
  if (runRefElement) {
    runRefElement.addEventListener("click", function () {
      runRef();
      runRefElement.style.display = 'none';
    });
  }

  // 运行 Det
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

  // 运行 Rank
  const runRankElement = document.querySelector('.run-rank');
  if (runRankElement) {
    runRankElement.addEventListener("click", function () {
      runRank();
      runRankElement.style.display = 'none';
    });
  }

  // 运行 Eigen
  const runEigenElement = document.querySelector('.run-eigen');
  if (runEigenElement) {
    runEigenElement.addEventListener("click", function () {
      runEigen();
      runEigenElement.style.display = 'none';
    });
  }

  // 运行 SVD
  const runSvdElement = document.querySelector('.run-svd');
  if (runSvdElement) {
    runSvdElement.addEventListener("click", function () {
      runSVD();
      runSvdElement.style.display = 'none';
    });
  }

  // 关闭全屏按钮
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
}

// 分析矩阵函数
function analyzeMatrix(expression) {
  // 显示加载状态
  showLoadingState();

  // 构建请求数据
  const requestData = {
    expression: expression,
    analysis_types: ['rank', 'determinant', 'eigenvalues', 'eigenvectors', 'svd']
  };

  fetch('/matrix_analyze', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(requestData)
  })
    .then(response => response.json())
    .then(data => {
      data_copy = data;
      data_copy['expression'] = expression;
      // 更新页面显示分析结果
      updateAnalysisResults(data);
    })
    .catch(error => {
      console.error('分析矩阵时出错: ', error);
    })
    .finally(() => {
      // 隐藏加载状态
      hideLoadingState();
    });
}

// 显示加载状态
function showLoadingState() {
  // 创建或显示加载指示器
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

    // 添加简单的 CSS 动画
    const style = document.createElement('style');
    style.textContent = `
            .spinner {
                border: 4px solid #f3f3f3;
                border-top: 4px solid #3498db;
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

// 隐藏加载状态
function hideLoadingState() {
  const loadingIndicator = document.getElementById('loading-indicator');
  if (loadingIndicator) {
    loadingIndicator.style.display = 'none';
  }
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

// 更新分析结果显示
function updateAnalysisResults(data) {
  // 更新秩
  if (data.rank !== undefined) {
    const rankElement = document.querySelector('.rank');
    if (rankElement) {
      rankElement.innerHTML = `\\( \\text{rank}(A) = ${data.rank} \\)`;
      // 重新添加控制按钮
      addMathOutputControls();
    }
  }

  // 更新行列式
  if (data.determinant !== undefined) {
    const detElement = document.querySelector('.det');
    if (detElement) {
      detElement.innerHTML = `\\( \\det(A) = ${data.determinant} \\)`;
      // 重新添加控制按钮
      addMathOutputControls();
    }
  }

  // 更新特征值
  if (data.eigenvalues) {
    const eigenValuesElement = document.querySelector('.eigenvalues');
    if (eigenValuesElement && typeof data.eigenvalues == 'object') {
      // 根据返回的数据格式构建矩阵显示
      let matrixLatex = '';
      data.eigenvalues.forEach((value, idx) => {
        matrixLatex += '\\lambda_{' + (idx + 1) + '} = ' + value;
        matrixLatex += idx < data.eigenvalues.length - 1 ? ',\\quad' : '';
      })
      eigenValuesElement.innerHTML = `\\( ${matrixLatex} \\)`
      // 重新添加控制按钮
      addMathOutputControls();
    }
    else if (data.eigenvalues) {
      if (eigenValuesElement) {
        eigenValuesElement.innerHTML = `\\( ${data.eigenvalues} \\)`;
        // 重新添加控制按钮
        addMathOutputControls();
      }
    }
  }

  // 更新特征向量
  if (data.eigenvectors) {
    const eigenVectorsElement = document.querySelector('.eigenvectors');
    if (eigenVectorsElement && typeof data.eigenvectors == 'object') {
      // 根据返回的数据格式构建矩阵显示
      let matrixLatex = '';
      data.eigenvectors.forEach((vector, idx) => {
        matrixLatex += '\\boldsymbol{v}_{' + (idx + 1) + '} = ' + vector;
        matrixLatex += idx < data.eigenvectors.length - 1 ? ',\\quad' : '';
      });
      eigenVectorsElement.innerHTML = `\\( ${matrixLatex} \\)`;
      // 重新添加控制按钮
      addMathOutputControls();
    }
    else if (eigenVectorsElement) {
      eigenVectorsElement.innerHTML = `\\( ${data.eigenvectors} \\)`;
      // 重新添加控制按钮
      addMathOutputControls();
    }
  }

  // 重新渲染 MathJax
  if (window.MathJax) {
    MathJax.typeset();
  }
}

function runRef() {
  if (data_copy.ref) {
    const refElement = document.querySelector('.ref');
    if (refElement) {
      refElement.textContent = data_copy.ref;
      // 重新添加控制按钮
      addMathOutputControls();
    }

    if (window.MathJax) {
      MathJax.typesetPromise([refElement]);
    }
  }
}

function runDet1() {
  fetch("/get_process", {
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
      const detStepElement1 = document.querySelector('.det-steps-1');
      detStepElement1.innerHTML = `\\( ${data.process} \\)`;
      // 重新添加控制按钮
      addMathOutputControls();

      if (window.MathJax) {
        MathJax.typesetPromise([detStepElement1]);
      }
    })
}

function runDet2() {
  fetch("/get_process", {
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
      const detStepElement2 = document.querySelector('.det-steps-2');
      detStepElement2.innerHTML = `\\( ${data.process} \\)`;
      // 重新添加控制按钮
      addMathOutputControls();

      if (window.MathJax) {
        MathJax.typesetPromise([detStepElement2]);
      }
    })
}

function runRank() {
  fetch("/get_process", {
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
      const rankStepElement = document.querySelector('.rank-steps');
      rankStepElement.innerHTML = `\\( ${data.process} \\)`;
      // 添加控制按钮
      addMathOutputControls();

      if (window.MathJax) {
        MathJax.typesetPromise([rankStepElement]);
      }
    })
}

function runEigen() {
  fetch("/get_process", {
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
      const eigenStepElement = document.querySelector('.eigen-steps');
      eigenStepElement.innerHTML = `\\( ${data.process} \\)`;
      // 添加控制按钮
      addMathOutputControls();
      if (window.MathJax) {
        MathJax.typesetPromise([eigenStepElement]);
      }
    })
}

function runSVD() {
  fetch("/get_process", {
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
      const svdStepElement = document.querySelector('.svd-steps');
      svdStepElement.innerHTML = `\\( ${data.process} \\)`;

      const UElement = document.querySelector('.U');
      const SElement = document.querySelector('.S');
      const VElement = document.querySelector('.V');
      if (data.U) {
        UElement.innerHTML = `\\(U = ${data.U} \\)`;
      }
      if (data.S) {
        SElement.innerHTML = `\\(\\Sigma = ${data.S} \\)`;
      }
      if (data.V) {
        VElement.innerHTML = `\\(V = ${data.V} \\)`;
      }

      // 添加控制按钮
      addMathOutputControls();

      if (window.MathJax) {
        MathJax.typesetPromise([svdStepElement, UElement, SElement, VElement]);
      }
    })
}