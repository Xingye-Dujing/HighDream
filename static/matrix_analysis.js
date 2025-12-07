let data_copy = {};

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
  if (window.MathJax) {
    MathJax.typesetPromise([content]);
  }
}

// Set up event listeners
function setupEventListeners() {
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
      data_copy = data;
      data_copy['expression'] = expression;
      // Update page display analysis results
      updateAnalysisResults(data);
    })
    .catch(error => {
      console.error('分析矩阵时出错: ', error);
    })
    .finally(() => {
      // Hide loading state
      hideLoadingState();
    });
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

// Hide loading state
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

// Update analysis results display
function updateAnalysisResults(data) {
  // Update rank
  if (data.rank !== undefined) {
    const rankElement = document.querySelector('.rank');
    if (rankElement) {
      rankElement.innerHTML = `\\( \\text{rank}(A) = ${data.rank} \\)`;
      // Re-add control buttons
      addMathOutputControls();
    }
  }

  // Update determinant
  if (data.determinant !== undefined) {
    const detElement = document.querySelector('.det');
    if (detElement) {
      detElement.innerHTML = `\\( \\det(A) = ${data.determinant} \\)`;
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
        matrixLatex += '\\lambda_{' + (idx + 1) + '} = ' + value;
        matrixLatex += idx < data.eigenvalues.length - 1 ? ',\\quad' : '';
      })
      eigenValuesElement.innerHTML = `\\( ${matrixLatex} \\)`
      // Re-add control buttons
      addMathOutputControls();
    }
    else if (data.eigenvalues) {
      if (eigenValuesElement) {
        eigenValuesElement.innerHTML = `\\( ${data.eigenvalues} \\)`;
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
        matrixLatex += '\\boldsymbol{v}_{' + (idx + 1) + '} = ' + vector;
        matrixLatex += idx < data.eigenvectors.length - 1 ? ',\\quad' : '';
      });
      eigenVectorsElement.innerHTML = `\\( ${matrixLatex} \\)`;
      // Re-add control buttons
      addMathOutputControls();
    }
    else if (data.eigenvectors) {
      if (eigenVectorsElement) {
        eigenVectorsElement.innerHTML = `\\( ${data.eigenvectors} \\)`;
        // Re-add control buttons
        addMathOutputControls();
      }
    }
  }

  // Re-render MathJax
  if (window.MathJax) {
    MathJax.typeset();
  }
}

function runRef() {
  if (data_copy.ref) {
    const refElement = document.querySelector('.ref');
    if (refElement) {
      refElement.textContent = data_copy.ref;
      // Re-add control buttons
      addMathOutputControls();
    }

    if (window.MathJax) {
      MathJax.typesetPromise([refElement]);
    }
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
      const detStepElement1 = document.querySelector('.det-steps-1');
      detStepElement1.innerHTML = `\\( ${data.process} \\)`;
      // Re-add control buttons
      addMathOutputControls();

      if (window.MathJax) {
        MathJax.typesetPromise([detStepElement1]);
      }
    })
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
      const detStepElement2 = document.querySelector('.det-steps-2');
      detStepElement2.innerHTML = `\\( ${data.process} \\)`;
      // Re-add control buttons
      addMathOutputControls();

      if (window.MathJax) {
        MathJax.typesetPromise([detStepElement2]);
      }
    })
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
      const rankStepElement = document.querySelector('.rank-steps');
      rankStepElement.innerHTML = `\\( ${data.process} \\)`;
      // Add control buttons
      addMathOutputControls();

      if (window.MathJax) {
        MathJax.typesetPromise([rankStepElement]);
      }
    })
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
      const eigenStepElement = document.querySelector('.eigen-steps');
      eigenStepElement.innerHTML = `\\( ${data.process} \\)`;
      // Add control buttons
      addMathOutputControls();
      if (window.MathJax) {
        MathJax.typesetPromise([eigenStepElement]);
      }
    })
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

      // Add control buttons
      addMathOutputControls();

      if (window.MathJax) {
        MathJax.typesetPromise([svdStepElement, UElement, SElement, VElement]);
      }
    })
}