class BlueprintCanvas {
  constructor() {
    this.nodes = {};
    this.connections = [];
    this.selectedNode = null;
    this.connectingPort = null;
    this.connectingType = null;
    this.nodeCounter = 0;
    this.tempConnection = null;

    this.init();
  }

  init() {
    this.canvas = document.getElementById('blueprint-canvas');
    this.nodesContainer = document.getElementById('nodes-container');
    this.connectionSvg = document.getElementById('connection-svg');

    // Initialize UI elements
    this.setupEventListeners();
    this.setupToolbarButtons();
    this.setupContextMenus();
    this.setupDialogs();
  }

  setupEventListeners() {
    // Click on canvas to deselect nodes and hide context menus
    this.canvas.addEventListener('click', (e) => {
      if (e.target === this.canvas || e.target === this.nodesContainer) {
        this.deselectNode();
        this.hideContextMenu();
      }
    });

    // Click anywhere to hide context menus
    document.addEventListener('click', (e) => {
      if (!e.target.closest('#node-context-menu') && !e.target.closest('#canvas-context-menu')) {
        this.hideContextMenu();
      }
    });

    // Right-click on canvas to show context menu
    this.canvas.addEventListener('contextmenu', (e) => {
      // Check if clicked on empty space (not on any node)
      if (!e.target.closest('.bp-node')) {
        e.preventDefault();
        e.stopPropagation(); // Prevent event bubbling
        this.hideContextMenu(); // Hide any existing context menu first
        this.showCanvasContextMenu(e.clientX, e.clientY);
      }
    });

    // Click on canvas to create temporary connection
    document.addEventListener('click', (e) => {
      if (this.connectingPort) {
        this.handleConnectionClick(e);
      }
    });

    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Delete' && this.selectedNode) {
        this.deleteNode(this.selectedNode.id);
      }
    });
  }

  setupToolbarButtons() {
    document.getElementById('addExpressionNodeBtn').addEventListener('click', () => {
      this.addNode('expression', { x: 100, y: 100 });
    });

    document.getElementById('addOperationNodeBtn').addEventListener('click', () => {
      this.addNode('operation', { x: 200, y: 100 });
    });

    document.getElementById('addResultNodeBtn').addEventListener('click', () => {
      this.addNode('result', { x: 300, y: 100 });
    });

    document.getElementById('addRenderNodeBtn').addEventListener('click', () => {
      this.addNode('render', { x: 400, y: 100 });
    });
  }

  setupContextMenus() {
    // Node context menu
    document.getElementById('node-context-menu').addEventListener('click', (e) => {
      if (e.target.classList.contains('context-menu-item')) {
        const action = e.target.dataset.action;
        if (this.selectedNode) {
          this.handleNodeContextMenuAction(action, this.selectedNode);
        }
        this.hideContextMenu();
      }
    });

    // Canvas context menu
    document.getElementById('canvas-context-menu').addEventListener('click', (e) => {
      if (e.target.classList.contains('context-menu-item')) {
        const action = e.target.dataset.action;
        this.handleCanvasContextMenuAction(action, this.contextMenuX, this.contextMenuY);
        this.hideContextMenu();
      }
    });
  }

  setupDialogs() {
    const dialog = document.getElementById('node-edit-dialog');
    const closeBtn = document.getElementById('closeDialogBtn');
    const cancelBtn = document.getElementById('cancelDialogBtn');
    const saveBtn = document.getElementById('saveDialogBtn');

    closeBtn.addEventListener('click', () => this.closeDialog());
    cancelBtn.addEventListener('click', () => this.closeDialog());

    saveBtn.addEventListener('click', () => {
      this.saveNodeChanges();
      this.closeDialog();
    });

    // Close dialog when clicking outside
    dialog.addEventListener('click', (e) => {
      if (e.target === dialog) {
        this.closeDialog();
      }
    });
  }

  addNode(type, position) {
    this.nodeCounter++;
    const nodeId = `node-${this.nodeCounter}`;

    let inputs = [];
    if (type === 'render') {
      inputs = [{}];
    } else if (type === 'result') {
      // Result node has two inputs: expression and operation
      inputs = [{}, {}];
    }

    const node = {
      id: nodeId,
      type: type,
      x: position.x,
      y: position.y,
      width: 220,
      height: 150,
      title: this.getNodeTitleByType(type),
      inputs: inputs,
      outputs: type === 'render' ? [] : [{}],
      data: this.getDefaultNodeData(type)
    };

    this.nodes[nodeId] = node;
    this.renderNode(node);

    return node;
  }

  getNodeTitleByType(type) {
    const titles = {
      'expression': '表达式节点',
      'operation': '运算节点',
      'result': '结果节点',
      'render': '渲染节点'
    };
    return titles[type] || '节点';
  }

  getDefaultNodeData(type) {
    switch (type) {
      case 'expression':
        return { expression: 'exp(3*x)' };
      case 'operation':
        return { operation: 'diff', variable: 'x', limitPoint: '0' };
      case 'result':
        return { result: '结果将显示在这里' };
      case 'render':
        return { latex: '' };
      default:
        return {};
    }
  }

  renderNode(node) {
    const nodeEl = document.createElement('div');
    nodeEl.className = 'bp-node';
    nodeEl.id = node.id;
    nodeEl.setAttribute('data-node-type', node.type); // Add node type attribute for CSS selector
    nodeEl.style.left = `${node.x}px`;
    nodeEl.style.top = `${node.y}px`;
    nodeEl.style.width = `${node.width}px`;
    nodeEl.style.height = `${node.height}px`;

    nodeEl.innerHTML = this.getNodeHTML(node);

    this.nodesContainer.appendChild(nodeEl);

    // Add event listeners to the node
    this.addNodeEventListeners(nodeEl, node);
  }

  getNodeHTML(node) {
    let content = '';

    switch (node.type) {
      case 'expression':
        content = `
          <textarea class="bp-node-expression" data-node-id="${node.id}">${node.data.expression}</textarea>
        `;
        break;

      case 'operation':
        const isLimit = node.data.operation === 'limit';
        content = `
          <select class="bp-node-operation" data-node-id="${node.id}">
            <option value="diff" ${node.data.operation === 'diff' ? 'selected' : ''}>微分</option>
            <option value="integral" ${node.data.operation === 'integral' ? 'selected' : ''}>积分</option>
            <option value="limit" ${node.data.operation === 'limit' ? 'selected' : ''}>极限</option>
          </select>
          <div class="bp-operation-inputs">
            <input type="text" class="bp-node-variable" data-node-id="${node.id}" value="${node.data.variable || 'x'}" placeholder="变量">
            <input type="text" class="bp-node-limit-point ${isLimit ? '' : 'hidden'}" data-node-id="${node.id}" value="${node.data.limitPoint || '0'}" placeholder="趋向点">
          </div>
        `;
        break;

      case 'result':
        content = `
          <div class="bp-node-result" data-node-id="${node.id}">${node.data.result}</div>
        `;
        break;

      case 'render':
        content = `
          <div class="bp-node-result" data-node-id="${node.id}">${node.data.latex ? this.renderLatex(node.data.latex) : '渲染结果'}</div>
        `;
        break;
    }

    return `
      <div class="bp-node-header">
        <div class="bp-node-title">${node.title}</div>
        <div class="bp-node-type">${this.getNodeTypeLabel(node.type)}</div>
      </div>
      <div class="bp-node-content">
        ${content}
      </div>
      ${this.getInputPortsHTML(node)}
      ${this.getOutputPortsHTML(node)}
      <div class="bp-node-resizer" data-node-id="${node.id}"></div>
    `;
  }

  getNodeTypeLabel(type) {
    const labels = {
      'expression': 'EXPR',
      'operation': 'OP',
      'result': 'RES',
      'render': 'RENDER'
    };
    return labels[type] || type.toUpperCase();
  }

  getInputPortsHTML(node) {
    if (node.inputs.length === 0) return '';

    return node.inputs.map((_, index) => {
      let portClass = 'bp-port input';
      let portLabel = '';

      if (node.type === 'result' && index === 0) {
        portLabel = '<span class="bp-port-label">表达式</span>';
      } else if (node.type === 'result' && index === 1) {
        portLabel = '<span class="bp-port-label">运算</span>';
      }

      return `
        <div class="${portClass}" data-node-id="${node.id}" data-port-type="input" data-port-index="${index}">${portLabel}</div>
      `;
    }).join('');
  }

  getOutputPortsHTML(node) {
    if (node.outputs.length === 0) return '';

    return node.outputs.map((_, index) => `
      <div class="bp-port output" data-node-id="${node.id}" data-port-type="output" data-port-index="${index}"></div>
    `).join('');
  }

  addNodeEventListeners(nodeEl, node) {
    // Node selection
    nodeEl.addEventListener('click', (e) => {
      if (!e.target.classList.contains('bp-port') &&
        !e.target.classList.contains('bp-node-btn') &&
        !e.target.tagName === 'BUTTON' &&
        !e.target.tagName === 'TEXTAREA' &&
        !e.target.tagName === 'SELECT') {
        this.selectNode(node.id);
      }
    });

    // Node context menu (right-click)
    nodeEl.addEventListener('contextmenu', (e) => {
      e.preventDefault();
      e.stopPropagation(); // Prevent event bubbling to canvas
      this.hideContextMenu(); // Hide any existing context menu first
      this.selectNode(node.id); // Select the node first
      this.showNodeContextMenu(e.clientX, e.clientY);
    });

    // Port click for connection
    const ports = nodeEl.querySelectorAll('.bp-port');
    ports.forEach(port => {
      port.addEventListener('click', (e) => {
        e.stopPropagation();
        this.handlePortClick(port, node);
      });
    });

    // Expression input change
    const expressionInput = nodeEl.querySelector('.bp-node-expression');
    if (expressionInput) {
      expressionInput.addEventListener('input', (e) => {
        this.nodes[node.id].data.expression = e.target.value;
        // Auto-update connected nodes
        this.updateConnectedNodes(node.id);
      });
    }

    // Variable input change
    const variableInput = nodeEl.querySelector('.bp-node-variable');
    if (variableInput) {
      variableInput.addEventListener('input', (e) => {
        this.nodes[node.id].data.variable = e.target.value;
        // Auto-update connected nodes
        this.updateConnectedNodes(node.id);
      });
    }

    // Limit point input change
    const limitPointInput = nodeEl.querySelector('.bp-node-limit-point');
    if (limitPointInput) {
      limitPointInput.addEventListener('input', (e) => {
        this.nodes[node.id].data.limitPoint = e.target.value;
        // Auto-update connected nodes
        this.updateConnectedNodes(node.id);
      });
    }

    // Operation selection change
    const operationSelect = nodeEl.querySelector('.bp-node-operation');
    if (operationSelect) {
      operationSelect.addEventListener('change', (e) => {
        this.nodes[node.id].data.operation = e.target.value;

        // Show/hide limit point input based on operation type
        const limitPointInput = nodeEl.querySelector('.bp-node-limit-point');
        if (limitPointInput) {
          if (e.target.value === 'limit') {
            limitPointInput.classList.remove('hidden');
          } else {
            limitPointInput.classList.add('hidden');
          }
        }

        // Auto-update connected nodes
        this.updateConnectedNodes(node.id);
      });
    }

    // Make node draggable (only on header)
    const header = nodeEl.querySelector('.bp-node-header');
    if (header) {
      let isDragging = false;
      let offsetX, offsetY;

      header.addEventListener('mousedown', (e) => {
        if (e.target === header || e.target === header.querySelector('.bp-node-title') || e.target === header.querySelector('.bp-node-type')) {
          isDragging = true;
          this.selectNode(node.id);

          const rect = nodeEl.getBoundingClientRect();
          offsetX = e.clientX - rect.left;
          offsetY = e.clientY - rect.top;

          nodeEl.classList.add('dragging');
        }
      });

      document.addEventListener('mousemove', (e) => {
        if (isDragging) {
          const canvasRect = this.canvas.getBoundingClientRect();
          const x = e.clientX - canvasRect.left - offsetX;
          const y = e.clientY - canvasRect.top - offsetY;

          nodeEl.style.left = `${x}px`;
          nodeEl.style.top = `${y}px`;

          // Update node position in data
          node.x = x;
          node.y = y;

          // Update any connected connections
          this.updateConnectionsForNode(node.id);
        }
      });

      document.addEventListener('mouseup', () => {
        if (isDragging) {
          isDragging = false;
          nodeEl.classList.remove('dragging');
        }
      });
    }

    // Node resizer functionality
    const resizer = nodeEl.querySelector('.bp-node-resizer');
    if (resizer) {
      let isResizing = false;
      let startX, startY, startWidth, startHeight;

      resizer.addEventListener('mousedown', (e) => {
        e.stopPropagation(); // Prevent node selection when resizing
        isResizing = true;
        this.selectNode(node.id); // Keep the node selected during resize

        startX = e.clientX;
        startY = e.clientY;
        startWidth = parseInt(document.defaultView.getComputedStyle(nodeEl).width, 10);
        startHeight = parseInt(document.defaultView.getComputedStyle(nodeEl).height, 10);

        nodeEl.classList.add('resizing');

        e.preventDefault();
      });

      document.addEventListener('mousemove', (e) => {
        if (isResizing) {
          const width = startWidth + (e.clientX - startX);
          const height = startHeight + (e.clientY - startY);

          // Set minimum size constraints
          const newWidth = Math.max(width, 180); // Minimum width from CSS
          const newHeight = Math.max(height, 120); // Minimum height from CSS

          nodeEl.style.width = `${newWidth}px`;
          nodeEl.style.height = `${newHeight}px`;

          // Update node data
          node.width = newWidth;
          node.height = newHeight;

          // Update any connected connections
          this.updateConnectionsForNode(node.id);
        }
      });

      document.addEventListener('mouseup', () => {
        if (isResizing) {
          isResizing = false;
          nodeEl.classList.remove('resizing');
        }
      });
    }
  }

  handlePortClick(port, node) {
    const portType = port.dataset.portType;
    const portIndex = parseInt(port.dataset.portIndex);

    if (!this.connectingPort) {
      // Start connection
      this.connectingPort = { nodeId: node.id, portType, portIndex };
      this.connectingType = portType;
      port.classList.add('connecting');
    } else {
      // Try to complete connection
      if (this.connectingPort.nodeId !== node.id ||
        this.connectingPort.portIndex !== portIndex ||
        this.connectingPort.portType === portType) {
        // Valid connection: input to output or output to input
        if ((this.connectingPort.portType === 'output' && portType === 'input') ||
          (this.connectingPort.portType === 'input' && portType === 'output')) {
          this.createConnection(this.connectingPort, { nodeId: node.id, portType, portIndex });
        }
      }

      // Reset connection state
      const oldPort = document.querySelector(`.bp-port[data-node-id="${this.connectingPort.nodeId}"][data-port-type="${this.connectingPort.portType}"][data-port-index="${this.connectingPort.portIndex}"]`);
      if (oldPort) oldPort.classList.remove('connecting');

      this.connectingPort = null;
      this.connectingType = null;

      if (this.tempConnection) {
        this.connectionSvg.removeChild(this.tempConnection);
        this.tempConnection = null;
      }
    }
  }

  handleConnectionClick(event) {
    if (this.connectingPort) {
      // If clicked outside a valid port, cancel connection
      const clickedOnPort = event.target.classList.contains('bp-port');
      if (!clickedOnPort) {
        const oldPort = document.querySelector(`.bp-port[data-node-id="${this.connectingPort.nodeId}"][data-port-type="${this.connectingPort.portType}"][data-port-index="${this.connectingPort.portIndex}"]`);
        if (oldPort) oldPort.classList.remove('connecting');

        this.connectingPort = null;
        this.connectingType = null;

        if (this.tempConnection) {
          this.connectionSvg.removeChild(this.tempConnection);
          this.tempConnection = null;
        }
      }
    }
  }

  createConnection(outputPort, inputPort) {
    // Ensure valid connection: output to input
    const isOutputToInput = outputPort.portType === 'output' && inputPort.portType === 'input';
    const isInputToOutput = outputPort.portType === 'input' && inputPort.portType === 'output';

    if (!isOutputToInput && !isInputToOutput) {
      return;
    }

    // Normalize to output->input direction
    let finalOutputPort, finalInputPort;
    if (isOutputToInput) {
      finalOutputPort = outputPort;
      finalInputPort = inputPort;
    } else {
      finalOutputPort = inputPort;
      finalInputPort = outputPort;
    }

    // Check if input port already has connection
    const existingConnection = this.connections.find(conn =>
      conn.input.nodeId === finalInputPort.nodeId &&
      conn.input.portIndex === finalInputPort.portIndex
    );

    if (existingConnection) {
      // Delete existing connection
      this.deleteConnection(existingConnection.id);
    }

    // Check if connection already exists
    const connectionExists = this.connections.some(conn =>
      conn.output.nodeId === finalOutputPort.nodeId &&
      conn.output.portIndex === finalOutputPort.portIndex &&
      conn.input.nodeId === finalInputPort.nodeId &&
      conn.input.portIndex === finalInputPort.portIndex
    );

    if (connectionExists) {
      return;
    }

    const connection = {
      id: `conn-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      output: finalOutputPort,
      input: finalInputPort
    };

    this.connections.push(connection);
    this.renderConnection(connection);

    // Auto-trigger calculation
    this.updateConnectedNodes(finalOutputPort.nodeId);

    // If connected to result node, check if all inputs are connected, then execute calculation
    if (finalInputPort.nodeId && this.nodes[finalInputPort.nodeId] && this.nodes[finalInputPort.nodeId].type === 'result') {
      // Check if result node has both inputs connected
      const resultNode = this.nodes[finalInputPort.nodeId];
      const inputConnections = this.connections.filter(conn => conn.input.nodeId === resultNode.id);

      // Only execute calculation when both inputs are connected
      if (inputConnections.length >= 2) {
        const expressionConn = inputConnections.find(conn => conn.input.portIndex === 0);
        const operationConn = inputConnections.find(conn => conn.input.portIndex === 1);

        if (expressionConn && operationConn) {
          this.performResultNodeCalculation(resultNode);
        }
      }
    }
  }

  renderConnection(connection) {
    const outputNode = this.nodes[connection.output.nodeId];
    const inputNode = this.nodes[connection.input.nodeId];

    if (!outputNode || !inputNode) return;

    const outputPort = document.querySelector(
      `.bp-port[data-node-id="${connection.output.nodeId}"][data-port-type="output"][data-port-index="${connection.output.portIndex}"]`
    );
    const inputPort = document.querySelector(
      `.bp-port[data-node-id="${connection.input.nodeId}"][data-port-type="input"][data-port-index="${connection.input.portIndex}"]`
    );

    if (!outputPort || !inputPort) return;

    const outputRect = outputPort.getBoundingClientRect();
    const inputRect = inputPort.getBoundingClientRect();
    const canvasRect = this.connectionSvg.getBoundingClientRect();

    const x1 = outputRect.left - canvasRect.left + outputRect.width / 2;
    const y1 = outputRect.top - canvasRect.top + outputRect.height / 2;
    const x2 = inputRect.left - canvasRect.left + inputRect.width / 2;
    const y2 = inputRect.top - canvasRect.top + inputRect.height / 2;

    const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    path.classList.add('connection-path');
    path.id = connection.id;

    // Create a curved path for the connection
    const midX = (x1 + x2) / 2;
    const pathData = `M ${x1} ${y1} C ${midX} ${y1} ${midX} ${y2} ${x2} ${y2}`;
    path.setAttribute('d', pathData);

    // Add double-click event
    path.addEventListener('dblclick', (e) => {
      e.stopPropagation();
      this.deleteConnection(connection.id);
    });

    this.connectionSvg.appendChild(path);
  }

  updateConnectionsForNode(nodeId) {
    // Find all connections involving this node
    this.connections.forEach(conn => {
      if (conn.output.nodeId === nodeId || conn.input.nodeId === nodeId) {
        const path = document.getElementById(conn.id);
        if (path) {
          const outputNode = this.nodes[conn.output.nodeId];
          const inputNode = this.nodes[conn.input.nodeId];

          if (outputNode && inputNode) {
            const outputPort = document.querySelector(
              `.bp-port[data-node-id="${conn.output.nodeId}"][data-port-type="output"][data-port-index="${conn.output.portIndex}"]`
            );
            const inputPort = document.querySelector(
              `.bp-port[data-node-id="${conn.input.nodeId}"][data-port-type="input"][data-port-index="${conn.input.portIndex}"]`
            );

            if (outputPort && inputPort) {
              const outputRect = outputPort.getBoundingClientRect();
              const inputRect = inputPort.getBoundingClientRect();
              const canvasRect = this.connectionSvg.getBoundingClientRect();

              const x1 = outputRect.left - canvasRect.left + outputRect.width / 2;
              const y1 = outputRect.top - canvasRect.top + outputRect.height / 2;
              const x2 = inputRect.left - canvasRect.left + inputRect.width / 2;
              const y2 = inputRect.top - canvasRect.top + inputRect.height / 2;

              const midX = (x1 + x2) / 2;
              const pathData = `M ${x1} ${y1} C ${midX} ${y1} ${midX} ${y2} ${x2} ${y2}`;
              path.setAttribute('d', pathData);
            }
          }
        }
      }
    });
  }

  selectNode(nodeId) {
    // Deselect previous node
    this.deselectNode();

    const node = this.nodes[nodeId];
    if (node) {
      this.selectedNode = node;

      const nodeEl = document.getElementById(nodeId);
      if (nodeEl) {
        nodeEl.classList.add('selected');
      }
    }
  }

  deselectNode() {
    if (this.selectedNode) {
      const nodeEl = document.getElementById(this.selectedNode.id);
      if (nodeEl) {
        nodeEl.classList.remove('selected');
      }
      this.selectedNode = null;
    }
  }

  deleteNode(nodeId) {
    // Remove node from data
    delete this.nodes[nodeId];

    // Remove node element
    const nodeEl = document.getElementById(nodeId);
    if (nodeEl) {
      nodeEl.remove();
    }

    // Remove connected connections
    this.connections = this.connections.filter(conn => {
      if (conn.output.nodeId === nodeId || conn.input.nodeId === nodeId) {
        // Remove connection from SVG
        const path = document.getElementById(conn.id);
        if (path) {
          path.remove();
        }
        return false;
      }
      return true;
    });

    this.deselectNode();
  }

  async calculateNode(nodeId) {
    const node = this.nodes[nodeId];
    if (!node) return;

    // Only result nodes perform calculation
    if (node.type === 'result') {
      await this.performResultNodeCalculation(node);
    }
  }

  // Call backend API to execute SymPy calculation
  async performSymPyCalculation(expression, operation, variable, limitPoint = '0') {
    try {
      const payload = {
        expression: expression,
        operation_type: operation,
        variable: variable
      };

      // Add limit point parameter for limit operations
      if (operation === 'limit') {
        payload.point = limitPoint;
      }

      const response = await fetch('/api/sympy_calculate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload)
      });

      const data = await response.json();
      if (data.success) {
        return data.result;
      } else {
        console.error('计算错误:', data.error);
        return '错误: ' + data.error;
      }
    } catch (error) {
      console.error('API 请求错误:', error);
      return 'API 请求错误: ' + error.message;
    }
  }

  async updateConnectedNodes(fromNodeId) {
    // Find connections where the fromNodeId is the output
    const connectionsToUpdate = this.connections.filter(conn => conn.output.nodeId === fromNodeId);

    for (const conn of connectionsToUpdate) {
      const targetNode = this.nodes[conn.input.nodeId];
      if (targetNode) {
        if (targetNode.type === 'result') {
          // Trigger result node calculation
          await this.performResultNodeCalculation(targetNode);
        } else if (targetNode.type === 'render') {
          // Render node directly displays input data
          const sourceNode = this.nodes[fromNodeId];
          if (sourceNode) {
            let dataToRender = '';
            if (sourceNode.type === 'expression') {
              dataToRender = sourceNode.data.expression;
            } else if (sourceNode.type === 'operation') {
              dataToRender = sourceNode.data.operation;
            } else if (sourceNode.type === 'result') {
              // Use result expression instead of displaying result
              dataToRender = sourceNode.data.resultExpression || sourceNode.data.result;
            }

            targetNode.data.latex = dataToRender;
            const renderEl = document.querySelector(`.bp-node-result[data-node-id="${targetNode.id}"]`);
            if (renderEl) {
              // Use SymPy's latex function to convert, then render
              this.renderSympyLatex(dataToRender, renderEl);
            }
          }
        }
      }
    }
  }

  renderSympyLatex(expression, element) {
    // Call backend API to use sympy's latex function
    fetch('/api/render_latex', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ expression: expression })
    })
      .then(response => response.json())
      .then(data => {
        if (data.success) {
          element.innerHTML = this.renderLatex(data.latex);
          // Re-render MathJax if available
          if (window.MathJax) {
            window.MathJax.typeset([element]);
          }
        } else {
          element.innerHTML = `错误: ${data.error}`;
        }
      })
      .catch(error => {
        console.error('Error rendering LaTeX:', error);
        element.innerHTML = '渲染错误';
      });
  }

  async performResultNodeCalculation(resultNode) {
    // Get inputs connected to result node
    const inputConnections = this.connections.filter(conn => conn.input.nodeId === resultNode.id);

    if (inputConnections.length < 2) {
      // If insufficient inputs, clear result
      resultNode.data.result = '缺少输入: 需要表达式和运算节点';
      resultNode.data.resultExpression = '';
      const resultEl = document.querySelector(`.bp-node-result[data-node-id="${resultNode.id}"]`);
      if (resultEl) {
        resultEl.textContent = resultNode.data.result;
      }
      return; // Need two inputs
    }

    // First input port (index 0) should connect to expression node
    const expressionConn = inputConnections.find(conn => conn.input.portIndex === 0);
    // Second input port (index 1) should connect to operation node
    const operationConn = inputConnections.find(conn => conn.input.portIndex === 1);

    if (!expressionConn || !operationConn) {
      resultNode.data.result = '缺少输入: 需要表达式和运算节点';
      resultNode.data.resultExpression = '';
      const resultEl = document.querySelector(`.bp-node-result[data-node-id="${resultNode.id}"]`);
      if (resultEl) {
        resultEl.textContent = resultNode.data.result;
      }
      return;
    }

    const expressionSource = this.nodes[expressionConn.output.nodeId];
    const operationSource = this.nodes[operationConn.output.nodeId];

    // Check expression input: can be expression node or result node
    if (!expressionSource || (expressionSource.type !== 'expression' && expressionSource.type !== 'result')) {
      resultNode.data.result = '表达式输入节点类型错误，应为表达式节点或结果节点';
      const resultEl = document.querySelector(`.bp-node-result[data-node-id="${resultNode.id}"]`);
      if (resultEl) {
        resultEl.textContent = resultNode.data.result;
      }
      return;
    }

    // Check operation input: must be operation node
    if (!operationSource || operationSource.type !== 'operation') {
      resultNode.data.result = '运算输入节点类型错误，应为运算节点';
      const resultEl = document.querySelector(`.bp-node-result[data-node-id="${resultNode.id}"]`);
      if (resultEl) {
        resultEl.textContent = resultNode.data.result;
      }
      return;
    }

    // Get expression: if result node, use its calculation result; if expression node, use its expression
    let expression = '';
    if (expressionSource.type === 'expression') {
      expression = expressionSource.data.expression || '';
    } else if (expressionSource.type === 'result') {
      // Use result node's calculation result as expression
      expression = expressionSource.data.resultExpression || expressionSource.data.result || '';
      // If result contains equals sign (like 'd/dx(f(x)) = result'), extract part after equals
      if (expression.includes('=')) {
        const parts = expression.split('=');
        if (parts.length > 1) {
          expression = parts[parts.length - 1].trim();
        }
      }
    }
    const operation = operationSource.data.operation || 'expr';
    const variable = operationSource.data.variable || 'x';

    // Show calculating status
    resultNode.data.result = '计算中...';
    const resultEl = document.querySelector(`.bp-node-result[data-node-id="${resultNode.id}"]`);
    if (resultEl) {
      resultEl.textContent = resultNode.data.result;
    }

    // Execute actual SymPy calculation
    const limitPoint = operationSource.data.limitPoint || '0';
    try {
      const resultExpression = await this.performSymPyCalculation(expression, operation, variable, limitPoint);

      let displayResult = '';
      if (operation === 'diff') {
        displayResult = `d/d${variable}(${expression}) = ${resultExpression}`;
      } else if (operation === 'integral') {
        displayResult = `∫ ${expression} d${variable} = ${resultExpression}`;
      } else if (operation === 'limit') {
        displayResult = `lim (${variable}→${limitPoint}) ${expression} = ${resultExpression}`;
      } else {
        displayResult = resultExpression;
      }

      // Update result node
      resultNode.data.result = displayResult;
      resultNode.data.resultExpression = resultExpression; // Store result expression for output use
      if (resultEl) {
        resultEl.textContent = displayResult;
      }
    } catch (error) {
      console.error('计算错误:', error);
      resultNode.data.result = `计算错误: ${error.message}`;
      if (resultEl) {
        resultEl.textContent = resultNode.data.result;
      }
    }

    // Update render nodes connected to result node
    this.updateConnectedNodes(resultNode.id);
  }

  deleteConnection(connectionId) {
    const connection = this.connections.find(conn => conn.id === connectionId);
    if (!connection) return;

    // Remove connection from array
    this.connections = this.connections.filter(conn => conn.id !== connectionId);

    // Remove connection line from SVG
    const path = document.getElementById(connectionId);
    if (path) {
      path.remove();
    }

    // If deleted connection was connected to result node, check if result node needs update
    if (connection.input.nodeId && this.nodes[connection.input.nodeId] && this.nodes[connection.input.nodeId].type === 'result') {
      // Check if result node still has enough inputs for calculation
      const inputConnections = this.connections.filter(conn => conn.input.nodeId === connection.input.nodeId);
      if (inputConnections.length < 2) {
        // If insufficient inputs, clear result
        const resultNode = this.nodes[connection.input.nodeId];
        if (resultNode) {
          resultNode.data.result = '缺少输入: 需要表达式和运算节点';
          resultNode.data.resultExpression = '';
          const resultEl = document.querySelector(`.bp-node-result[data-node-id="${resultNode.id}"]`);
          if (resultEl) {
            resultEl.textContent = resultNode.data.result;
          }
        }
      } else {
        // If still have enough inputs, re-execute calculation
        this.performResultNodeCalculation(this.nodes[connection.input.nodeId]);
      }
    }
  }

  renderLatex(latex) {
    // Process LaTeX for MathJax rendering
    // Replace common math notations to be MathJax compatible
    let processedLatex = latex;

    // Wrap the expression in MathJax delimiters
    if (!processedLatex.startsWith('\\(') && !processedLatex.startsWith('\\[')) {
      processedLatex = `\\(${processedLatex}\\)`;
    }

    return processedLatex;
  }

  showNodeContextMenu(x, y) {
    this.hideContextMenu(); // Hide any existing context menu first
    const menu = document.getElementById('node-context-menu');
    menu.style.left = `${x}px`;
    menu.style.top = `${y}px`;
    menu.style.display = 'block';
  }

  showCanvasContextMenu(x, y) {
    this.hideContextMenu(); // Hide any existing context menu first
    this.contextMenuX = x;
    this.contextMenuY = y;

    const menu = document.getElementById('canvas-context-menu');
    menu.style.left = `${x}px`;
    menu.style.top = `${y}px`;
    menu.style.display = 'block';
  }

  hideContextMenu() {
    document.getElementById('node-context-menu').style.display = 'none';
    document.getElementById('canvas-context-menu').style.display = 'none';
  }

  handleNodeContextMenuAction(action, node) {
    switch (action) {
      case 'copy':
        this.copyNode(node);
        break;
      case 'delete':
        this.deleteNode(node.id);
        break;
    }
  }

  copyNode(node) {
    this.copiedNode = JSON.parse(JSON.stringify(node));
    this.copiedNode.id = null; // Clear ID for creating new node
  }

  pasteNode(x, y) {
    if (!this.copiedNode) return;

    const newNode = JSON.parse(JSON.stringify(this.copiedNode));
    this.nodeCounter++;
    newNode.id = `node-${this.nodeCounter}`;
    newNode.x = x;
    newNode.y = y;

    this.nodes[newNode.id] = newNode;
    this.renderNode(newNode);

    return newNode;
  }

  handleCanvasContextMenuAction(action, x, y) {
    const canvasRect = this.canvas.getBoundingClientRect();
    const position = {
      x: x - canvasRect.left - 100, // Offset to center of node
      y: y - canvasRect.top - 75
    };

    switch (action) {
      case 'add-expression':
        this.addNode('expression', position);
        break;
      case 'add-operation':
        this.addNode('operation', position);
        break;
      case 'add-result':
        this.addNode('result', position);
        break;
      case 'add-render':
        this.addNode('render', position);
        break;
      case 'paste':
        this.pasteNode(position.x, position.y);
        break;
    }
  }

  openNodeEditDialog(node) {
    document.getElementById('node-title').value = node.title;
    document.getElementById('node-expression').value = node.data.expression || '';

    if (node.type === 'operation') {
      document.getElementById('operation-group').style.display = 'block';
      document.getElementById('node-operation').value = node.data.operation || 'diff';
    } else {
      document.getElementById('operation-group').style.display = 'none';
    }

    this.currentEditNodeId = node.id;
    document.getElementById('node-edit-dialog').classList.add('active');
  }

  saveNodeChanges() {
    if (!this.currentEditNodeId) return;

    const node = this.nodes[this.currentEditNodeId];
    if (!node) return;

    node.title = document.getElementById('node-title').value;
    node.data.expression = document.getElementById('node-expression').value;

    if (node.type === 'operation') {
      node.data.operation = document.getElementById('node-operation').value;
    }

    // Update the node UI
    const nodeEl = document.getElementById(this.currentEditNodeId);
    if (nodeEl) {
      nodeEl.querySelector('.bp-node-title').textContent = node.title;

      const expressionInput = nodeEl.querySelector('.bp-node-expression');
      if (expressionInput) {
        expressionInput.value = node.data.expression;
      }

      const operationSelect = nodeEl.querySelector('.bp-node-operation');
      if (operationSelect) {
        operationSelect.value = node.data.operation;
      }
    }
  }

  closeDialog() {
    document.getElementById('node-edit-dialog').classList.remove('active');
    this.currentEditNodeId = null;
  }
}

// Initialize the canvas when the page loads
document.addEventListener('DOMContentLoaded', () => {
  window.blueprintCanvas = new BlueprintCanvas();
});