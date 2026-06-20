// Method-tree page: blueprint-style interactive tree (ports + SVG bezier paths).

(function () {
  'use strict';

  var CARD_WIDTH = 240;
  var BODY_MIN_HEIGHT = 50;
  var HEADER_HEIGHT = 36;
  var FOOTER_HEIGHT = 26;
  var H_GAP = 30;
  var V_GAP = 100;
  var MIN_ZOOM = 0.15;
  var MAX_ZOOM = 2.0;
  var POLL_INTERVAL_MS = 400;
  var MATHJAX_WAIT_MS = 15000;

  // DOM handles
  var elDomain, elExpr, elVar, elPoint, elDir, elDepth, elNodes, elTime;
  var elStart, elCancel, elError;
  var elViewport, elWorld, elEdges, elNodesContainer, elLoading, elLoadingText, elStats;
  var limitFields;

  // Tree state
  var currentTaskId = null;
  var pollTimer = null;
  var renderedNodeIds = new Set();
  var renderedTree = null;

  // User-mutable structure (initialized from backend tree; then user can rewire).
  var userChildren = {};  // nodeId -> [childNodeIds]
  var userPositions = {}; // nodeId -> { x, y, cx, height }
  var nodeData = {};      // nodeId -> backend node payload

  // Pan / zoom
  var panX = 0, panY = 0, scale = 1;
  var isPanning = false;
  var panStartX = 0, panStartY = 0, panOriginX = 0, panOriginY = 0;

  // Node dragging
  var dragState = null; // { id, offsetX, offsetY }
  var resizeState = null; // { id, startWidth, startHeight, startClientX, startClientY }

  // Port connecting
  var connectingPort = null; // { id, type: 'output'|'input' }
  var tempPath = null;

  // MathJax
  var mathJaxReady = false;

  // ----------------------------------------------------------------- bootstrap

  document.addEventListener('DOMContentLoaded', function () {
    elDomain = document.getElementById('mt-domain');
    elExpr = document.getElementById('mt-expr');
    elVar = document.getElementById('mt-var');
    elPoint = document.getElementById('mt-point');
    elDir = document.getElementById('mt-dir');
    elDepth = document.getElementById('mt-depth');
    elNodes = document.getElementById('mt-max-nodes');
    elTime = document.getElementById('mt-time');
    elStart = document.getElementById('mt-start-btn');
    elCancel = document.getElementById('mt-cancel-btn');
    elError = document.getElementById('mt-error');
    elViewport = document.getElementById('mt-viewport');
    elWorld = document.getElementById('mt-world');
    elEdges = document.getElementById('mt-edges');
    elNodesContainer = document.getElementById('mt-nodes');
    elLoading = document.getElementById('mt-loading');
    elLoadingText = document.getElementById('mt-loading-text');
    elStats = document.getElementById('mt-stats');
    limitFields = document.querySelectorAll('.mt-limit-only');

    readQueryParams();
    toggleLimitFields();

    elDomain.addEventListener('change', toggleLimitFields);
    elStart.addEventListener('click', startEnumeration);
    elCancel.addEventListener('click', cancelEnumeration);
    document.getElementById('mt-fit-btn').addEventListener('click', fitToScreen);
    document.getElementById('mt-home-btn').addEventListener('click', goToRoot);
    window.addEventListener('resize', function () {
      if (renderedTree) renderEdges();
    });

    setupPanZoom();
    setupPortConnecting();
    waitForMathJax();

    var qs = new URLSearchParams(window.location.search);
    if (qs.get('expression') && qs.get('domain')) {
      setTimeout(startEnumeration, 200);
    }
  });

  function waitForMathJax() {
    var start = Date.now();
    (function poll() {
      if (window.MathJax && typeof MathJax.typesetPromise === 'function') {
        mathJaxReady = true;
        var sweep = function () {
          if (elNodesContainer && elNodesContainer.children.length > 0) {
            try { MathJax.typesetPromise([elNodesContainer]).catch(function () {}); }
            catch (e) { /* ignore */ }
          }
        };
        if (MathJax.startup && MathJax.startup.promise) {
          MathJax.startup.promise.then(sweep);
        } else {
          setTimeout(sweep, 200);
        }
        return;
      }
      if (Date.now() - start < MATHJAX_WAIT_MS) setTimeout(poll, 150);
    })();
    // Belt-and-braces fallback.
    setTimeout(function () {
      if (window.MathJax && typeof MathJax.typesetPromise === 'function') {
        mathJaxReady = true;
        if (elNodesContainer && elNodesContainer.children.length > 0) {
          try { MathJax.typesetPromise([elNodesContainer]).catch(function () {}); }
          catch (e) { /* ignore */ }
        }
      }
    }, 5000);
  }

  function readQueryParams() {
    var qs = new URLSearchParams(window.location.search);
    if (qs.get('domain')) elDomain.value = qs.get('domain');
    if (qs.get('expression')) elExpr.value = qs.get('expression');
    if (qs.get('variable')) elVar.value = qs.get('variable');
    if (qs.get('point')) elPoint.value = qs.get('point');
    if (qs.get('direction')) elDir.value = qs.get('direction');
    if (qs.get('max_depth')) elDepth.value = qs.get('max_depth');
    if (qs.get('max_nodes')) elNodes.value = qs.get('max_nodes');
    if (qs.get('time_limit_seconds')) elTime.value = qs.get('time_limit_seconds');
  }

  function toggleLimitFields() {
    var show = elDomain.value === 'limit';
    for (var i = 0; i < limitFields.length; i++) {
      limitFields[i].style.display = show ? '' : 'none';
    }
  }

  // ------------------------------------------------------------------- polling

  function startEnumeration() {
    hideError();
    if (!elExpr.value.trim()) { showError('请输入表达式'); return; }
    var body = {
      domain: elDomain.value,
      expression: elExpr.value,
      variable: elVar.value || 'x',
      point: elPoint.value || '0',
      direction: elDir.value || '+',
      max_depth: parseInt(elDepth.value, 10) || 8,
      max_nodes: parseInt(elNodes.value, 10) || 500,
      time_limit_seconds: parseInt(elTime.value, 10) || 30,
    };
    setUiBusy(true);
    resetCanvas();
    fetch('/api/method_tree_start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
      .then(function (r) { return r.json(); })
      .then(function (payload) {
        if (!payload.success) {
          showError(payload.error || '启动失败');
          setUiBusy(false);
          return;
        }
        currentTaskId = payload.task_id;
        openOverlay('枚举中…');
        pollTimer = setInterval(pollOnce, POLL_INTERVAL_MS);
      })
      .catch(function (err) { showError('网络错误: ' + err); setUiBusy(false); });
  }

  function cancelEnumeration() {
    if (!currentTaskId) return;
    fetch('/api/method_tree_cancel', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ task_id: currentTaskId }),
    }).catch(function () {});
  }

  function pollOnce() {
    if (!currentTaskId) return;
    fetch('/api/method_tree_status', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ task_id: currentTaskId }),
    })
      .then(function (r) { return r.json(); })
      .then(function (payload) {
        if (payload.tree && !window.__mtFirstPayloadLogged) {
          window.__mtFirstPayloadLogged = true;
          console.log('[mt] FIRST tree payload:', JSON.stringify(payload.tree, null, 2));
        }
        handlePoll(payload);
      })
      .catch(function (err) { showError('轮询失败: ' + err); });
  }

  function handlePoll(payload) {
    if (!payload.success) {
      stopPolling(); setUiBusy(false);
      showError(payload.error || '服务端错误');
      return;
    }
    var status = payload.task && payload.task.status;
    var hasTree = !!(payload.tree && payload.tree.nodes);
    var nodeCount = hasTree ? Object.keys(payload.tree.nodes).length : 0;
    console.log('[mt] poll status=' + status +
                ' tree=' + hasTree + ' nodes=' + nodeCount +
                ' rendered=' + renderedNodeIds.size);

    if (hasTree && treeHasNewNodes(payload.tree)) {
      console.log('[mt] new nodes detected, rendering');
      renderedTree = payload.tree;
      ingestTree(renderedTree);
      console.log('[mt] ingest done, nodeData keys=', Object.keys(nodeData));
      layoutTree();
      console.log('[mt] layout done, positions=', Object.keys(userPositions));
      renderTree();
      console.log('[mt] renderTree done, mt-nodes children=',
                  elNodesContainer.children.length);
      renderEdges();
      if (renderedNodeIds.size <= 1) goToRoot();
    } else if (hasTree) {
      console.log('[mt] tree present but no new nodes');
    }
    if (payload.tree) updateOverlay(payload.tree.stats);
    if (status === 'completed' || status === 'failed') {
      stopPolling(); closeOverlay(); setUiBusy(false);
      if (payload.tree) {
        renderedTree = payload.tree;
        ingestTree(renderedTree);
        layoutTree();
        renderTree();
        renderEdges();
        renderStats(payload.tree.stats);
        fitToScreen();
        console.log('[mt] FINAL render, mt-nodes children=',
                    elNodesContainer.children.length);
      } else if (payload.task && payload.task.error) {
        showError(payload.task.error);
      }
    }
  }

  function stopPolling() {
    if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
    currentTaskId = null;
  }

  function treeHasNewNodes(tree) {
    if (!tree || !tree.nodes) return false;
    var ids = Object.keys(tree.nodes);
    for (var i = 0; i < ids.length; i++) {
      if (!renderedNodeIds.has(ids[i])) return true;
    }
    return false;
  }

  function setUiBusy(busy) {
    elStart.disabled = busy;
    elCancel.style.display = busy ? '' : 'none';
  }

  function showError(msg) { elError.textContent = msg; elError.style.display = ''; }
  function hideError() { elError.style.display = 'none'; elError.textContent = ''; }
  function openOverlay(text) { elLoadingText.textContent = text; elLoading.style.display = ''; }
  function updateOverlay(stats) {
    if (!stats) return;
    elLoadingText.textContent =
      '枚举中… ' + stats.node_count + ' 个节点, 深度 ' +
      stats.max_depth_seen + ', 用时 ' + stats.elapsed_seconds.toFixed(1) + ' 秒';
  }
  function closeOverlay() { elLoading.style.display = 'none'; }

  function renderStats(stats) {
    if (!stats) { elStats.innerHTML = ''; return; }
    var reasonMap = {
      completed: '完成', depth_limit: '达到深度上限',
      node_limit: '达到节点上限', time_limit: '达到时间上限',
      cancelled: '已取消', error: '出错',
    };
    var label = reasonMap[stats.reason] || stats.reason;
    var cls = 'mt-reason' + (stats.truncated ? ' mt-truncated' : '') +
              (stats.error ? ' mt-error' : '');
    elStats.innerHTML =
      '<div class="mt-stat-line"><span class="' + cls + '">' + label + '</span></div>' +
      '<div class="mt-stat-line">节点 ' + stats.node_count +
        ' · 最大深度 ' + stats.max_depth_seen +
        ' · 用时 ' + stats.elapsed_seconds.toFixed(2) + ' 秒</div>' +
      (stats.error ? '<div class="mt-stat-line mt-reason mt-error">' +
                     escapeHtml(stats.error) + '</div>' : '');
  }

  function resetCanvas() {
    renderedNodeIds = new Set();
    renderedTree = null;
    userChildren = {};
    userPositions = {};
    nodeData = {};
    elNodesContainer.innerHTML = '';
    clearEdgesSvg();
    elStats.innerHTML = '';
    panX = 0; panY = 0; scale = 1;
    applyTransform();
  }

  // ------------------------------------------------------------ ingest / layout

  /** Copy backend tree into the user-mutable structure. Preserves any edits. */
  function ingestTree(tree) {
    var ids = Object.keys(tree.nodes);
    for (var i = 0; i < ids.length; i++) {
      nodeData[ids[i]] = tree.nodes[ids[i]];
    }

    // Rebuild userChildren from the authoritative `parent` field on each node.
    // The backend's `children` map is often incomplete in incremental
    // snapshots (early polls only include the root), so we must NOT rely on it.
    var newUserChildren = {};
    ids.forEach(function (id) { newUserChildren[id] = []; });
    ids.forEach(function (id) {
      var parent = tree.nodes[id].parent;
      if (parent && newUserChildren[parent]) {
        newUserChildren[parent].push(id);
      }
    });

    // Merge in any user-added edges (from interactive rewiring) that still
    // reference nodes present in the current tree.
    Object.keys(userChildren).forEach(function (pid) {
      if (!newUserChildren[pid]) return;
      (userChildren[pid] || []).forEach(function (cid) {
        if (!newUserChildren[cid]) return;
        // Skip if the backend already established this edge via `parent`.
        if (tree.nodes[cid] && tree.nodes[cid].parent === pid) return;
        if (newUserChildren[pid].indexOf(cid) < 0) {
          newUserChildren[pid].push(cid);
        }
      });
    });

    userChildren = newUserChildren;

    // Diagnostic exposure.
    window.__debug_userChildren = userChildren;
    window.__debug_userPositions = userPositions;
    window.__debug_nodeData = nodeData;
    console.log('[mt] ingest: userChildren =', JSON.stringify(userChildren));
  }

  function layoutTree() {
    // Compute depth via BFS from every root (node with no incoming edge).
    var incoming = {};
    Object.keys(userChildren).forEach(function (pid) {
      (userChildren[pid] || []).forEach(function (cid) { incoming[cid] = pid; });
    });
    var roots = Object.keys(nodeData).filter(function (id) { return !incoming[id]; });
    if (roots.length === 0 && Object.keys(nodeData).length > 0) {
      roots = [Object.keys(nodeData)[0]];
    }

    var depthOf = {};
    var subtreeWidth = {};

    function widthOf(id, visited) {
      if (subtreeWidth[id] !== undefined) return subtreeWidth[id];
      if (!visited) visited = new Set();
      if (visited.has(id)) { subtreeWidth[id] = CARD_WIDTH; return CARD_WIDTH; }
      visited.add(id);
      var kids = (userChildren[id] || []).filter(function (k) { return nodeData[k]; });
      if (kids.length === 0) { subtreeWidth[id] = CARD_WIDTH; return CARD_WIDTH; }
      var total = 0;
      for (var i = 0; i < kids.length; i++) total += widthOf(kids[i], visited);
      total += H_GAP * (kids.length - 1);
      subtreeWidth[id] = Math.max(CARD_WIDTH, total);
      return subtreeWidth[id];
    }

    roots.forEach(function (r) { widthOf(r, new Set()); });

    var positions = {};
    function assign(id, left, depth, visited) {
      if (!visited) visited = new Set();
      if (visited.has(id)) return;
      visited.add(id);
      var nodeW = (userPositions[id] && userPositions[id].width) || CARD_WIDTH;
      var nodeH = (userPositions[id] && userPositions[id].height) ||
                  estimateHeight(nodeData[id]);
      var w = subtreeWidth[id] || nodeW;
      var cx = left + w / 2;
      var y = depth * (HEADER_HEIGHT + BODY_MIN_HEIGHT + V_GAP) + 40;
      positions[id] = {
        x: cx - nodeW / 2, y: y, cx: cx,
        width: nodeW, height: nodeH, depth: depth,
      };
      depthOf[id] = depth;
      var kids = (userChildren[id] || []).filter(function (k) { return nodeData[k]; });
      var cursor = left + (w - sumKidsWidth(kids)) / 2;
      for (var i = 0; i < kids.length; i++) {
        var kw = subtreeWidth[kids[i]] || CARD_WIDTH;
        assign(kids[i], cursor, depth + 1, visited);
        cursor += kw + H_GAP;
      }
    }

    function sumKidsWidth(kids) {
      var s = 0;
      for (var i = 0; i < kids.length; i++) s += (subtreeWidth[kids[i]] || CARD_WIDTH);
      if (kids.length > 1) s += H_GAP * (kids.length - 1);
      return s;
    }

    var cursor = 40;
    roots.forEach(function (r) {
      assign(r, cursor, 0, new Set());
      cursor += (subtreeWidth[r] || CARD_WIDTH) + H_GAP * 2;
    });

    userPositions = positions;
  }

  function estimateHeight(node) {
    var h = HEADER_HEIGHT + BODY_MIN_HEIGHT;
    if (node && (node.done || node.truncated)) h += FOOTER_HEIGHT;
    return h;
  }

  // ------------------------------------------------------------------ render

  function renderTree() {
    var ids = Object.keys(nodeData);
    console.log('[mt] renderTree: nodeData ids =', ids,
                ' userPositions ids =', Object.keys(userPositions));
    var newIds = [];
    for (var i = 0; i < ids.length; i++) {
      if (!renderedNodeIds.has(ids[i])) newIds.push(ids[i]);
    }
    console.log('[mt] renderTree: newIds to render =', newIds);

    for (var j = 0; j < newIds.length; j++) {
      var id = newIds[j];
      var pos = userPositions[id];
      if (!pos) {
        console.warn('[mt] node ' + id + ' has no position — skipped');
        continue;
      }
      createNodeEl(nodeData[id], pos);
      renderedNodeIds.add(id);
    }
    console.log('[mt] renderTree: mt-nodes now has ' +
                elNodesContainer.children.length + ' children');

    ids.forEach(function (id) {
      var node = nodeData[id];
      var el = document.getElementById('mt-node-' + id);
      if (!el) return;
      var pos = userPositions[id];
      if (pos) {
        el.style.left = pos.x + 'px';
        el.style.top = pos.y + 'px';
      }
      el.classList.toggle('mt-done', !!node.done);
      el.classList.toggle('mt-truncated', !!node.truncated);
      updateFooter(el, node);
    });

    if (newIds.length > 0) {
      var newEls = newIds.map(function (id) {
        return document.getElementById('mt-node-' + id);
      }).filter(Boolean);
      safeTypeset(newEls);
    }
  }

  function createNodeEl(node, pos) {
    var el = document.createElement('div');
    el.className = 'mt-node';
    if (!node.parent && Object.keys(nodeData).length > 0 &&
        Object.values(nodeData).every(function (n) { return n.parent !== node.id; })) {
      // Mark original root only on first render.
    }
    if (!node.parent) el.className += ' mt-root';
    if (node.done) el.className += ' mt-done';
    if (node.truncated) el.className += ' mt-truncated';
    el.id = 'mt-node-' + node.id;
    el.style.left = pos.x + 'px';
    el.style.top = pos.y + 'px';
    el.style.width = (pos.width || CARD_WIDTH) + 'px';
    if (pos.height) el.style.height = pos.height + 'px';

    // Input port (top-center).
    var inPort = document.createElement('div');
    inPort.className = 'mt-port mt-port-input';
    inPort.dataset.nodeId = node.id;
    inPort.dataset.portType = 'input';
    el.appendChild(inPort);

    // Output port (bottom-center).
    var outPort = document.createElement('div');
    outPort.className = 'mt-port mt-port-output';
    outPort.dataset.nodeId = node.id;
    outPort.dataset.portType = 'output';
    el.appendChild(outPort);

    // Header (rule name + type badge) — blueprint-style.
    var header = document.createElement('div');
    header.className = 'mt-node-header';

    var title = document.createElement('div');
    title.className = 'mt-node-title';
    title.textContent = node.rule_display || (node.parent ? (node.rule_applied || '规则') : '起点');
    header.appendChild(title);

    var typeBadge = document.createElement('div');
    typeBadge.className = 'mt-node-type';
    typeBadge.textContent = node.parent ? 'RULE' : 'ROOT';
    header.appendChild(typeBadge);

    el.appendChild(header);

    // Content — MathJax-rendered LaTeX. Use \(...\) which MathJax v3 enables
    // by default as an inline-math delimiter (more robust than $...$).
    // For non-root nodes, prefer top_latex which shows the whole expression
    // after the rule is applied (not just the sub-expression being worked on).
    var content = document.createElement('div');
    content.className = 'mt-node-content';
    var latex = (node.parent ? (node.top_latex || node.latex) : node.latex) || '';
    content.innerHTML = latex ? ('\\(' + latex + '\\)') : '<em>(空)</em>';
    el.appendChild(content);

    updateFooter(el, node);

    // Drag handle: the header.
    attachDragHandler(el, header, node.id);

    // Resize handle: bottom-right dot (blueprint_canvas style).
    var resizer = document.createElement('div');
    resizer.className = 'mt-node-resizer';
    el.appendChild(resizer);
    attachResizeHandler(el, resizer, node.id);

    elNodesContainer.appendChild(el);
  }

  function updateFooter(el, node) {
    var existing = el.querySelector('.mt-node-footer');
    var wantFinal = node && node.done && node.final_latex;
    var wantTrunc = node && node.truncated && !wantFinal;
    if (wantFinal) {
      if (!existing || !existing.classList.contains('mt-final')) {
        if (existing) existing.remove();
        var f = document.createElement('div');
        f.className = 'mt-node-footer mt-final';
        f.textContent = '✓ 最终结果';
        el.appendChild(f);
      }
    } else if (wantTrunc) {
      if (!existing || !existing.classList.contains('mt-trunc')) {
        if (existing) existing.remove();
        var t = document.createElement('div');
        t.className = 'mt-node-footer mt-trunc';
        t.textContent = '· 搜索在此截止';
        el.appendChild(t);
      }
    }
  }

  function safeTypeset(elements) {
    if (!elements || elements.length === 0) return;
    console.log('[mt] safeTypeset called with', elements.length,
                'elements, mathJaxReady=' + mathJaxReady);
    if (mathJaxReady && window.MathJax && MathJax.typesetPromise) {
      MathJax.typesetPromise(elements).then(function () {
        var mjxCount = 0;
        elements.forEach(function (el) {
          mjxCount += el.querySelectorAll('mjx-container').length;
        });
        console.log('[mt] typesetPromise resolved; mjx-containers created =', mjxCount);
      }).catch(function (err) {
        console.warn('[mt] typesetPromise rejected:', err);
        setTimeout(function () {
          MathJax.typesetPromise(elements).catch(function () {});
        }, 500);
      });
      return;
    }
    var waited = 0, step = 150;
    var timer = setInterval(function () {
      waited += step;
      if (window.MathJax && MathJax.typesetPromise) {
        clearInterval(timer); mathJaxReady = true;
        console.log('[mt] MathJax became ready after', waited, 'ms; typesetting now');
        MathJax.typesetPromise(elements).then(function () {
          var mjxCount = 0;
          elements.forEach(function (el) {
            mjxCount += el.querySelectorAll('mjx-container').length;
          });
          console.log('[mt] (delayed) typeset resolved; mjx-containers =', mjxCount);
        }).catch(function (err) {
          console.warn('[mt] (delayed) typeset rejected:', err);
        });
      } else if (waited > MATHJAX_WAIT_MS) {
        clearInterval(timer);
        console.warn('[mt] MathJax never became ready after', waited, 'ms');
      }
    }, step);
  }

  // ----------------------------------------------------- SVG bezier edges

  function clearEdgesSvg() {
    // Keep the <defs> (arrowhead marker) intact.
    var defs = elEdges.querySelector('defs');
    elEdges.innerHTML = '';
    if (defs) elEdges.appendChild(defs);
  }

  function renderEdges() {
    clearEdgesSvg();

    var maxX = 0, maxY = 0;
    Object.keys(userPositions).forEach(function (id) {
      var p = userPositions[id];
      var node = nodeData[id];
      var h = node ? estimateHeight(node) : (HEADER_HEIGHT + BODY_MIN_HEIGHT);
      maxX = Math.max(maxX, p.x + CARD_WIDTH + H_GAP);
      maxY = Math.max(maxY, p.y + h + V_GAP);
    });
    elEdges.setAttribute('width', String(maxX || 1));
    elEdges.setAttribute('height', String(maxY || 1));
    elEdges.style.width = (maxX || 1) + 'px';
    elEdges.style.height = (maxY || 1) + 'px';

    var pathCount = 0;
    console.log('[mt] renderEdges: userChildren =',
                JSON.stringify(userChildren));
    Object.keys(userChildren).forEach(function (pid) {
      (userChildren[pid] || []).forEach(function (cid) {
        var parentPos = userPositions[pid];
        var childPos = userPositions[cid];
        if (!parentPos || !childPos) {
          console.warn('[mt] renderEdges: missing pos for', pid, cid);
          return;
        }
        var parentNode = nodeData[pid];
        var parentH = parentNode ? estimateHeight(parentNode) : (HEADER_HEIGHT + BODY_MIN_HEIGHT);

        var x1 = parentPos.cx;
        var y1 = parentPos.y + parentH;
        var x2 = childPos.cx;
        var y2 = childPos.y;
        var midY = (y1 + y2) / 2;

        var path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        path.setAttribute('class', 'mt-connection-path');
        path.setAttribute('d',
          'M ' + x1 + ' ' + y1 +
          ' C ' + x1 + ' ' + midY + ', ' +
                x2 + ' ' + midY + ', ' +
                x2 + ' ' + y2);
        path.setAttribute('marker-end', 'url(#mt-arrowhead)');
        path.dataset.parent = pid;
        path.dataset.child = cid;
        path.addEventListener('dblclick', function (e) {
          e.stopPropagation();
          deleteConnection(pid, cid);
        });
        elEdges.appendChild(path);
        pathCount++;
      });
    });
    console.log('[mt] renderEdges: created', pathCount, 'paths, svg children now =',
                elEdges.childElementCount);
  }

  function deleteConnection(parentId, childId) {
    var kids = userChildren[parentId] || [];
    var idx = kids.indexOf(childId);
    if (idx >= 0) kids.splice(idx, 1);
    userChildren[parentId] = kids;
    layoutTree();
    // Reposition every node (structure changed).
    Object.keys(userPositions).forEach(function (id) {
      var el = document.getElementById('mt-node-' + id);
      var p = userPositions[id];
      if (el && p) {
        el.style.left = p.x + 'px';
        el.style.top = p.y + 'px';
      }
    });
    renderEdges();
  }

  function createConnection(parentId, childId) {
    // Prevent cycles: if childId is an ancestor of parentId, refuse.
    if (isAncestor(childId, parentId)) return;
    if (parentId === childId) return;
    // Remove any existing incoming edge to childId (each node has at most one parent).
    Object.keys(userChildren).forEach(function (pid) {
      var kids = userChildren[pid] || [];
      var i = kids.indexOf(childId);
      if (i >= 0) kids.splice(i, 1);
    });
    var kids = userChildren[parentId] || [];
    if (kids.indexOf(childId) < 0) kids.push(childId);
    userChildren[parentId] = kids;
    layoutTree();
    Object.keys(userPositions).forEach(function (id) {
      var el = document.getElementById('mt-node-' + id);
      var p = userPositions[id];
      if (el && p) { el.style.left = p.x + 'px'; el.style.top = p.y + 'px'; }
    });
    renderEdges();
  }

  function isAncestor(ancestorId, descendantId) {
    var visited = new Set();
    var stack = [ancestorId];
    while (stack.length) {
      var cur = stack.pop();
      if (cur === descendantId) return true;
      if (visited.has(cur)) continue;
      visited.add(cur);
      (userChildren[cur] || []).forEach(function (k) { stack.push(k); });
    }
    return false;
  }

  // -------------------------------------------------------- port connecting

  function setupPortConnecting() {
    // Delegate port clicks from the nodes container.
    elNodesContainer.addEventListener('click', function (e) {
      var port = e.target.closest('.mt-port');
      if (!port) return;
      e.stopPropagation();
      handlePortClick(port);
    });

    // Click on empty canvas while connecting → cancel.
    elViewport.addEventListener('click', function (e) {
      if (connectingPort && !e.target.closest('.mt-port') && !e.target.closest('.mt-node')) {
        cancelConnecting();
      }
    });

    // Drag temp connection line to follow cursor.
    elViewport.addEventListener('mousemove', function (e) {
      if (!connectingPort) return;
      updateTempConnection(e);
    });
  }

  function handlePortClick(portEl) {
    var nodeId = portEl.dataset.nodeId;
    var type = portEl.dataset.portType;

    if (!connectingPort) {
      connectingPort = { id: nodeId, type: type };
      portEl.classList.add('mt-port-connecting');
      // Create temp path.
      tempPath = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      tempPath.setAttribute('class', 'mt-temp-connection');
      elEdges.appendChild(tempPath);
      updateTempConnection({ clientX: 0, clientY: 0, _init: true });
      return;
    }

    // Complete or cancel.
    if (connectingPort.id === nodeId && connectingPort.type === type) {
      cancelConnecting();
      return;
    }
    // Valid connection: output → input (in either order).
    var outputId = null, inputId = null;
    if (connectingPort.type === 'output' && type === 'input') {
      outputId = connectingPort.id; inputId = nodeId;
    } else if (connectingPort.type === 'input' && type === 'output') {
      outputId = nodeId; inputId = connectingPort.id;
    } else {
      cancelConnecting();
      return;
    }
    cancelConnecting();
    createConnection(outputId, inputId);
  }

  function updateTempConnection(e) {
    if (!connectingPort || !tempPath) return;
    var portEl = elNodesContainer.querySelector(
      '.mt-port[data-node-id="' + connectingPort.id + '"][data-port-type="' + connectingPort.type + '"]');
    if (!portEl) return;

    var portRect = portEl.getBoundingClientRect();
    var svgRect = elEdges.getBoundingClientRect();
    // Convert port center to world coordinates (accounting for pan/zoom).
    var x1 = (portRect.left + portRect.width / 2 - svgRect.left) / scale;
    var y1 = (portRect.top + portRect.height / 2 - svgRect.top) / scale;

    var x2, y2;
    if (e._init) { x2 = x1; y2 = y1; }
    else {
      x2 = (e.clientX - svgRect.left) / scale;
      y2 = (e.clientY - svgRect.top) / scale;
    }
    // Always draw from the output side (bottom) for visual clarity.
    var fromX, fromY, toX, toY;
    if (connectingPort.type === 'output') {
      fromX = x1; fromY = y1; toX = x2; toY = y2;
    } else {
      fromX = x2; fromY = y2; toX = x1; toY = y1;
    }
    var midY = (fromY + toY) / 2;
    tempPath.setAttribute('d',
      'M ' + fromX + ' ' + fromY +
      ' C ' + fromX + ' ' + midY + ', ' +
            toX + ' ' + midY + ', ' +
            toX + ' ' + toY);
  }

  function cancelConnecting() {
    if (connectingPort) {
      var el = elNodesContainer.querySelector(
        '.mt-port[data-node-id="' + connectingPort.id + '"][data-port-type="' + connectingPort.type + '"]');
      if (el) el.classList.remove('mt-port-connecting');
    }
    connectingPort = null;
    if (tempPath && tempPath.parentNode) tempPath.parentNode.removeChild(tempPath);
    tempPath = null;
  }

  // ------------------------------------------------------------ node dragging

  function attachDragHandler(nodeEl, headerEl, nodeId) {
    var dragStartPos = null;
    var didDrag = false;

    headerEl.addEventListener('mousedown', function (e) {
      // Only drag on the header proper, not on ports.
      if (e.target.closest('.mt-port')) return;
      if (e.target.closest('.mt-node-resizer')) return;
      if (e.button !== 0) return;
      e.stopPropagation();
      var rect = nodeEl.getBoundingClientRect();
      dragState = {
        id: nodeId,
        offsetX: e.clientX - rect.left,
        offsetY: e.clientY - rect.top,
      };
      dragStartPos = { x: e.clientX, y: e.clientY };
      didDrag = false;
      e.preventDefault();
    });

    // Track whether the mouse actually moved while the button was down.
    window.addEventListener('mousemove', function (e) {
      if (!dragStartPos || dragState === null || dragState.id !== nodeId) return;
      var dx = e.clientX - dragStartPos.x;
      var dy = e.clientY - dragStartPos.y;
      if (Math.abs(dx) + Math.abs(dy) > 4) didDrag = true;
    });

    window.addEventListener('mouseup', function () {
      dragStartPos = null;
    });
  }

  function attachResizeHandler(nodeEl, resizerEl, nodeId) {
    resizerEl.addEventListener('mousedown', function (e) {
      if (e.button !== 0) return;
      e.stopPropagation();
      e.preventDefault();
      var rect = nodeEl.getBoundingClientRect();
      resizeState = {
        id: nodeId,
        startWidth: rect.width / scale,
        startHeight: rect.height / scale,
        startClientX: e.clientX,
        startClientY: e.clientY,
      };
    });
  }

  document.addEventListener('mousemove', function (e) {
    if (resizeState) {
      var dx = (e.clientX - resizeState.startClientX) / scale;
      var dy = (e.clientY - resizeState.startClientY) / scale;
      var newW = Math.max(160, resizeState.startWidth + dx);
      var newH = Math.max(80, resizeState.startHeight + dy);
      var el = document.getElementById('mt-node-' + resizeState.id);
      if (el) {
        el.style.width = newW + 'px';
        el.style.height = newH + 'px';
      }
      if (userPositions[resizeState.id]) {
        userPositions[resizeState.id].cx = userPositions[resizeState.id].x + newW / 2;
        userPositions[resizeState.id].height = newH;
      }
      renderEdges();
      return;
    }
    if (!dragState) return;
    var viewportRect = elViewport.getBoundingClientRect();
    // Convert client coords to world coords.
    var wx = (e.clientX - viewportRect.left - panX) / scale;
    var wy = (e.clientY - viewportRect.top - panY) / scale;
    var newX = wx - dragState.offsetX / scale;
    var newY = wy - dragState.offsetY / scale;

    var el = document.getElementById('mt-node-' + dragState.id);
    if (el) {
      el.style.left = newX + 'px';
      el.style.top = newY + 'px';
    }
    if (userPositions[dragState.id]) {
      userPositions[dragState.id].x = newX;
      userPositions[dragState.id].y = newY;
      userPositions[dragState.id].cx = newX +
        (userPositions[dragState.id].width || CARD_WIDTH) / 2;
    }
    renderEdges();
  });

  document.addEventListener('mouseup', function () {
    if (dragState) dragState = null;
    if (resizeState) resizeState = null;
  });

  // ----------------------------------------------------------- pan / zoom

  function applyTransform() {
    elWorld.style.transform =
      'translate(' + panX + 'px, ' + panY + 'px) scale(' + scale + ')';
  }

  function setupPanZoom() {
    // Suppress the browser context menu on the canvas so right-click can pan.
    elViewport.addEventListener('contextmenu', function (e) {
      e.preventDefault();
    });

    elViewport.addEventListener('mousedown', function (e) {
      // Right button (2) pans; middle button (1) also pans as a fallback.
      if (e.button !== 2 && e.button !== 1) return;
      if (e.target.closest('.mt-node')) return;
      if (connectingPort) return;
      isPanning = true;
      panStartX = e.clientX; panStartY = e.clientY;
      panOriginX = panX; panOriginY = panY;
      elViewport.classList.add('mt-panning');
      e.preventDefault();
    });
    window.addEventListener('mouseup', function (e) {
      if (isPanning && (e.button === 2 || e.button === 1 || e.button === undefined)) {
        isPanning = false;
        elViewport.classList.remove('mt-panning');
      }
    });
    window.addEventListener('mousemove', function (e) {
      if (!isPanning) return;
      panX = panOriginX + (e.clientX - panStartX);
      panY = panOriginY + (e.clientY - panStartY);
      applyTransform();
    });
    elViewport.addEventListener('wheel', function (e) {
      e.preventDefault();
      var rect = elViewport.getBoundingClientRect();
      var mx = e.clientX - rect.left;
      var my = e.clientY - rect.top;
      var wx = (mx - panX) / scale;
      var wy = (my - panY) / scale;
      var factor = Math.exp(-e.deltaY * 0.0015);
      var newScale = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, scale * factor));
      panX = mx - wx * newScale;
      panY = my - wy * newScale;
      scale = newScale;
      applyTransform();
    }, { passive: false });
  }

  function fitToScreen() {
    var ids = Object.keys(userPositions);
    if (ids.length === 0) return;
    var minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    ids.forEach(function (id) {
      var p = userPositions[id];
      var node = nodeData[id];
      var h = node ? estimateHeight(node) : (HEADER_HEIGHT + BODY_MIN_HEIGHT);
      minX = Math.min(minX, p.x);
      minY = Math.min(minY, p.y);
      maxX = Math.max(maxX, p.x + CARD_WIDTH);
      maxY = Math.max(maxY, p.y + h);
    });
    if (!isFinite(minX)) return;
    var rect = elViewport.getBoundingClientRect();
    var padding = 40;
    var w = maxX - minX + padding * 2;
    var h = maxY - minY + padding * 2;
    var sx = rect.width / w;
    var sy = rect.height / h;
    scale = Math.min(sx, sy, 1.0);
    scale = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, scale));
    panX = (rect.width - (maxX + minX) * scale) / 2;
    panY = padding;
    applyTransform();
  }

  function goToRoot() {
    // Find any root node (no incoming edge) and center on it.
    var incoming = {};
    Object.keys(userChildren).forEach(function (pid) {
      (userChildren[pid] || []).forEach(function (cid) { incoming[cid] = pid; });
    });
    var rootId = Object.keys(nodeData).find(function (id) { return !incoming[id]; });
    if (!rootId) return;
    var pos = userPositions[rootId];
    if (!pos) return;
    var rect = elViewport.getBoundingClientRect();
    scale = 1;
    panX = rect.width / 2 - pos.cx;
    panY = 30;
    applyTransform();
  }

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  }
})();
