// Keyboard shortcuts
document.addEventListener('keydown', function(e) {
  // Ctrl+G: Plot grid toggle hint
  if ((e.ctrlKey || e.metaKey) && e.key === 'g') {
    e.preventDefault();
    console.log('Plot operations: use mouse wheel to zoom, drag to pan, double-click to reset');
  }
});

// Auto-save functionality
let autoSaveTimer = null;
function setupAutoSave() {
  // Clear any existing timer
  if (autoSaveTimer) {
    clearInterval(autoSaveTimer);
  }
  // Auto-save every 2 minutes
  autoSaveTimer = setInterval(function() {
    if (typeof saveNotebook === 'function') {
      try {
        const notebookData = getNotebookData();
        if (notebookData.cells && notebookData.cells.length > 0) {
          localStorage.setItem('highdream_autosave', JSON.stringify(notebookData));
          localStorage.setItem('highdream_autosave_time', new Date().toISOString());
        }
      } catch (e) {
        console.error('Auto-save failed:', e);
      }
    }
  }, 120000); // 2 minutes
}

function loadAutoSave() {
  try {
    const autosaveData = localStorage.getItem('highdream_autosave');
    const autosaveTime = localStorage.getItem('highdream_autosave_time');
    if (autosaveData && autosaveTime) {
      const saveTime = new Date(autosaveTime);
      const now = new Date();
      const diffMinutes = (now - saveTime) / 60000;
      if (diffMinutes < 60) { // Only restore if saved within last hour
        if (confirm('发现自动保存的笔记本（' + saveTime.toLocaleString() + '），是否恢复？')) {
          if (typeof loadNotebookData === 'function') {
            loadNotebookData(JSON.parse(autosaveData));
          }
        }
      }
    }
  } catch (e) {
    console.error('Failed to load auto-save:', e);
  }
}

// Initialize auto-save on page load
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', function() {
    setupAutoSave();
    setTimeout(loadAutoSave, 1000);
  });
} else {
  setupAutoSave();
  setTimeout(loadAutoSave, 1000);
}

document.addEventListener('keydown', function(e) {
  // Ctrl+Enter or Cmd+Enter: Run current cell
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
    e.preventDefault();
    const activeCell = document.querySelector('.cell-active');
    if (activeCell && !activeCell.classList.contains('markdown-cell')) {
      const runBtn = activeCell.querySelector('.run-cell-button');
      if (runBtn) runBtn.click();
    } else {
      // If no active cell or in markdown, find the last cell and run it or add new
      const cells = document.querySelectorAll('.cell');
      if (cells.length > 0) {
        const lastCell = cells[cells.length - 1];
        if (lastCell.classList.contains('code-cell')) {
          setActiveCell(lastCell);
          const runBtn = lastCell.querySelector('.run-cell-button');
          if (runBtn) runBtn.click();
        }
      }
    }
  }
  
  // Ctrl+S or Cmd+S: Save notebook
  if ((e.ctrlKey || e.metaKey) && e.key === 's') {
    e.preventDefault();
    const saveBtn = document.getElementById('saveNotebookBtn');
    if (saveBtn) saveBtn.click();
  }
  
  // Ctrl+Shift+N or Cmd+Shift+N: Add new code cell
  if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === 'N') {
    e.preventDefault();
    if (typeof addNewCell === 'function') {
      addNewCell('code');
    }
  }
  
  // Ctrl+Shift+M or Cmd+Shift+M: Add new markdown cell
  if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === 'M') {
    e.preventDefault();
    if (typeof addNewCell === 'function') {
      addNewCell('markdown');
    }
  }
  
  // Escape: Close fullscreen or clear focus
  if (e.key === 'Escape') {
    const overlay = document.querySelector('.math-fullscreen-overlay.active');
    if (overlay) {
      const closeBtn = overlay.querySelector('.math-fullscreen-close');
      if (closeBtn) closeBtn.click();
    }
  }
});

// Plot helpers
function _plotSvgEl(tag, attrs) {
  const el = document.createElementNS('http://www.w3.org/2000/svg', tag);
  if (attrs) {
    Object.keys(attrs).forEach((k) => el.setAttribute(k, attrs[k]));
  }
  return el;
}

function collectPlotPayload(cell) {
  const varEl = cell.querySelector('.var-input input');
  const payload = {
    expression: readExpr(cell),
    variable: (varEl && varEl.value.trim()) || 'x',
  };
  // A definite-integral window is a natural x-range.
  const lo = cell.querySelector('.integral-lower');
  const hi = cell.querySelector('.integral-upper');
  if (lo && hi && String(hi.value).trim() !== '') {
    const a = parseFloat(lo.value);
    const b = parseFloat(hi.value);
    if (isFinite(a) && isFinite(b) && b > a) { payload.xmin = a; payload.xmax = b; }
  }
  if (getOp(cell) === 'tangent') {
    const p = cell.querySelector('.applied-point');
    if (p) {
      const t = parseFloat(p.value);
      if (isFinite(t)) payload.tangent_at = t;
    }
  }
  return payload;
}

function renderPlotSvg(data) {
  const W = 320; const H = 220; const P = 30;
  const pts = data.points || [];
  const xs = pts.map((p) => p[0]);
  let xmin0 = Math.min.apply(null, xs); let xmax0 = Math.max.apply(null, xs);
  if (!(xmax0 > xmin0)) xmax0 = xmin0 + 1;
  let ymin0 = (typeof data.ymin === 'number') ? data.ymin : -1;
  let ymax0 = (typeof data.ymax === 'number') ? data.ymax : 1;
  if (!(ymax0 > ymin0)) ymax0 = ymin0 + 1;

  let xmin = xmin0, xmax = xmax0, ymin = ymin0, ymax = ymax0;

  const svg = _plotSvgEl('svg', { viewBox: '0 0 ' + W + ' ' + H, class: 'hd-plot-svg hd-plot-interactive', preserveAspectRatio: 'xMidYMid meet' });
  const sx = (x) => P + (x - xmin) / (xmax - xmin) * (W - 2 * P);
  const sy = (y) => H - P - (y - ymin) / (ymax - ymin) * (H - 2 * P);

  let showGrid = false;
  let gridGroup = null;
  
  const redraw = () => {
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    
    // Draw grid if enabled
    if (showGrid) {
      gridGroup = _plotSvgEl('g', { class: 'hd-plot-grid-group' });
      // Vertical grid lines
      const xStep = Math.pow(10, Math.floor(Math.log10((xmax - xmin) / 5)));
      for (let gx = Math.ceil(xmin / xStep) * xStep; gx <= xmax; gx += xStep) {
        if (Math.abs(gx) < 1e-10) continue;
        gridGroup.appendChild(_plotSvgEl('line', { 
          x1: sx(gx), y1: P, x2: sx(gx), y2: H - P, 
          class: 'hd-plot-grid' 
        }));
      }
      // Horizontal grid lines
      const yStep = Math.pow(10, Math.floor(Math.log10((ymax - ymin) / 5)));
      for (let gy = Math.ceil(ymin / yStep) * yStep; gy <= ymax; gy += yStep) {
        if (Math.abs(gy) < 1e-10) continue;
        gridGroup.appendChild(_plotSvgEl('line', { 
          x1: P, y1: sy(gy), x2: W - P, y2: sy(gy), 
          class: 'hd-plot-grid' 
        }));
      }
      svg.appendChild(gridGroup);
    }
    
    svg.appendChild(_plotSvgEl('line', { x1: P, y1: H - P, x2: W - P, y2: H - P, class: 'hd-plot-axis' }));
    svg.appendChild(_plotSvgEl('line', { x1: P, y1: P, x2: P, y2: H - P, class: 'hd-plot-axis' }));
    if (ymin <= 0 && 0 <= ymax) svg.appendChild(_plotSvgEl('line', { x1: P, y1: sy(0), x2: W - P, y2: sy(0), class: 'hd-plot-zero' }));
    if (xmin <= 0 && 0 <= xmax) svg.appendChild(_plotSvgEl('line', { x1: sx(0), y1: P, x2: sx(0), y2: H - P, class: 'hd-plot-zero' }));
    let run = [];
    const flush = () => {
      if (run.length > 1) {
        const d = run.map((p, i) => (i === 0 ? 'M' : 'L') + sx(p[0]).toFixed(1) + ' ' + sy(p[1]).toFixed(1)).join(' ');
        svg.appendChild(_plotSvgEl('path', { d, class: 'hd-plot-curve' }));
      }
      run = [];
    };
    pts.forEach((p) => { if (p[1] === null || p[1] === undefined) flush(); else run.push(p); });
    flush();
    if (data.tangent) {
      const t = data.tangent;
      svg.appendChild(_plotSvgEl('line', { x1: sx(xmin), y1: sy(t.y1), x2: sx(xmax), y2: sy(t.y2), class: 'hd-plot-tangent' }));
      svg.appendChild(_plotSvgEl('circle', { cx: sx(t.x), cy: sy(t.y), r: 3, class: 'hd-plot-marker' }));
    }
    if (Array.isArray(data.roots)) {
      const g = _plotSvgEl('g', { class: 'hd-plot-roots' });
      data.roots.forEach((r) => {
        if (typeof r.x !== 'number') return;
        if (r.x < xmin || r.x > xmax) return;
        const cx = sx(r.x); const cy = sy(0);
        g.appendChild(_plotSvgEl('circle', { cx, cy, r: 3, class: 'hd-plot-root' }));
        const tt = _plotSvgEl('text', { x: cx + 4, y: cy - 4, class: 'hd-plot-root-label' });
        const rx = Math.abs(r.x) >= 1e3 || Math.abs(r.x) < 1e-3 ? r.x.toExponential(2) : r.x.toFixed(3);
        tt.textContent = '(' + rx.replace(/\.?0+e/, 'e') + ', 0)';
        g.appendChild(tt);
      });
      if (g.children.length) svg.appendChild(g);
    }
    const label = (str, x, y) => { const e = _plotSvgEl('text', { x, y, class: 'hd-plot-tick' }); e.textContent = str; svg.appendChild(e); };
    const r2 = (v) => String(Math.round(v * 100) / 100);
    label(r2(xmin), P, H - P + 12);
    label(r2(xmax), W - P, H - P + 12);
  };

  redraw();

  svg.addEventListener('wheel', (e) => {
    e.preventDefault();
    const factor = e.deltaY > 0 ? 1.1 : 0.9;
    const rect = svg.getBoundingClientRect();
    const px = (e.clientX - rect.left) / rect.width;
    const py = (e.clientY - rect.top) / rect.height;
    const dx = xmin + px * (xmax - xmin);
    const dy = ymin + (1 - py) * (ymax - ymin);
    xmin = dx - (dx - xmin) * factor;
    xmax = dx + (xmax - dx) * factor;
    ymin = dy - (dy - ymin) * factor;
    ymax = dy + (ymax - dy) * factor;
    redraw();
  }, { passive: false });

  let dragging = false, lastX = 0, lastY = 0;
  svg.addEventListener('mousedown', (e) => {
    dragging = true;
    lastX = e.clientX; lastY = e.clientY;
    svg.style.cursor = 'grabbing';
  });
  window.addEventListener('mousemove', (e) => {
    if (!dragging) return;
    const rect = svg.getBoundingClientRect();
    const dx = (e.clientX - lastX) / rect.width * (xmax - xmin);
    const dy = (e.clientY - lastY) / rect.height * (ymax - ymin);
    xmin -= dx; xmax -= dx;
    ymin += dy; ymax += dy;
    lastX = e.clientX; lastY = e.clientY;
    redraw();
  });
  window.addEventListener('mouseup', () => {
    if (dragging) {
      dragging = false;
      svg.style.cursor = 'grab';
    }
  });
  svg.addEventListener('dblclick', () => {
    xmin = xmin0; xmax = xmax0; ymin = ymin0; ymax = ymax0;
    redraw();
  });
  
  // Add plot controls
  svg.addEventListener('contextmenu', (e) => {
    e.preventDefault();
    // Simple toggle for grid via double click + shift
  });
  
  svg.style.cursor = 'grab';
  return svg;
}
