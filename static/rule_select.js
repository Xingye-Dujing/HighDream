// Manual rule-selection page logic

document.addEventListener('DOMContentLoaded', function () {
  'use strict';

  // State
  let sessionId = null;
  let currentState = null;
  let busy = false;

  // DOM refs
  const $domain = document.getElementById('rs-domain');
  const $expr = document.getElementById('rs-expr');
  const $var = document.getElementById('rs-var');
  const $point = document.getElementById('rs-point');
  const $dir = document.getElementById('rs-dir');
  const $startBtn = document.getElementById('rs-start-btn');
  const $resetBtn = document.getElementById('rs-reset-btn');
  const $error = document.getElementById('rs-error');
  const $stepsList = document.getElementById('rs-steps-list');
  const $rulesList = document.getElementById('rs-rules-list');
  const $actions = document.getElementById('rs-actions');
  const $fallbackBtn = document.getElementById('rs-fallback-btn');
  const $finishBtn = document.getElementById('rs-finish-btn');
  const $statusPanel = document.getElementById('rs-status-panel');
  const $statusContent = document.getElementById('rs-status-content');
  const $currentLabel = document.getElementById('rs-current-expr-label');
  const $divider = document.getElementById('rs-divider');
  const $leftCol = document.getElementById('rule-select-left');
  const $layout = document.getElementById('rule-select-layout');

  // --------------------------------------------------------------- helpers

  async function api(path, body) {
    const r = await fetch('/api' + path, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body),
    });
    return r.json();
  }

  function showError(msg) {
    $error.textContent = msg;
    $error.style.display = 'block';
  }

  function hideError() {
    $error.style.display = 'none';
  }

  function disableInputs(v) {
    $domain.disabled = v;
    $expr.disabled = v;
    $var.disabled = v;
    $point.disabled = v;
    $dir.disabled = v;
    $startBtn.disabled = v;
  }

  // ----------------------------------------------------------- re-render

  function render(state) {
    // Steps
    let html = '';
    for (let i = 0; i < state.steps.length; i++) {
      const st = state.steps[i];
      const active = !state.done && i === state.steps.length - 1;
      let expl = i === 0 ? '初始表达式' : (st.explanation ? st.explanation : '');
      /* replace $...$ with \(...\) for inline math — tex-chtml-full.js default */
      expl = expl.replace(/\$([^$]+)\$/g, '\\(' + '$1' + '\\)');
      html += `<div class="rs-step${active ? ' active' : ''}">
        <div class="rs-step-explanation">${expl}</div>
        <div class="rs-step-latex">$$${st.latex}$$</div>
      </div>`;
    }
    $stepsList.innerHTML = html;

    // MathJax re-render
    if (window.MathJax && MathJax.typesetPromise) {
      MathJax.typesetPromise([$stepsList]).catch(function () {});
    }

    // Current expression label
    if (state.current_expr_latex && !state.done) {
      $currentLabel.innerHTML = '当前: \\(' + state.current_expr_latex + '\\)';
      if (window.MathJax && MathJax.typesetPromise) {
        MathJax.typesetPromise([$currentLabel]).catch(function () {});
      }
    } else {
      $currentLabel.textContent = '';
    }

    // Applicable rules
    if (state.done) {
      $rulesList.innerHTML = '<p class="rs-placeholder"><i class="fas fa-check-circle" style="color:#789262"></i> 推导完成</p>';
      $actions.style.display = 'none';
      renderStatus(state);
      return;
    }

    if (state.error) {
      showError(state.error);
      // Still show rules so the user can try a different rule
    }

    const rules = state.applicable_rules || [];
    if (rules.length === 0) {
      $rulesList.innerHTML = '<p class="rs-placeholder">没有可应用的规则 — 试试 SymPy 回退</p>';
      $actions.style.display = 'block';
      return;
    }

    let rulesHtml = '';
    for (const r of rules) {
      const previewHtml = r.latex_preview
        ? '<span class="rs-rule-preview">\\(' + r.latex_preview + '\\)</span>'
        : '';
      rulesHtml += `<div class="rs-rule-card" data-rule="${r.name}">
        <div class="rs-rule-name">${r.display_name || r.name}</div>
        ${previewHtml}
      </div>`;
    }
    $rulesList.innerHTML = rulesHtml;

    // Wire rule clicks
    $rulesList.querySelectorAll('.rs-rule-card').forEach(function (card) {
      card.addEventListener('click', function () {
        if (busy) return;
        applyRule(card.dataset.rule);
      });
    });

    // Re-render MathJax for previews
    if (window.MathJax && MathJax.typesetPromise) {
      MathJax.typesetPromise([$rulesList]).catch(function () {});
    }

    $actions.style.display = 'block';
    if (!state.error) {
      hideError();
    }
  }

  function renderStatus(state) {
    $statusPanel.style.display = 'block';
    const lines = [];
    lines.push('<div>步骤数: <strong>' + state.steps.length + '</strong></div>');
    lines.push('<div>域: <strong>' + state.domain + '</strong></div>');
    if (state.done) {
      lines.push('<div style="margin-top:6px;color:#789262"><i class="fas fa-check-circle"></i> 推导完成</div>');
    }
    $statusContent.innerHTML = lines.join('');
  }

  // -------------------------------------------------------------- actions

  function clearUI() {
    $stepsList.innerHTML = '<p class="rs-placeholder">输入表达式后点击「开始」</p>';
    $rulesList.innerHTML = '<p class="rs-placeholder">点击「开始」加载可用规则</p>';
    $actions.style.display = 'none';
    $statusPanel.style.display = 'none';
    $currentLabel.textContent = '';
    hideError();
    disableInputs(false);
    $resetBtn.style.display = 'none';
  }

  async function startSession() {
    if (busy) return;
    busy = true;
    hideError();
    disableInputs(true);

    const body = {
      domain: $domain.value,
      expression: $expr.value.trim(),
      variable: $var.value.trim() || 'x',
    };
    if ($domain.value === 'limit') {
      body.point = $point.value.trim() || '0';
      body.direction = $dir.value;
    }

    try {
      const resp = await api('/manual_start', body);
      if (!resp.success) {
        showError(resp.error || '启动失败');
        disableInputs(false);
        busy = false;
        return;
      }
      sessionId = resp.session_id;
      currentState = resp.state;
      $resetBtn.style.display = 'inline-block';
      render(currentState);
    } catch (e) {
      showError('网络错误: ' + e.message);
      disableInputs(false);
    }
    busy = false;
  }

  async function applyRule(ruleName) {
    if (busy || !sessionId) return;
    busy = true;
    hideError();

    try {
      const resp = await api('/manual_step', {session_id: sessionId, rule_name: ruleName});
      if (!resp.success) {
        showError(resp.error || '应用规则失败');
        busy = false;
        return;
      }
      currentState = resp.state;
      render(currentState);
    } catch (e) {
      showError('网络错误: ' + e.message);
    }
    busy = false;
  }

  async function fallback() {
    if (busy || !sessionId) return;
    busy = true;
    hideError();

    try {
      const resp = await api('/manual_fallback', {session_id: sessionId});
      if (!resp.success) {
        showError(resp.error || '回退失败');
        busy = false;
        return;
      }
      currentState = resp.state;
      render(currentState);
    } catch (e) {
      showError('网络错误: ' + e.message);
    }
    busy = false;
  }

  function finishSession() {
    if (!sessionId) return;
    // Mark done locally and show status
    if (currentState) {
      currentState.done = true;
    }
    $rulesList.innerHTML = '<p class="rs-placeholder"><i class="fas fa-check-circle" style="color:#789262"></i> 推导完成</p>';
    $actions.style.display = 'none';
    renderStatus(currentState || {steps: [], domain: $domain.value, done: true});
    // Keep session alive for status review
  }

  async function resetSession() {
    if (sessionId) {
      try {
        await api('/manual_reset', {session_id: sessionId});
      } catch (_) {}
    }
    sessionId = null;
    currentState = null;
    clearUI();
  }

  // ------------------------------------------------------- divider drag

  let dragState = null;

  function initDividerWidth() {
    var saved = localStorage.getItem('rs_divider_width');
    if (saved && $leftCol) {
      $leftCol.style.width = saved;
      $leftCol.style.flex = '0 0 auto';
    }
  }

  if ($divider && $leftCol && $layout) {
    initDividerWidth();

    $divider.addEventListener('mousedown', function (e) {
      e.preventDefault();
      dragState = {
        startX: e.clientX,
        startWidth: $leftCol.getBoundingClientRect().width,
      };
      $divider.classList.add('dragging');
      document.body.style.cursor = 'col-resize';
      document.body.style.userSelect = 'none';
    });
  }

  document.addEventListener('mousemove', function (e) {
    if (!dragState) return;
    var dx = e.clientX - dragState.startX;
    var newWidth = Math.max(260, dragState.startWidth + dx);
    var maxWidth = $layout.getBoundingClientRect().width - 380;
    if (newWidth > maxWidth) newWidth = maxWidth;
    $leftCol.style.width = newWidth + 'px';
    $leftCol.style.flex = '0 0 auto';
  });

  document.addEventListener('mouseup', function () {
    if (!dragState) return;
    $divider.classList.remove('dragging');
    document.body.style.cursor = '';
    document.body.style.userSelect = '';
    // Save to localStorage
    try {
      localStorage.setItem('rs_divider_width', $leftCol.style.width);
    } catch (_) {}
    dragState = null;
  });

  // ----------------------------------------------------------- events

  function onDomainChange() {
    const isLimit = $domain.value === 'limit';
    document.querySelectorAll('.rs-limits-only').forEach(function (el) {
      el.style.display = isLimit ? 'flex' : 'none';
    });
  }

  $domain.addEventListener('change', onDomainChange);
  onDomainChange();

  $startBtn.addEventListener('click', startSession);
  $resetBtn.addEventListener('click', resetSession);
  $fallbackBtn.addEventListener('click', fallback);
  $finishBtn.addEventListener('click', finishSession);

  // Enter key in expr input triggers start
  $expr.addEventListener('keydown', function (e) {
    if (e.key === 'Enter') {
      e.preventDefault();
      startSession();
    }
  });

  // Keyboard shortcut: Escape to reset
  document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape' && sessionId) {
      resetSession();
    }
  });

  // Initial placeholder
  clearUI();
});
