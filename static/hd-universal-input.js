/* ==========================================================================
   HighDream — universal input for the matrix pages
   --------------------------------------------------------------------------
   The matrix pages used a plain <textarea>.  They now use the same
   WYSIWYG MathLive editor as the notebook's 万能输入框, and the matrix form is
   real math (bmatrix), not literal "[[1,2],[3,4]]" text:

     * an operator-form prefill such as [[1,2],[3,4]] is converted to LaTeX
       before it is shown, so it renders as a real 2x2 matrix;
     * the matrix palette inserts LaTeX templates whose \placeholder{} slots
       become MathLive's native fillable boxes;
     * the hidden textarea stays the canonical value store, so existing page
       scripts (run/save/load) keep working unchanged.

   Usage:  <textarea data-universal-input placeholder="..."></textarea>
           <script src="/static/hd-universal-input.js"></script>
   ========================================================================== */
;(function () {
  'use strict'

  var proto = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value')

  /* MathLive paints a keyboard toggle and an insert-menu button inside its
     shadow root.  The palette is the only insertion path, so hide both. */
  function hideToggles(field) {
    var apply = function () {
      var sr = field.shadowRoot
      if (!sr || sr.querySelector('#hd-hide-ml-toggles')) return
      var style = document.createElement('style')
      style.id = 'hd-hide-ml-toggles'
      style.textContent =
        '.ML__virtual-keyboard-toggle,.ML__menu-toggle{display:none!important}'
      sr.appendChild(style)
    }
    apply()
    requestAnimationFrame(apply)
    setTimeout(apply, 400)
  }

  function isLatex(text) {
    return /\\[a-zA-Z]|\\begin|\\frac|\\placeholder/.test(text)
  }

  /* Split on a separator that is not nested inside (), [] or {}. */
  function splitTopLevel(text, separator) {
    var parts = []
    var depth = 0
    var current = ''
    for (var i = 0; i < text.length; i += 1) {
      var ch = text[i]
      if (ch === '(' || ch === '[' || ch === '{') depth += 1
      else if (ch === ')' || ch === ']' || ch === '}') depth -= 1
      if (ch === separator && depth === 0) {
        parts.push(current)
        current = ''
      } else {
        current += ch
      }
    }
    parts.push(current)
    return parts
  }

  /* [[1,2],[3,4]] -> \begin{bmatrix}1 & 2 \\ 3 & 4\end{bmatrix} */
  function nestedMatrixToLatex(text) {
    var trimmed = String(text || '').trim()
    if (trimmed.indexOf('[[') !== 0 || trimmed.slice(-2) !== ']]') return ''
    var inner = trimmed.slice(1, -1)
    var rows = splitTopLevel(inner, ',')
    var body = []
    for (var r = 0; r < rows.length; r += 1) {
      var row = rows[r].trim()
      if (row[0] !== '[' || row.slice(-1) !== ']') return ''
      var cells = splitTopLevel(row.slice(1, -1), ',')
      var latexCells = cells.map(function (cell) {
        return cell
          .trim()
          .replace(/\*\*/g, '^')
          .replace(/\*/g, '\\cdot ')
          .replace(/pi\b/g, '\\pi ')
      })
      body.push(latexCells.join(' & '))
    }
    if (!body.length) return ''
    return '\\begin{bmatrix}' + body.join(' \\\\ ') + '\\end{bmatrix}'
  }

  /* Return a displayable LaTeX string. Local cases resolve synchronously; any
     other operator form is handed to the existing /api/convert_form endpoint. */
  function toDisplayLatex(text) {
    var raw = String(text || '')
    if (!raw.trim()) return Promise.resolve('')
    if (isLatex(raw)) return Promise.resolve(raw)
    var local = nestedMatrixToLatex(raw)
    if (local) return Promise.resolve(local)
    return fetch('/api/convert_form', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ expression: raw }),
    })
      .then(function (response) {
        return response.json()
      })
      .then(function (data) {
        return data && data.success && typeof data.result === 'string' ? data.result : raw
      })
      .catch(function () {
        return raw
      })
  }

  function attach(textarea) {
    if (!textarea || textarea.__hdUniversalField) return textarea && textarea.__hdUniversalField

    var field = document.createElement('math-field')
    field.className = 'math-editor hd-universal-field'
    field.setAttribute('virtual-keyboard-mode', 'manual')
    field.setAttribute('toolbar', 'hide')
    if (textarea.placeholder) field.setAttribute('placeholder', textarea.placeholder)

    textarea.classList.add('hd-universal-source')
    textarea.parentNode.insertBefore(field, textarea)

    var sequence = 0
    function showValue(value) {
      var my = sequence + 1
      sequence = my
      toDisplayLatex(value).then(function (latex) {
        if (sequence === my) field.value = latex
      })
    }

    showValue(proto.get.call(textarea))

    var syncing = false
    field.addEventListener('input', function () {
      if (syncing) return
      syncing = true
      proto.set.call(textarea, field.value)
      syncing = false
      textarea.dispatchEvent(new Event('input', { bubbles: true }))
    })

    /* Focusing the visible editor activates its notebook cell, exactly like
       clicking the old textarea did. */
    field.addEventListener('focus', function () {
      var cell = textarea.closest('.cell')
      if (cell && typeof window.setActiveCell === 'function') window.setActiveCell(cell)
    })

    /* Programmatic writes (loadNotebook, restoring a snapshot, ...) still go to
       the textarea; mirror them into the visible editor as real math. */
    Object.defineProperty(textarea, 'value', {
      configurable: true,
      get: function () {
        return proto.get.call(this)
      },
      set: function (value) {
        proto.set.call(this, value)
        if (!syncing) showValue(value === null || value === undefined ? '' : String(value))
      },
    })

    textarea.__hdUniversalField = field
    /* The matrix keyboard handlers look for this to insert at the visible
       editor's caret instead of the hidden textarea. */
    textarea.__hdField = field
    hideToggles(field)
    return field
  }

  function scan(root) {
    if (!root || !root.querySelectorAll) return
    if (root.matches && root.matches('textarea[data-universal-input]')) attach(root)
    root.querySelectorAll('textarea[data-universal-input]').forEach(attach)
  }

  /* Render a key's insert-LaTeX as the math it actually inserts, filling empty
     template slots with \square so the placeholder structure is visible — the
     same treatment the notebook palette gets. */
  function displayLatex(raw) {
    var s = String(raw || '')
      .replace(/\\placeholder\s*\{\s*\}/g, '\\square')
      .replace(/\\placeholder\b/g, '\\square')
    s = s.replace(/\{\s*\}/g, '{\\square}').replace(/\[\s*\]/g, '[\\square]')
    if (/^[\^_]/.test(s)) s = '\\square' + s
    s = s.replace(
      /(?<![\\A-Za-z0-9])([A-Za-z][A-Za-z0-9]*)(?=\()/g,
      '\\operatorname{$1}'
    )
    var depth = 0
    var out = ''
    for (var i = 0; i < s.length; i += 1) {
      var ch = s[i]
      if (ch === '{') {
        depth += 1
        out += ch
      } else if (ch === '}') {
        if (depth > 0) {
          depth -= 1
          out += ch
        }
      } else {
        out += ch
      }
    }
    return out
  }

  function typesetKeyboard() {
    var kb = document.getElementById('math-keyboard')
    if (!kb) return
    if (window.MathJax && typeof window.MathJax.typesetPromise === 'function') {
      if (typeof window.MathJax.typesetClear === 'function') window.MathJax.typesetClear([kb])
      window.MathJax.typesetPromise([kb]).catch(function () {
        /* MathJax failure must not break the palette */
      })
    } else {
      setTimeout(typesetKeyboard, 200)
    }
  }

  function decorateKeyboardKeys() {
    document.querySelectorAll('#math-keyboard .keyboard-key[data-latex]').forEach(function (key) {
      if (key.__hdDecorated) return
      var raw = key.getAttribute('data-latex')
      if (!raw) return
      var label = key.textContent.trim()
      if (label) key.title = label
      key.classList.add('keyboard-key-math')
      key.textContent = '\\(' + displayLatex(raw) + '\\)'
      key.__hdDecorated = true
    })
    typesetKeyboard()
  }

  function boot() {
    if (window.MathLive && typeof window.MathLive.setFonts === 'function') {
      try {
        window.MathLive.setFonts('/static/mathlive/fonts/')
      } catch (e) {
        /* offline font preload is best-effort */
      }
    }
    scan(document)
    decorateKeyboardKeys()

    if (window.MutationObserver) {
      new MutationObserver(function (records) {
        records.forEach(function (record) {
          record.addedNodes.forEach(function (node) {
            if (node.nodeType === 1) scan(node)
          })
        })
      }).observe(document.body, { childList: true, subtree: true })
    }
  }

  window.HDUniversalInput = {
    attach: attach,
    scan: scan,
    toDisplayLatex: toDisplayLatex,
    fieldFor: function (textarea) {
      return textarea && textarea.__hdUniversalField
    },
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot)
  } else {
    boot()
  }
})()
