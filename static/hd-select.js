/* ==========================================================================
   HighDream custom <select>
   --------------------------------------------------------------------------
   A native dropdown's option list is painted by the operating system, so no
   amount of CSS can restyle it -- it is the one control in the product that
   would otherwise look foreign. This replaces the popup with our own, while
   leaving the real <select> in the DOM so that everything the page scripts
   already do keeps working untouched:

     * select.value            read and written as before
     * select change events    dispatched natively (bubbles: true)
     * form submission         the select is still a real form control
     * .options / optgroups    still the single source of truth

   Usage:  <script src="/static/hd-select.js"></script>
   ========================================================================== */
;(function () {
  'use strict'

  var CARET_SVG = '<svg class="hd-caret-svg" viewBox="0 0 10 6" width="10" height="6" aria-hidden="true" focusable="false"><path d="M1 1L5 5L9 1" stroke="currentColor" stroke-width="1" fill="none" stroke-linecap="round" stroke-linejoin="round"/></svg>'
  var open = null // the single open popup, if any

  /* ------------------------------------------------------------- helpers */

  function optionsOf(select) {
    var out = []
    Array.prototype.forEach.call(select.children, function (child) {
      if (child.tagName === 'OPTGROUP') {
        out.push({ group: child.label || '', options: [] })
        Array.prototype.forEach.call(child.children, function (o) {
          if (o.tagName === 'OPTION') out[out.length - 1].options.push(o)
        })
      } else if (child.tagName === 'OPTION') {
        out.push({ group: '', options: [child] })
      }
    })
    return out
  }

  function selectedOption(select) {
    if (select.selectedIndex >= 0 && select.options[select.selectedIndex]) {
      return select.options[select.selectedIndex]
    }
    return select.options[0] || null
  }

  function labelFor(select) {
    var o = selectedOption(select)
    return o ? o.textContent.trim() : ''
  }

  /*  The popup is one shared panel, moved to whichever trigger is open.      */
  function ensurePanel() {
    if (open && open.panel) return open
    var panel = document.createElement('div')
    panel.className = 'hd-select-menu'
    panel.setAttribute('role', 'listbox')
    panel.hidden = true

    /*  The border and the corner radius live on this box, which never
        scrolls. The list scrolls in a plain square child underneath it, so
        the scrollbar can never poke out past a rounded corner.             */
    var scroll = document.createElement('div')
    scroll.className = 'hd-select-scroll'
    panel.appendChild(scroll)

    document.body.appendChild(panel)
    open = { panel: panel, scroll: scroll, trigger: null, select: null, items: [], index: -1 }
    return open
  }

  /*  Drop the highlight from every row. Leaving .hd-hl behind is what made a
      previously hovered option keep its tint after a selection.           */
  function clearHighlight(st) {
    st.items.forEach(function (el) { el.classList.remove('hd-hl') })
    st.index = -1
  }

  function closeMenu() {
    var st = open
    if (!st || !st.trigger) return
    clearHighlight(st)
    st.panel.hidden = true
    st.trigger.classList.remove('hd-open')
    st.trigger.setAttribute('aria-expanded', 'false')
    st.trigger = null
    st.select = null
    st.items = []
    st.index = -1
  }

  function position(st) {
    var r = st.trigger.getBoundingClientRect()
    var panel = st.panel
    panel.style.left = Math.round(r.left) + 'px'
    panel.style.minWidth = Math.round(r.width) + 'px'

    /*  Flip above the trigger when there is not enough room below.           */
    panel.style.top = 'auto'
    panel.style.bottom = 'auto'
    var h = panel.offsetHeight
    var below = window.innerHeight - r.bottom
    if (below < h + 12 && r.top > below) {
      panel.style.bottom = Math.round(window.innerHeight - r.top + 4) + 'px'
    } else {
      panel.style.top = Math.round(r.bottom + 4) + 'px'
    }
    /*  Keep it on screen horizontally.                                       */
    var w = panel.offsetWidth
    var left = Math.min(r.left, window.innerWidth - w - 8)
    panel.style.left = Math.max(8, Math.round(left)) + 'px'
  }

  function highlight(st, next) {
    if (!st.items.length) return
    st.index = (next + st.items.length) % st.items.length
    st.items.forEach(function (el, i) {
      var on = i === st.index
      el.classList.toggle('hd-hl', on)
      if (on && el.scrollIntoView) el.scrollIntoView({ block: 'nearest' })
    })
  }

  function build(st) {
    var select = st.select
    var groups = optionsOf(select)
    var html = ''
    groups.forEach(function (g) {
      if (g.group) html += '<div class="hd-select-label">' + escapeHtml(g.group) + '</div>'
      g.options.forEach(function (o) {
        var disabled = o.disabled ? ' disabled' : ''
        var selected = o === selectedOption(select) ? ' aria-selected="true"' : ''
        html +=
          '<div class="hd-select-option' + (o === selectedOption(select) ? ' hd-sel' : '') +
          '" role="option"' + selected + disabled +
          ' data-value="' + escapeHtml(o.value) + '">' +
          '<span>' + escapeHtml(o.textContent.trim()) + '</span></div>'
      })
    })
    st.scroll.innerHTML = html
    st.scroll.scrollTop = 0
    st.items = Array.prototype.slice.call(
      st.scroll.querySelectorAll('.hd-select-option:not([disabled])'))

    /*  The pointer leaving the list means nothing is hovered, so the tint
        goes with it. */
    st.scroll.addEventListener('mouseleave', function () {
      clearHighlight(st)
    })

    st.items.forEach(function (el) {
      el.addEventListener('mouseenter', function () {
        highlight(st, st.items.indexOf(el))
      })
      el.addEventListener('click', function () {
        commit(st, el)
      })
    })
  }

  function escapeHtml(s) {
    return String(s == null ? '' : s).replace(/[&<>"']/g, function (m) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[m]
    })
  }

  function commit(st, el) {
    var select = st.select
    var trigger = st.trigger
    var value = el.getAttribute('data-value')
    closeMenu()
    trigger.focus()
    if (select.value === value) {
      /*  Still fire, so callers that rebuild on every change stay in step.    */
      select.dispatchEvent(new Event('change', { bubbles: true }))
      return
    }
    select.value = value
    select.dispatchEvent(new Event('change', { bubbles: true }))
  }

  function openMenu(trigger, select) {
    closeMenu()
    /*  Re-read the selection on open. A script that assigns select.value
        without firing `change` would otherwise leave a stale label showing.  */
    syncLabel(select)
    var st = ensurePanel()
    st.trigger = trigger
    st.select = select
    st.panel.hidden = false
    trigger.classList.add('hd-open')
    trigger.setAttribute('aria-expanded', 'true')
    build(st)
    position(st)
    var sel = selectedOption(select)
    var at = -1
    if (sel) {
      for (var i = 0; i < st.items.length; i++) {
        if (st.items[i].getAttribute('data-value') === sel.value) {
          at = i
          break
        }
      }
    }
    highlight(st, at < 0 ? 0 : at)
  }

  /* ------------------------------------------------------------ one select */

  function buildTrigger(select) {
    if (select.hdTrigger) return select.hdTrigger
    if (select.multiple || select.size > 1) return null // leave those native

    var wrapper = document.createElement('span')
    wrapper.className = 'hd-select'

    var cs = getComputedStyle(select)
    /*  The wrapper takes over the box the select used to occupy, so existing
        layout rules keep working. `width` is deliberately NOT copied: a
        computed width is a used value in px, and freezing it would break
        every select that is meant to be width:100% or to stretch. Those
        rely on the wrapper being a stretched flex item instead.             */
    ;['minWidth', 'maxWidth', 'flexGrow', 'flexShrink', 'flexBasis'].forEach(
      function (prop) {
        var v = cs[prop]
        if (v && v !== 'auto' && v !== '0px') wrapper.style[prop] = v
      }
    )

    var trigger = document.createElement('button')
    trigger.type = 'button'
    trigger.className = 'hd-select-trigger'
    trigger.setAttribute('aria-haspopup', 'listbox')
    trigger.setAttribute('aria-expanded', 'false')
    trigger.disabled = select.disabled

    var text = document.createElement('span')
    text.className = 'hd-select-text'
    trigger.appendChild(text)
    var caret = document.createElement('span')
    caret.className = 'hd-select-caret'
    caret.setAttribute('aria-hidden', 'true')
    caret.innerHTML = CARET_SVG
    trigger.appendChild(caret)

    select.parentNode.insertBefore(wrapper, select)
    wrapper.appendChild(trigger)
    wrapper.appendChild(select)

    /*  The real select stays for value / change / form submit, but it is no
        longer a tab stop and is hidden from assistive tech -- the trigger
        carries the combobox role instead.                                   */
    select.classList.add('hd-select-native')
    select.setAttribute('tabindex', '-1')
    select.setAttribute('aria-hidden', 'true')

    select.hdTrigger = trigger
    select.hdText = text

    trigger.addEventListener('click', function () {
      if (open && open.trigger === trigger) closeMenu()
      else openMenu(trigger, select)
    })
    trigger.addEventListener('keydown', function (e) {
      if (e.key === 'ArrowDown' || e.key === 'ArrowUp' || e.key === 'Enter' || e.key === ' ') {
        e.preventDefault()
        openMenu(trigger, select)
        if (e.key === 'ArrowUp') highlight(open, open.index - 1)
      }
    })

    syncLabel(select)
    return trigger
  }

  function syncLabel(select) {
    if (!select.hdText) return
    select.hdText.textContent = labelFor(select)
  }

  /* ------------------------------------------------------------------ init */

  function upgrade(select) {
    if (!(select instanceof HTMLSelectElement)) return
    if (select.hdTrigger || select.multiple || select.size > 1) return
    if (select.dataset.hdNoSelect) return
    buildTrigger(select)
  }

  function upgradeAll(root) {
    var scope = root || document
    Array.prototype.forEach.call(scope.querySelectorAll('select'), upgrade)
    if (scope.tagName === 'SELECT') upgrade(scope)
  }

  function init() {
    if (document.getElementById('hd-select-menu')) return
    ensurePanel()
    upgradeAll()

    /*  Cells, blueprint nodes and the word editor build selects at runtime,
        so watch for new ones.                                               */
    if (window.MutationObserver) {
      new MutationObserver(function () {
        upgradeAll()
      }).observe(document.body, { childList: true, subtree: true })
    }

    /*  Keep every label honest if the options change under us.               */
    document.addEventListener('change', function (e) {
      if (e.target && e.target.tagName === 'SELECT') syncLabel(e.target)
    })

    document.addEventListener('click', function (e) {
      if (!open || !open.trigger) return
      if (e.target.closest('.hd-select-menu') || e.target.closest('.hd-select-trigger')) return
      closeMenu()
    })

    document.addEventListener('keydown', function (e) {
      if (!open || !open.trigger) return
      var st = open
      switch (e.key) {
        case 'Escape':
          e.preventDefault()
          st.trigger.focus()
          closeMenu()
          break
        case 'Tab':
          closeMenu()
          break
        case 'ArrowDown':
          e.preventDefault()
          highlight(st, st.index + 1)
          break
        case 'ArrowUp':
          e.preventDefault()
          highlight(st, st.index - 1)
          break
        case 'Home':
          e.preventDefault()
          highlight(st, 0)
          break
        case 'End':
          e.preventDefault()
          highlight(st, st.items.length - 1)
          break
        case 'Enter':
        case ' ':
          if (st.items[st.index]) {
            e.preventDefault()
            commit(st, st.items[st.index])
          }
          break
        default:
          break
      }
    })

    window.addEventListener('resize', closeMenu)
    /*  Capture phase, so a page scroll still closes the menu -- but a scroll
        inside the list itself must not, or the list could never be used.   */
    window.addEventListener(
      'scroll',
      function (e) {
        if (open && open.panel && open.panel.contains(e.target)) return
        closeMenu()
      },
      true
    )
  }

  /*  Let the app re-sync a select it has just rewritten.                    */
  window.HD = window.HD || {}
  window.HD.upgradeSelects = upgradeAll

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init)
  } else {
    init()
  }
})()
