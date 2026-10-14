/* ==========================================================================
   HighDream product menu bar
   --------------------------------------------------------------------------
   A direct port of the navigation bar used by `static/omm-planner.html`:
   the bar lives just above the viewport, slides down when the pointer reaches
   the top edge, and can be pinned open.  Injected at runtime so every page
   shares one implementation and one visual language.

   Usage:  <script src="/static/hd-menu.js"></script>
   ========================================================================== */
;(function () {
  'use strict'

  var LS_PIN = 'hd-menu-pinned'
  var LS_THEME = 'darkTheme' /* shared with every page's own toggle button */
  var LS_RADIUS = 'hd-radius'
  var LS_FONT = 'hd-fscale'
  var MAX_RADIUS = 12
  /* omm-planner lets the interface font run 11–17px around a 13px base. */
  var BASE_FONT = 13
  var MIN_FONT = 11
  var MAX_FONT = 17

  /* Grouped exactly like omm-planner: label + a row of quiet text buttons. */
  var NAV = [
    {
      label: '视图',
      items: [
        { href: '/', text: '笔记本', icon: 'fas fa-book' },
        { href: '/matrix_panel', text: '矩阵面板', icon: 'fas fa-table' },
        { href: '/matrix_lab', text: '矩阵实验', icon: 'fas fa-flask' },
        { href: '/matrix_analysis', text: '矩阵分析', icon: 'fas fa-magnifying-glass' }
      ]
    },
    {
      label: '工具',
      items: [
        { href: '/blueprint_canvas', text: '蓝图', icon: 'fas fa-diagram-project' },
        { href: '/word_editor', text: '文档', icon: 'fas fa-file-lines' },
        { href: '/rule_select', text: '规则选择', icon: 'fas fa-hand-pointer' },
        { href: '/method_tree', text: '方法树', icon: 'fas fa-sitemap' }
      ]
    },
    {
      label: '参考',
      items: [{ href: '/help', text: '帮助', icon: 'fas fa-book-open' }]
    }
  ]

  /* ---------------------------------------------------------------- utils */

  function read(key, fallback) {
    try {
      var v = localStorage.getItem(key)
      return v === null ? fallback : v
    } catch (_) {
      return fallback
    }
  }

  function write(key, value) {
    try {
      localStorage.setItem(key, value)
    } catch (_) {}
  }

  function toast(msg) {
    var el = document.getElementById('hd-toast')
    if (!el) return
    el.textContent = msg
    el.classList.add('hd-show')
    clearTimeout(toast._t)
    toast._t = setTimeout(function () {
      el.classList.remove('hd-show')
    }, 1600)
  }

  function isDark() {
    return document.body.classList.contains('theme-dark')
  }

  function applyTheme(dark) {
    document.body.classList.toggle('theme-dark', dark)
    write(LS_THEME, dark ? 'true' : 'false')
    syncThemeBtn(dark)
  }

  /*  The bar's own toggle is the only one in the product, so it has to show
      the state it will switch away from: moon while light, sun while dark. */
  function syncThemeBtn(dark) {
    var btn = document.getElementById('hd-theme-btn')
    if (!btn) return
    btn.innerHTML = dark ? '<i class="fas fa-sun"></i>' : '<i class="fas fa-moon"></i>'
    btn.title = dark ? '\u5207\u6362\u5230\u6d45\u8272' : '\u5207\u6362\u5230\u6df1\u8272'
    btn.setAttribute('aria-label', btn.title)
  }

  function applyRadius(px) {
    document.documentElement.style.setProperty('--radius', px + 'px')
    write(LS_RADIUS, String(px))
  }

  function applyFont(px) {
    document.documentElement.style.setProperty('--fscale', String(px / BASE_FONT))
    write(LS_FONT, String(px))
  }

  function currentFont() {
    var raw = getComputedStyle(document.documentElement).getPropertyValue('--fscale') || '1'
    return Math.round(parseFloat(raw) * BASE_FONT) || BASE_FONT
  }

  /* ---------------------------------------------------------------- build */

  function buildMenu() {
    var path = location.pathname.replace(/\/+$/, '') || '/'

    var trigger = document.createElement('div')
    trigger.id = 'hd-menu-trigger'

    var menu = document.createElement('nav')
    menu.id = 'hd-top-menu'

    NAV.forEach(function (group) {
      var wrap = document.createElement('div')
      wrap.className = 'hd-menu-group'

      var label = document.createElement('span')
      label.className = 'hd-menu-label'
      label.textContent = group.label
      wrap.appendChild(label)

      group.items.forEach(function (item) {
        var href = item.href.replace(/\/+$/, '') || '/'
        var a = document.createElement('a')
        a.className = 'hd-menu-btn'
        a.href = item.href
        a.title = item.text
        if (href === path) a.classList.add('active')
        a.innerHTML = '<i class="fas ' + item.icon + '"></i><span>' + item.text + '</span>'
        wrap.appendChild(a)
      })

      menu.appendChild(wrap)
    })

    /* settings popover ---------------------------------------------------- */
    var styleBtn = document.createElement('button')
    styleBtn.type = 'button'
    styleBtn.className = 'hd-menu-btn'
    styleBtn.id = 'hd-style-btn'
    styleBtn.title = '样式'
    styleBtn.innerHTML = '<i class="fas fa-sliders"></i><span>样式</span>'

    var themeBtn = document.createElement('button')
    themeBtn.type = 'button'
    themeBtn.className = 'hd-menu-btn'
    themeBtn.id = 'hd-theme-btn'
    themeBtn.title = '切换主题'
    themeBtn.innerHTML = '<i class="fas fa-circle-half-stroke"></i>'

    var group = document.createElement('div')
    group.className = 'hd-menu-group'
    group.appendChild(styleBtn)
    group.appendChild(themeBtn)
    menu.appendChild(group)

    /* pin ----------------------------------------------------------------- */
    var pinAnchor = document.createElement('span')
    pinAnchor.className = 'hd-pin-anchor'
    menu.appendChild(pinAnchor)

    var pin = document.createElement('button')
    pin.type = 'button'
    pin.className = 'hd-pin-btn'
    pin.id = 'hd-pin-btn'
    pin.title = '固定菜单栏'
    pin.innerHTML =
      '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"' +
      ' stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 12 15 18 9"/></svg>'
    menu.appendChild(pin)

    /*  Adopt the page's own action buttons ------------------------------- */
    /*  Save / load / virtual-keyboard and friends used to live in the old   */
    /*  per-page #header bar.  There is no wrapper div left: each one is a   */
    /*  bare <button data-hd-action hidden> in the markup, found here and     */
    /*  moved into the bar as the 操作 group.  The nodes are moved, not      */
    /*  cloned, so every getElementById listener the page script attached    */
    /*  survives.  `hidden` is dropped on the way in, because otherwise the  */
    /*  button would arrive already invisible and the feature would have no  */
    /*  UI at all.                                                          */
    var lift = document.querySelectorAll('button[data-hd-action]')
    if (lift.length) {
      var actionGroup = document.createElement('div')
      actionGroup.className = 'hd-menu-group hd-menu-group-actions'
      var actionLabel = document.createElement('span')
      actionLabel.className = 'hd-menu-label'
      actionLabel.textContent = '\u64cd\u4f5c'
      actionGroup.appendChild(actionLabel)
      Array.prototype.forEach.call(lift, function (node) {
        node.removeAttribute('hidden')
        actionGroup.appendChild(node)
      })
      menu.insertBefore(actionGroup, pinAnchor)
    }

    /* dropdown ------------------------------------------------------------ */
    var dd = document.createElement('div')
    dd.className = 'hd-dropdown'
    dd.id = 'hd-dd-style'
    dd.innerHTML =
      '<div class="hd-dd-label">外观</div>' +
      '<button type="button" class="hd-dd-item" data-hd="theme">' +
      '<span>主题</span><span class="hd-hint" id="hd-v-theme">浅色</span></button>' +
      '<div class="hd-dd-row"><span>圆角</span>' +
      '<span class="hd-control">' +
      '<span class="hd-step-value" id="hd-v-radius">0</span>' +
      '<span class="hd-step-btn" data-hd="radius-down">−</span>' +
      '<span class="hd-step-btn" data-hd="radius-up">+</span>' +
      '</span></div>' +
      '<div class="hd-dd-row"><span>界面字体</span>' +
      '<span class="hd-control">' +
      '<span class="hd-step-value" id="hd-v-font">13px</span>' +
      '<span class="hd-step-btn" data-hd="font-down">−</span>' +
      '<span class="hd-step-btn" data-hd="font-up">+</span>' +
      '</span></div>' +
      '<div class="hd-dd-divider"></div>' +
      '<button type="button" class="hd-dd-item" data-hd="reset">' +
      '<span>重置样式</span><span class="hd-hint">R</span></button>'

    var toastEl = document.createElement('div')
    toastEl.id = 'hd-toast'

    document.body.appendChild(trigger)
    document.body.appendChild(menu)
    document.body.appendChild(dd)
    document.body.appendChild(toastEl)

    return { menu: menu, pin: pin, dd: dd, styleBtn: styleBtn }
  }

  /* ----------------------------------------------------------------- wire */

  function init() {
    if (document.getElementById('hd-top-menu')) return

    applyRadius(Math.min(MAX_RADIUS, Math.max(0, parseInt(read(LS_RADIUS, '0'), 10) || 0)))
    applyFont(
      Math.min(MAX_FONT, Math.max(MIN_FONT, parseInt(read(LS_FONT, String(BASE_FONT)), 10) || BASE_FONT))
    )
    applyTheme(read(LS_THEME, 'false') === 'true')
    syncThemeBtn(isDark())

    var el = buildMenu()
    var pinned = read(LS_PIN, '0') === '1'

    function syncPin() {
      el.pin.classList.toggle('hd-pinned', pinned)
      el.menu.classList.toggle('hd-pinned', pinned)
      document.body.classList.toggle('hd-menu-pinned', pinned)
    }

    function closeDd() {
      el.dd.classList.remove('hd-open')
      el.styleBtn.classList.remove('active')
      document.body.classList.remove('hd-submenu-open')
    }

    function syncDd() {
      document.getElementById('hd-v-theme').textContent = isDark() ? '深色' : '浅色'
      document.getElementById('hd-v-radius').textContent =
        getComputedStyle(document.documentElement).getPropertyValue('--radius').replace('px', '') || '0'
      document.getElementById('hd-v-font').textContent = currentFont() + 'px'
    }

    el.pin.addEventListener('click', function (e) {
      e.stopPropagation()
      pinned = !pinned
      write(LS_PIN, pinned ? '1' : '0')
      syncPin()
      toast(pinned ? '菜单栏已固定' : '菜单栏已隐藏')
    })

    el.styleBtn.addEventListener('click', function (e) {
      e.stopPropagation()
      var open = el.dd.classList.contains('hd-open')
      closeDd()
      if (open) return
      syncDd()
      el.dd.classList.add('hd-open')
      el.styleBtn.classList.add('active')
      document.body.classList.add('hd-submenu-open')
      var r = el.styleBtn.getBoundingClientRect()
      el.dd.style.top = r.bottom + 6 + 'px'
      el.dd.style.left = Math.max(8, Math.min(r.left, window.innerWidth - el.dd.offsetWidth - 10)) + 'px'
    })

    document.getElementById('hd-theme-btn').addEventListener('click', function () {
      applyTheme(!isDark())
      toast(isDark() ? '深色' : '浅色')
    })

    el.dd.addEventListener('click', function (e) {
      var target = e.target.closest('[data-hd]')
      if (!target) return
      var radius = parseInt(
        getComputedStyle(document.documentElement).getPropertyValue('--radius') || '0',
        10
      )
      var font = currentFont()
      switch (target.dataset.hd) {
        case 'theme':
          applyTheme(!isDark())
          break
        case 'radius-down':
          applyRadius(Math.max(0, radius - 2))
          break
        case 'radius-up':
          applyRadius(Math.min(MAX_RADIUS, radius + 2))
          break
        case 'font-down':
          applyFont(Math.max(MIN_FONT, font - 1))
          break
        case 'font-up':
          applyFont(Math.min(MAX_FONT, font + 1))
          break
        case 'reset':
          applyRadius(0)
          applyFont(BASE_FONT)
          applyTheme(false)
          break
        default:
          return
      }
      syncDd()
    })

    document.addEventListener('click', function (e) {
      if (!e.target.closest('.hd-dropdown') && !e.target.closest('#hd-style-btn')) closeDd()
    })

    document.addEventListener('keydown', function (e) {
      if (e.key === 'Escape') closeDd()
    })

    syncPin()
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init)
  } else {
    init()
  }
})()
