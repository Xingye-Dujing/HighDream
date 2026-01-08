# -*- mode: python ; coding: utf-8 -*-
# 使用命令 pyinstaller app_local.spec 进行打包


a = Analysis(
    ['app_local.py'],
    pathex=['F:\python_project\HighDream'],
    binaries=[],
    datas=[('static', 'static'), ('templates', 'templates')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # Exclude libraries not used in the exe
    excludes=['tkinter', 'matplotlib', 'IPython', 'jupyter'],
    noarchive=False,
    # Optimization level can only be set to 1, setting it to 2 will cause problems
    optimize=1,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='HighDreamWeb',
    debug=False,
    bootloader_ignore_signals=False,
    strip=True,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
