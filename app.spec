# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['app.py'],
    pathex=['C:\\Users\\skhalil\\Documents\\Application Web'],
    binaries=[],
    datas=[
        ('C:\\Users\\skhalil\\Documents\\Application Web\\model_trained.pth', '.'),
        ('C:\\Users\\skhalil\\Documents\\Application Web\\templates\\*', 'templates'),
        ('C:\\Users\\skhalil\\Documents\\Application Web\\static\\*', 'static'),
        (r'C:\\Users\\skhalil\\Documents\\Application Web\\static\\css\\*', 'static\\css'),
        (r'C:\\Users\\skhalil\\Documents\\Application Web\\static\\images\\*', 'static\\images'),
        ('C:\\Users\\skhalil\\Documents\\Application Web\\uploads\\*','uploads'),
        ('C:\\Users\\skhalil\\Documents\\Application Web\\categories.json','categories.json')
   

    ],
    hiddenimports=[
        'scikit-learn',
        'pandas',
        'numpy',
        'torch',
        'torchvision',
        'torch.utils.tensorboard',
        'PIL.Image',
        'openpyxl',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='app',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
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

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='app',
)
