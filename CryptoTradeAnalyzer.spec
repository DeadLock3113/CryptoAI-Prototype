# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

added_files = [
    ('web/templates', 'web/templates'),
    ('web/static', 'web/static'),
    ('models', 'models'),
    ('strategies', 'strategies'),
    ('indicators', 'indicators'),
    ('utils', 'utils'),
    ('data', 'data'),
    ('backtesting', 'backtesting')
]

a = Analysis(
    ['start.py'],
    pathex=[],
    binaries=[],
    datas=added_files,
    hiddenimports=[
        'matplotlib.backends.backend_tkagg',
        'sklearn.utils._cython_blas',
        'sklearn.neighbors.typedefs',
        'sklearn.neighbors.quad_tree',
        'sklearn.tree._utils',
        'sqlalchemy.sql.default_comparator',
        'pandas._libs.tslibs.timedeltas',
        'pandas._libs.tslibs.nattype',
        'pandas._libs.tslibs.np_datetime',
        'nltk',
        'gensim',
        'torch',
        'torchvision',
        'tensorflow',
        'flask_sqlalchemy',
        'flask_login',
        'flask_wtf',
        'email_validator',
        'gunicorn.workers',
        'trading_calendar',
        'backtrader',
        'trafilatura',
        'textblob',
        'vadersentiment',
        'vadersentiment.vaderSentiment',
        'werkzeug.security',
        'training_handler',
        'sqlalchemy.dialects.postgresql'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='CryptoTradeAnalyzer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='web/static/images/logo.ico',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='CryptoTradeAnalyzer',
)
