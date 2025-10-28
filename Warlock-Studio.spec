# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['Warlock-Studio.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('AI-onnx', 'AI-onnx'),  # <-- Añade esta línea
        ('Assets', 'Assets')    # <-- Añade esta línea
    ],
    hiddenimports=[
        'onnxruntime.capi._pybind_state',
        'onnxruntime.providers',
        'moviepy.editor'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Frameworks de Machine Learning pesados
        'torch', 'torchaudio', 'torchvision', 'transformers', 'accelerate', 'diffusers',
        'lightning', 'pytorch-lightning',

        # Librerías de Ciencia de Datos y Gráficos
        'matplotlib', 'pandas', 'scipy', 'scikit-learn', 'scikit-image',

        # Otros Toolkits de GUI
        'PyQt5', 'PyQt5-Qt5', 'PyQt5_sip', 'dearpygui', 'QtAwesome', 'QtPy',

        # Herramientas de desarrollo y testing
        'pytest', 'unittest', 'poetry', 'virtualenv',

        # Compiladores y librerías de bajo nivel que no se usan
        'numba', 'llvmlite',

        # Otros paquetes grandes no relacionados
        'pygame', 'PyMuPDF', 'requests', 'aiohttp', 'httpx', 'fonttools', 'GitPython', 'musicbrainzngs', 'mido', 'rtmidi', 'simpleaudio', 'vulkan', 'vgamepad'
    ],
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
    name='Warlock-Studio',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    icon='logo.ico',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Warlock-Studio',
)
