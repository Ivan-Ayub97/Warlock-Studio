# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    # 1. Aquí agregamos los dos archivos nuevos a la lista de scripts
    ['Warlock-Studio.py', 'drag_drop.py', 'console.py', 'warlock_preferences.py', 'file_queue_manager.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('AI-onnx', 'AI-onnx'),
        ('Assets', 'Assets')
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
        'pygame', 'PyMuPDF', 'aiohttp', 'httpx', 'GitPython', 'musicbrainzngs', 'mido', 'rtmidi', 'simpleaudio', 'vulkan', 'vgamepad'
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
    console=False,
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
