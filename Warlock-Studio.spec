# -*- mode: python ; coding: utf-8 -*-
# PyInstaller SPEC file for Warlock-Studio - actualizado y optimizado

import os

# Obtener el directorio base
current_dir = os.path.dirname(os.path.abspath(SPEC))

# Análisis de los archivos fuente
a = Analysis(
    ['Warlock-Studio.py', 'model_downloader.py'],
    pathex=[current_dir],
    binaries=[],
    datas=[
        ('Assets', 'Assets'),  # Incluir carpeta de recursos
        ('model_downloader.py', '.'),  # Asegurar descarga de modelos
    ],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=1,  # Nivel medio para evitar errores con numpy
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
)

# Crear archivo PYZ (bytecode comprimido)
pyz = PYZ(a.pure, a.zipped_data, cipher=None)

# Crear el ejecutable EXE
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='Warlock-Studio',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # ⬅ Oculta consola
    disable_windowed_traceback=False,
    icon='logo.ico'  # Asegúrate de que el ícono exista en esta ruta
)
