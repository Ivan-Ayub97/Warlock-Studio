; ===================================================================
;   Warlock-Studio 2.2 - Inno Setup Script
; ===================================================================

#define AppName "Warlock-Studio"
#define AppVersion "2.2"
#define AppPublisher "Iván Eduardo Chavez Ayub"
#define AppURL "https://github.com/Ivan-Ayub97/Warlock-Studio"
#define AppExeName "Warlock-Studio.exe"

[Setup]
; --- Configuración Básica ---
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
AppSupportURL={#AppURL}
AppUpdatesURL={#AppURL}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
AllowNoIcons=yes
PrivilegesRequired=admin
ArchitecturesInstallIn64BitMode=x64

; --- Configuración del Instalador ---
OutputDir=Output
OutputBaseFilename=Warlock-Studio-{#AppVersion}-Installer
SetupIconFile=..\Warlock-Studio\logo.ico
Compression=lzma2
SolidCompression=yes
WizardStyle=modern

; --- Imágenes del Asistente ---
; NOTA: Asegúrate de que las rutas relativas sean correctas desde la ubicación de este script.
WizardImageFile=..\Warlock-Studio\Assets\wizard-image.bmp
WizardSmallImageFile=..\Warlock-Studio\Assets\wizard-small.bmp
UninstallDisplayIcon={app}\{#AppExeName}

; CORRECTO: LicenseFile se define ahora en la sección [Languages]
; para que cada idioma muestre su propia licencia.

[Languages]
; --- Definición de Idiomas y Licencias ---
; NOTA: Asegúrate de tener los archivos de licencia en la ruta especificada.
Name: "english"; MessagesFile: "compiler:Default.isl"; LicenseFile: "..\Warlock-Studio\License.txt"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; --- Archivos de la Aplicación ---
; NOTA: La estructura de carpetas asumida es:
;  - Proyecto/
;    - InnoSetup_Script/ (Aquí va este archivo .iss)
;    - Warlock-Studio/   (Aquí van los archivos de la aplicación)
Source: "..\Warlock-Studio\{#AppExeName}"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\Warlock-Studio\logo.ico"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\Warlock-Studio\AI-onnx\*"; DestDir: "{app}\AI-onnx"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\Warlock-Studio\Assets\*"; DestDir: "{app}\Assets"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\Warlock-Studio\rsc\*"; DestDir: "{app}\rsc"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\Warlock-Studio\LICENSE"; DestDir: "{app}"; DestName: "License.txt"; Flags: ignoreversion
Source: "..\Warlock-Studio\NOTICE.md"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
; --- Accesos Directos ---
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExeName}"; IconFilename: "{app}\logo.ico"; WorkingDir: "{app}"
Name: "{group}\{cm:UninstallProgram,{#AppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; IconFilename: "{app}\logo.ico"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(AppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; --- Limpieza Adicional Durante la Desinstalación ---
Type: filesandordirs; Name: "{app}\AI-onnx"
Type: filesandordirs; Name: "{app}\Assets"
Type: filesandordirs; Name: "{app}\rsc"