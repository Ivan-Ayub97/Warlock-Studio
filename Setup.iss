; ===================================================================
;  Warlock-Studio 4.2.1 - Inno Setup Script (Full Offline Installer)
; ===================================================================

#define AppName "Warlock-Studio"
#define AppVersion "4.2.1"
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
; CHANGED: Filename reflects it's a full/offline installer
OutputBaseFilename=Warlock-Studio-{#AppVersion}-Full-Installer
SetupIconFile=..\Warlock-Studio\logo.ico
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern

; --- Imágenes del Asistente ---
WizardImageFile=..\Warlock-Studio\Assets\wizard-image.bmp
WizardSmallImageFile=..\Warlock-Studio\Assets\wizard-small.bmp
UninstallDisplayIcon={app}\{#AppExeName}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"; LicenseFile: "..\Warlock-Studio\License.txt"

[Tasks]
; The download task has been removed as all files are now included.
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; --- Archivos Básicos de la Aplicación ---
Source: "..\Warlock-Studio\{#AppExeName}"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\Warlock-Studio\logo.ico"; DestDir: "{app}"; Flags: ignoreversion

; --- CHANGED: Package the entire '_internal' folder and its contents ---
Source: "..\Warlock-Studio\_internal\*"; DestDir: "{app}\_internal"; Flags: recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExeName}"; IconFilename: "{app}\logo.ico"; WorkingDir: "{app}"
Name: "{group}\{cm:UninstallProgram,{#AppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; IconFilename: "{app}\logo.ico"; WorkingDir: "{app}"; Tasks: desktopicon

; --- REMOVED: The entire [Code] section for downloading is no longer needed. ---

[Run]
Filename: "{app}\{#AppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(AppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; This section is still needed to clean up the installed folder on uninstall.
Type: filesandordirs; Name: "{app}\_internal"
Type: filesandordirs; Name: "{app}\Assets"

; --- REMOVED: [Messages] section related to downloading is no longer needed. ---