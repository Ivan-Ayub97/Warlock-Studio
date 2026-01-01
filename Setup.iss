; =========================================================
; WARLOCK-STUDIO – INSTALLER SCRIPT (STABLE)
; =========================================================

; ---------------------------------------------------------
; FIRMA DIGITAL
; ---------------------------------------------------------
#define SignToolPath "C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64\signtool.exe"
#define CertPath "-------------------------------"
#define CertPass "******"

; ---------------------------------------------------------
; DEFINICIONES
; ---------------------------------------------------------
#define AppName "Warlock-Studio"
#define AppVersion "5.1.1"
#define AppPublisher "Ivan-Ayub97"
#define AppURL "https://github.com/Ivan-Ayub97/Warlock-Studio"
#define AppExeName "Warlock-Studio.exe"
#define SourcePath "..\Warlock-Studio-main"

; =========================================================
; SETUP
; =========================================================
[Setup]

AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL={#AppURL}
AppSupportURL={#AppURL}
AppUpdatesURL={#AppURL}

AppId={{***************************************}

; ---- INSTALACIÓN EN DOCUMENTOS ----
DefaultDirName={userdocs}\{#AppName}
DefaultGroupName={#AppName}
AllowNoIcons=yes
PrivilegesRequired=lowest

; ---- ASISTENTE ----
WizardStyle=modern
DisableWelcomePage=no
DisableDirPage=no
DisableProgramGroupPage=no

; ---- SALIDA ----
OutputDir=Output
OutputBaseFilename=Warlock-Studio-{#AppVersion}-Full-Installer
Compression=lzma2/max
SolidCompression=yes
SetupLogging=yes

; ---- ESTÉTICA ----
SetupIconFile={#SourcePath}\logo.ico
UninstallDisplayIcon={app}\{#AppExeName}
WizardImageFile={#SourcePath}\Assets\wizard-image.bmp
WizardSmallImageFile={#SourcePath}\Assets\wizard-small.bmp

; ---- FIRMA ----
SignTool=MySignTool

; =========================================================
; LANGUAGES (SEGURO)
; =========================================================
[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"; LicenseFile: "{#SourcePath}\Assets\License.txt"
; Name: "spanish"; MessagesFile: "compiler:Languages\Spanish.isl"; LicenseFile: "{#SourcePath}\Assets\License.txt"

; =========================================================
; TASKS (OPCIONES DEL USUARIO)
; =========================================================
[Tasks]
Name: "desktopicon"; Description: "Create a desktop icon"; GroupDescription: "Additional options:"; Flags: unchecked
Name: "autorun"; Description: "Run Warlock-Studio when Windows starts"; GroupDescription: "Startup:"; Flags: unchecked
Name: "userdata"; Description: "Create user data folder in Documents"; GroupDescription: "Data:"; Flags: checkedonce

; =========================================================
; FILES
; =========================================================
[Files]

Source: "{#SourcePath}\{#AppExeName}"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#SourcePath}\logo.ico"; DestDir: "{app}"; Flags: ignoreversion

Source: "{#SourcePath}\_internal\*"; DestDir: "{app}\_internal"; \
Flags: ignoreversion recursesubdirs createallsubdirs

Source: "{#SourcePath}\Assets\*"; DestDir: "{app}\Assets"; \
Flags: ignoreversion recursesubdirs createallsubdirs

; =========================================================
; ICONS
; =========================================================
[Icons]

Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExeName}"
Name: "{group}\Uninstall {#AppName}"; Filename: "{uninstallexe}"

Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; \
Tasks: desktopicon

; =========================================================
; REGISTRY
; =========================================================
[Registry]

Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\Run"; \
ValueType: string; ValueName: "{#AppName}"; \
ValueData: """{app}\{#AppExeName}"""; Tasks: autorun

; =========================================================
; RUN
; =========================================================
[Run]

Filename: "{app}\{#AppExeName}"; \
Description: "Launch {#AppName}"; \
Flags: nowait postinstall skipifsilent

; =========================================================
; UNINSTALL
; =========================================================
[UninstallDelete]
Type: filesandordirs; Name: "{app}\_internal"
Type: filesandordirs; Name: "{app}\Assets"
Type: filesandordirs; Name: "{userdocs}\{#AppName}\UserData"

; =========================================================
; CODE
; =========================================================
[Code]

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if (CurStep = ssInstall) and WizardIsTaskSelected('userdata') then
  begin
    ForceDirectories(ExpandConstant('{userdocs}\{#AppName}\UserData'));
  end;
end;

[SignTools]
Name: "MySignTool"; Command: """{#SignToolPath}"" sign /f ""{#CertPath}"" /p ""{#CertPass}"" /fd sha256 /tr http://timestamp.digicert.com /td sha256 $f"
