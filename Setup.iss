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
#define AppVersion "6.0"
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

AppId={{D1168ED1-6227-441F-8B88-EE6DBD45F336}

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
SetupIconFile= C:\Users\negro\Desktop\Warlock-Studio-main\Warlock-Studio\logo.ico
UninstallDisplayIcon={app}\{#AppExeName}
WizardImageFile= C:\Users\negro\Desktop\Warlock-Studio-main\Warlock-Studio\Assets\wizard-image.bmp
WizardSmallImageFile=C:\Users\negro\Desktop\Warlock-Studio-main\Warlock-Studio\Assets\wizard-small.bmp

; ---- FIRMA ----
SignTool=MySignTool

; =========================================================
; LANGUAGES (SEGURO)
; =========================================================
[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"; LicenseFile: "C:\Users\negro\Desktop\Warlock-Studio-main\Warlock-Studio\Assets\License.txt"

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

Source: "C:\Users\negro\Desktop\Warlock-Studio-main\Warlock-Studio\{#AppExeName}"; DestDir: "{app}"; Flags: ignoreversion
Source: "C:\Users\negro\Desktop\Warlock-Studio-main\Warlock-Studio\logo.ico"; DestDir: "{app}"; Flags: ignoreversion

Source: "C:\Users\negro\Desktop\Warlock-Studio-main\Warlock-Studio\_internal\*"; DestDir: "{app}\_internal"; \
Flags: ignoreversion recursesubdirs createallsubdirs

Source: "C:\Users\negro\Desktop\Warlock-Studio-main\Warlock-Studio\Assets\*"; DestDir: "{app}\Assets"; \
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
