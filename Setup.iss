; ===================================================================
;   Warlock-Studio 4.0.1 - Inno Setup Script with Online Download
; ===================================================================

#define AppName "Warlock-Studio"
#define AppVersion "4.1"
#define AppPublisher "Iván Eduardo Chavez Ayub"
#define AppURL "https://github.com/Ivan-Ayub97/Warlock-Studio"
#define AppExeName "Warlock-Studio.exe"

; URLs para descargar los componentes AI (ajustar según tu servidor)
#define AIModelsURL "https://github.com/Ivan-Ayub97/Warlock-Studio/releases/download/4.0/AI-onnx.zip"
#define AIModelsSize "327000000"  ; Tamaño aproximado en bytes

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
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "downloadai"; Description: "Download AI Models (Required - 327MB)"; GroupDescription: "Components"; Flags: checkedonce

[Files]
; --- Archivos Básicos de la Aplicación (SIN AI-onnx) ---
Source: "..\Warlock-Studio\{#AppExeName}"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\Warlock-Studio\logo.ico"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\Warlock-Studio\Assets\*"; DestDir: "{app}\Assets"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\Warlock-Studio\LICENSE"; DestDir: "{app}"; DestName: "License.txt"; Flags: ignoreversion
Source: "..\Warlock-Studio\NOTICE.md"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExeName}"; IconFilename: "{app}\logo.ico"; WorkingDir: "{app}"
Name: "{group}\{cm:UninstallProgram,{#AppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; IconFilename: "{app}\logo.ico"; WorkingDir: "{app}"; Tasks: desktopicon

[Code]
var
  DownloadPage: TDownloadWizardPage;

function OnDownloadProgress(const Url, FileName: String; const Progress, ProgressMax: Int64): Boolean;
begin
  if Progress = ProgressMax then
    Log(Format('Successfully downloaded %s', [FileName]));
  Result := True;
end;

procedure InitializeWizard;
begin
  // Create the pages
  DownloadPage := CreateDownloadPage(SetupMessage(msgWizardPreparing), SetupMessage(msgPreparingDesc), @OnDownloadProgress);
end;

function NextButtonClick(CurPageID: Integer): Boolean;
begin
  if CurPageID = wpReady then begin
    if IsTaskSelected('downloadai') then begin
      DownloadPage.Clear;
      DownloadPage.Add('{#AIModelsURL}', 'AI-onnx.zip', '');
      DownloadPage.Show;
      try
        try
          DownloadPage.Download; // This downloads the file(s)
          Result := True;
        except
          if DownloadPage.AbortedByUser then
            Log('Aborted by user.')
          else
            SuppressibleMsgBox(AddPeriod(GetExceptionMessage), mbCriticalError, MB_OK, IDOK);
          Result := False;
        end;
      finally
        DownloadPage.Hide;
      end;
    end else
      Result := True;
  end else
    Result := True;
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  ZipPath, ExtractPath: String;
  ResultCode: Integer;
begin
  if CurStep = ssPostInstall then begin
    if IsTaskSelected('downloadai') then begin
      // Extraer el archivo ZIP descargado
      ZipPath := ExpandConstant('{tmp}\AI-onnx.zip');
      ExtractPath := ExpandConstant('{app}\AI-onnx');
      
      if FileExists(ZipPath) then begin
        // Crear carpeta destino
        CreateDir(ExtractPath);
        
        // Extraer usando PowerShell (disponible en Windows 10+)
        if Exec('powershell.exe', 
               '-Command "Expand-Archive -Path ''' + ZipPath + ''' -DestinationPath ''' + ExtractPath + ''' -Force"', 
               '', SW_HIDE, ewWaitUntilTerminated, ResultCode) then begin
          Log('AI models extracted successfully');
          DeleteFile(ZipPath); // Limpiar archivo temporal
        end else begin
          MsgBox('Error extracting AI models. Please download manually from: ' + '{#AIModelsURL}', 
                 mbError, MB_OK);
        end;
      end;
    end;
  end;
end;

[Run]
Filename: "{app}\{#AppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(AppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}\AI-onnx"
Type: filesandordirs; Name: "{app}\Assets"

[Messages]
english.BeveledLabel=AI Models will be downloaded during installation (327MB)
