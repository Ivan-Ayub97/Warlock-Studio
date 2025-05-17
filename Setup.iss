[Setup]
; Basic installation configuration
AppName=Warlock-Studio  
AppVersion=1.0
DefaultDirName={pf}\Warlock-Studio  
DefaultGroupName=Warlock-Studio  
OutputDir=.\Output
OutputBaseFilename=Warlock-Studio_Installer
SetupIconFile=C:\Users\negro\Desktop\Warlock-Studio1.0\logo.ico
Compression=lzma
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=admin

[Files]
; Files to include in the installation
Source: "Warlock-Studio.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "C:\Users\negro\Desktop\Warlock-Studio1.0\logo.ico"; DestDir: "{app}"; Flags: ignoreversion
Source: "C:\Users\negro\Desktop\Warlock-Studio1.0\AI-onnx"; DestDir: "{app}\AI-onnx"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "C:\Users\negro\Desktop\Warlock-Studio1.0\Assets"; DestDir: "{app}\Assets"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
; Create shortcuts in the menu group and on the desktop
Name: "{group}\Warlock-Studio"; Filename: "{app}\Warlock-Studio.exe"; IconFilename: "{app}\logo.ico"; WorkingDir: "{app}"
Name: "{commondesktop}\Warlock-Studio"; Filename: "{app}\Warlock-Studio.exe"; IconFilename: "{app}\logo.ico"; WorkingDir: "{app}"

[Registry]
; Associate MedIA-Witch with .mp4 files
Root: HKCU; Subkey: "Software\Classes\.mp4"; ValueType: string; ValueData: "Warlock-Studio.File"
Root: HKCU; Subkey: "Software\Classes\Warlock-Studio.File\shell\open\command"; ValueType: string; ValueData: """{app}\\Warlock-Studio.exe"" ""%1"""

[Code]
function ShowCustomLicensePage(): Boolean;
begin
MsgBox('*** TERMS AND CONDITIONS ***'#13#10#13#10 +
       'Warlock-Studio is an open-source application that unifies the MedIA-Wizard and MedIA-Witch tools—software created by Iván E.C. Ayub ("Ivan-Ayub97"), based on and inspired by QualityScaler and RealScaler, originally developed by Djfrag. Its primary purpose is to enhance image resolution using advanced artificial intelligence models.'#13#10#13#10 +
              '*** TECHNOLOGIES USED ***'#13#10 +
       'This software is distributed under the MIT License and incorporates multiple third-party technologies, whose rights and credits remain the property of their respective authors.'#13#10#13#10 +
       ' - Python (Python Software Foundation)'#13#10 +
       ' - ONNX Runtime (Microsoft)'#13#10 +
       ' - Real-ESRGAN (Xintao Wang et al.)'#13#10 +
       ' - SRGAN (Ledig et al.)'#13#10 +
       ' - BSRGAN (Zhang et al.)'#13#10 +
       ' - IRCNN (Kai Zhang et al.)'#13#10 +
       ' - FFmpeg (FFmpeg Team)'#13#10 +
       ' - OpenGL (Khronos Group)'#13#10 +
       ' - PyInstaller (Giovanni Bajo et al.)'#13#10 +
       ' - Inno Setup (Jordan Russell)'#13#10#13#10 +
       '*** DISCLAIMER ***'#13#10 +
       'The developer, Iván E.C. Ayub (Ivan-Ayub97), along with the contributors to the WarlockHub project, disclaims any liability for direct, indirect, incidental, or consequential damages resulting from the use or inability to use this application.'#13#10 +
       'This software is provided "as is", without any warranties of any kind, either express or implied, including but not limited to warranties of merchantability, fitness for a particular purpose, or non-infringement.'#13#10 +
       'Additionally, the original authors are not responsible for any issues, modifications, or consequences arising from the use of this software.'#13#10#13#10 +
       'By installing or using this software, you acknowledge that you have read, understood, and agreed to these terms and conditions.'#13#10 +
       'If you do not agree, please close this window and cancel the installation.'#13#10#13#10 +
       'For more details, refer to the official license documentation.',
       mbInformation, MB_OK);
  
  Result := True;  // Continuar con la instalación
end;

procedure InitializeWizard();
begin
  ShowCustomLicensePage();  // Muestra la página de licencia
end;