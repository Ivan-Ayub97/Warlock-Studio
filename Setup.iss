[Setup]
; Basic installation configuration
AppName=Warlock-Studio
AppVersion=2.0
DefaultDirName={pf}\Warlock-Studio  
DefaultGroupName=Warlock-Studio  
OutputDir=.\Output
OutputBaseFilename=Warlock-Studio_Installer
SetupIconFile=C:\Users\negro\Desktop\Warlock-Studio\logo.ico
Compression=lzma
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=admin

[Files]
; Files to include in the installation
Source: "Warlock-Studio.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "C:\Users\negro\Desktop\Warlock-Studio\Assets\logo.ico"; DestDir: "{app}"; Flags: ignoreversion
Source: "C:\Users\negro\Desktop\Warlock-Studio\AI-onnx"; DestDir: "{app}\AI-onnx"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "C:\Users\negro\Desktop\Warlock-Studio\Assets"; DestDir: "{app}\Assets"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "C:\Users\negro\Desktop\Warlock-Studio\rsc"; DestDir: "{app}\rsc"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
; Create shortcuts in the menu group and on the desktop
Name: "{group}\Warlock-Studio"; Filename: "{app}\Warlock-Studio.exe"; IconFilename: "{app}\logo.ico"; WorkingDir: "{app}"
Name: "{commondesktop}\Warlock-Studio"; Filename: "{app}\Warlock-Studio.exe"; IconFilename: "{app}\logo.ico"; WorkingDir: "{app}"

[Registry]
; Associate Warlock-Studio with files
Root: HKCU; Subkey: "Software\Classes\.tif"; ValueType: string; ValueData: "Warlock-Studio.File"
Root: HKCU; Subkey: "Software\Classes\.bmp"; ValueType: string; ValueData: "Warlock-Studio.File"
Root: HKCU; Subkey: "Software\Classes\.webm"; ValueType: string; ValueData: "Warlock-Studio.File"
Root: HKCU; Subkey: "Software\Classes\.heic"; ValueType: string; ValueData: "Warlock-Studio.File"
Root: HKCU; Subkey: "Software\Classes\.fiv"; ValueType: string; ValueData: "Warlock-Studio.File"
Root: HKCU; Subkey: "Software\Classes\.avi"; ValueType: string; ValueData: "Warlock-Studio.File"
Root: HKCU; Subkey: "Software\Classes\.gif"; ValueType: string; ValueData: "Warlock-Studio.File"
Root: HKCU; Subkey: "Software\Classes\.mp4"; ValueType: string; ValueData: "Warlock-Studio.File"
Root: HKCU; Subkey: "Software\Classes\.mov"; ValueType: string; ValueData: "Warlock-Studio.File"
Root: HKCU; Subkey: "Software\Classes\.mkv"; ValueType: string; ValueData: "Warlock-Studio.File"
Root: HKCU; Subkey: "Software\Classes\.png"; ValueType: string; ValueData: "Warlock-Studio.File"
Root: HKCU; Subkey: "Software\Classes\.jpg"; ValueType: string; ValueData: "Warlock-Studio.File"
Root: HKCU; Subkey: "Software\Classes\.mpg"; ValueType: string; ValueData: "Warlock-Studio.File"

Root: HKCU; Subkey: "Software\Classes\Warlock-Studio.File\shell\open\command"; ValueType: string; ValueData: """{app}\\Warlock-Studio.exe"" ""%1"""

[Code]
function ShowCustomLicensePage(): Boolean;
begin
  MsgBox('© 2025 Iván Eduardo Chavez Ayub'#13#10 +
         'Licensed under the MIT License. Additional conditions are described in the NOTICE file.'#13#10#13#10 +

         'This software, Warlock-Studio, is distributed under the MIT License and extended with an additional NOTICE file.'#13#10 +
         'By installing or using this software, you agree to comply with both the MIT License and the additional terms specified in the NOTICE document.'#13#10#13#10 +

         '*** PROJECT OVERVIEW ***'#13#10 +
         'Warlock-Studio unifies the MedIA-Wizard and MedIA-Witch tools. It is developed by Iván Eduardo Chavez Ayub ("Ivan-Ayub97"), and is inspired by tools such as QualityScaler, FluidFrames, and RealScaler (originally developed by Djdefrag).'#13#10 +
         'Its main goal is to improve image resolution using AI-powered models with an intuitive interface.'#13#10#13#10 +

         '*** INTEGRATED TECHNOLOGIES & LICENSES ***'#13#10 +
         ' - QualityScaler, RealScaler, FluidFrames: MIT License (Djdefrag)'#13#10 +
         ' - Real-ESRGAN, RealESRGAN-G, RealESR-Anime, RealESR-Net: BSD 3-Clause / Apache 2.0 (Xintao Wang)'#13#10 +
         ' - RIFE: Apache 2.0 (hzwer, Megvii Research)'#13#10 +
         ' - SRGAN: CC BY-NC-SA 4.0 (TensorLayer Community)'#13#10 +
         ' - BSRGAN: Apache 2.0 (Kai Zhang)'#13#10 +
         ' - IRCNN: BSD / Mixed (Kai Zhang)'#13#10 +
         ' - Anime4K: MIT License (Tianyang Zhang / bloc97)'#13#10 +
         ' - ONNX Runtime: MIT License (Microsoft)'#13#10 +
         ' - PyTorch: BSD 3-Clause (Meta AI)'#13#10 +
         ' - FFmpeg: LGPL-2.1 / GPL (FFmpeg Team)'#13#10 +
         ' - ExifTool: Perl Artistic License (Phil Harvey)'#13#10 +
         ' - DirectML: MIT License (Microsoft)'#13#10 +
         ' - Python: PSF License (Python Software Foundation)'#13#10 +
         ' - PyInstaller: GPLv2+ (PyInstaller Team)'#13#10 +
         ' - Inno Setup: Custom Inno License (Jordan Russell)'#13#10#13#10 +

         '*** LIMITATION OF LIABILITY ***'#13#10 +
         'This software is provided "AS IS", without warranty of any kind, express or implied. The author and contributors are not liable for any damages, data loss, or consequences resulting from its use, misuse, or failure.'#13#10 +
         'Use of this software in critical or commercial systems is at your own risk.'#13#10#13#10 +

         '*** INTELLECTUAL PROPERTY NOTICE ***'#13#10 +
         'All original branding belong to Iván Eduardo Chavez Ayub. The name "Warlock-Studio" and associated logos may not be used commercially without express written permission.'#13#10#13#10 +

         'By continuing, you acknowledge that you have read, understood, and accepted these terms and conditions.'#13#10 +
         'If you do not agree, please cancel the installation.'#13#10#13#10 +
         'Refer to the LICENSE and NOTICE files for full legal terms.',
         mbInformation, MB_OK);

  Result := True;
end;

procedure InitializeWizard();
begin
  ShowCustomLicensePage();  // Muestra la página de licencia
end;