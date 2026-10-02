; PulseX Z-Image - the installer (Inno Setup 6.5 or later, https://jrsoftware.org/isinfo.php). 2026-10-02.
; For people who do not want to clone a repository, install Python and fetch model files by hand: one file, next,
; next, and the window is there with its model.
;
; * What goes in: the window (bin\pulsex-zimage.exe), the chain it runs (zimage-make, zimage-encode,
;   zimage-dit-stream, taef1_decode, the ggml / llama DLLs, the NPU module libggml-htp-v7x.so and its catalog), the
;   word list cues.tsv and a ready zimage.json. No Python, none of the command-line tools.
; * The picture model is NOT in the setup file: it is downloaded during the installation, from the place it is
;   published (Hugging Face), each file checked against its SHA-256. The addresses name one fixed revision, so a
;   setup file keeps installing the files it was tested with.
;     z-image-turbo-ultrareal06-q4_0.gguf  3.47 GB   Pexqman/Z-Image-Turbo-Q4_0-GGUF   (Apache 2.0)
;     qwen3-4b-zimage-q4_0.gguf            2.37 GB   Pexqman/Z-Image-Turbo-Q4_0-GGUF   (Apache 2.0)
;     taef1 diffusion_pytorch_model.safetensors 10 MB madebyollin/taef1                (MIT)
;   "Program only" skips the download - for someone who has the files: they go in <program folder>\models.
; * Per user, no administrator. The NPU driver loads its files only from a folder whose path is plain A-Z, so when
;   the user's own program folder is not (a user name with a letter such as an a with a ring), the suggestion is a
;   folder at the root of the system drive, and a folder with other letters is not accepted.
; * The first picture writes the model once more in the NPU's own layout beside it (.hexpack, 5.4 GB together);
;   the uninstaller removes those, the downloaded model and the program. Pictures (Pictures\PulseX Z-Image) and the
;   window's own notes (%LOCALAPPDATA%\PulseX Z-Image) are the user's and stay.
; * The enlarging (2x, 4x) needs two things that are not in the repository and are packed only when the build names
;   them: /DQnn=<folder> - QnnHtp.dll, QnnSystem.dll and QnnHtpV73Stub.dll of Qualcomm's AI runtime 2.46 - with
;   /DQnnSkel=<folder> - the runtime's DSP-side module libQnnHtpV73Skel.so and its catalog libqnnhtpv73.cat, without
;   which the runtime ends in 0x80000406 on a PC where nobody set ADSP_LIBRARY_PATH (Qualcomm's license allows these
;   files as part of an application, not on their own: see ABOUT-QNN.txt) - and /DUpscaler=<file> -
;   QuickSRNet-Large x4 compiled for 512 x 512 (BSD-3-Clause, LICENSE-QuickSRNet.txt). Without them the setup
;   installs a window that makes pictures at their own size.
; * ARM64 only.
;
;   ISCC.exe installer\pulsex_zimage.iss                  from a clone: packs the clone, writes installer\output\
;   ISCC.exe /DSrc=<folder> /DOut=<folder> /DAppVersion=0.10 installer\pulsex_zimage.iss

#ifndef Src
  #define Src SourcePath + "\.."
#endif
#ifndef Out
  #define Out SourcePath + "\output"
#endif
#ifndef AppVersion
  #define AppVersion "0.10"
#endif
#define AppName "PulseX Z-Image"
#define ModelRepo "https://huggingface.co/Pexqman/Z-Image-Turbo-Q4_0-GGUF/resolve/444474db53a7aaaee5c88a5a14b414b86f8f4a67"
#define Taef1Repo "https://huggingface.co/madebyollin/taef1/resolve/b1b2d00e9e440cfbf3dedb34266864da86016ceb"

[Setup]
AppId={{89770219-8BCB-41F9-8B14-D2C831DC1F28}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher=PulseCore
AppPublisherURL=https://github.com/Erik-matrix/pulsex-zimage
AppSupportURL=https://github.com/Erik-matrix/pulsex-zimage
AppUpdatesURL=https://github.com/Erik-matrix/pulsex-zimage/releases/latest
DefaultDirName={code:DefaultDir}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
AllowNoIcons=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=arm64
ArchitecturesInstallIn64BitMode=arm64
LicenseFile={#Src}\LICENSE
OutputDir={#Out}
OutputBaseFilename=PulseX-Z-Image-Setup-{#AppVersion}
SetupIconFile={#Src}\bin\pulsex-zimage.ico
UninstallDisplayIcon={app}\bin\pulsex-zimage.ico
UninstallDisplayName={#AppName}
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
VersionInfoVersion={#AppVersion}.0.0
VersionInfoProductName={#AppName}
VersionInfoDescription={#AppName} Setup

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"; InfoBeforeFile: "before_en.txt"
Name: "swedish"; MessagesFile: "compiler:Languages\Swedish.isl"; InfoBeforeFile: "before_sv.txt"

[CustomMessages]
english.TypeFull=The program and the picture model (recommended)
english.TypeProgram=The program only - I already have the model files
english.CompProgram=PulseX Z-Image
english.CompModel=The picture model, downloaded from Hugging Face (5.8 GB)
english.NotPlain=The NPU can only load PulseX from a folder whose path has plain letters (A-Z), digits and spaces. Choose a folder such as C:\PulseX Z-Image.
english.NeedVC=PulseX Z-Image needs the Microsoft Visual C++ Redistributable for ARM64, which is not installed on this computer.%n%nOpen Microsoft's download now? Run the file you get, then go on with this installation.
swedish.TypeFull=Programmet och bildmodellen (rekommenderas)
swedish.TypeProgram=Bara programmet - jag har redan modellfilerna
swedish.CompProgram=PulseX Z-Image
swedish.CompModel=Bildmodellen, hämtas från Hugging Face (5,8 GB)
swedish.NotPlain=NPU:n kan bara läsa in PulseX från en mapp vars sökväg har vanliga bokstäver (A-Z), siffror och mellanslag. Välj en mapp som C:\PulseX Z-Image.
swedish.NeedVC=PulseX Z-Image behöver Microsoft Visual C++ Redistributable för ARM64, som inte finns på den här datorn.%n%nÖppna Microsofts nedladdning nu? Kör filen du får och fortsätt sedan med den här installationen.

[Types]
Name: "full"; Description: "{cm:TypeFull}"
Name: "program"; Description: "{cm:TypeProgram}"

[Components]
Name: "program"; Description: "{cm:CompProgram}"; Types: full program; Flags: fixed
; ExtraDiskSpaceRequired: the two .hexpack files the first picture writes beside the model
Name: "model"; Description: "{cm:CompModel}"; Types: full; ExtraDiskSpaceRequired: 5447614464

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Dirs]
Name: "{app}\models"

[Files]
; the window and the chain it runs
Source: "{#Src}\bin\pulsex-zimage.exe";      DestDir: "{app}\bin"; Components: program; Flags: ignoreversion
Source: "{#Src}\bin\pulsex-zimage.ico";      DestDir: "{app}\bin"; Components: program; Flags: ignoreversion
Source: "{#Src}\bin\zimage-make.exe";        DestDir: "{app}\bin"; Components: program; Flags: ignoreversion
Source: "{#Src}\bin\zimage-encode.exe";      DestDir: "{app}\bin"; Components: program; Flags: ignoreversion
Source: "{#Src}\bin\zimage-dit-stream.exe";  DestDir: "{app}\bin"; Components: program; Flags: ignoreversion
Source: "{#Src}\bin\taef1_decode.exe";       DestDir: "{app}\bin"; Components: program; Flags: ignoreversion
Source: "{#Src}\bin\llama.dll";              DestDir: "{app}\bin"; Components: program; Flags: ignoreversion
Source: "{#Src}\bin\ggml.dll";               DestDir: "{app}\bin"; Components: program; Flags: ignoreversion
Source: "{#Src}\bin\ggml-base.dll";          DestDir: "{app}\bin"; Components: program; Flags: ignoreversion
Source: "{#Src}\bin\ggml-cpu.dll";           DestDir: "{app}\bin"; Components: program; Flags: ignoreversion
Source: "{#Src}\bin\ggml-hexagon.dll";       DestDir: "{app}\bin"; Components: program; Flags: ignoreversion
Source: "{#Src}\bin\libggml-htp-v73.so";     DestDir: "{app}\bin"; Components: program; Flags: ignoreversion
Source: "{#Src}\bin\libggml-htp-v75.so";     DestDir: "{app}\bin"; Components: program; Flags: ignoreversion
Source: "{#Src}\bin\libggml-htp-v79.so";     DestDir: "{app}\bin"; Components: program; Flags: ignoreversion
Source: "{#Src}\bin\libggml-htp-v81.so";     DestDir: "{app}\bin"; Components: program; Flags: ignoreversion
Source: "{#Src}\bin\libggml-htp.cat";        DestDir: "{app}\bin"; Components: program; Flags: ignoreversion
Source: "{#Src}\bin\LICENSE-llama.cpp-ggml.txt"; DestDir: "{app}\bin"; Components: program; Flags: ignoreversion
Source: "{#Src}\cues.tsv";                   DestDir: "{app}"; Components: program; Flags: ignoreversion
; the settings: kept when a newer version is installed over this one (the user may have moved the models)
Source: "zimage.installed.json";             DestDir: "{app}"; DestName: "zimage.json"; Components: program; Flags: onlyifdoesntexist
Source: "{#Src}\LICENSE";                    DestDir: "{app}"; DestName: "LICENSE.txt"; Components: program; Flags: ignoreversion
Source: "{#Src}\README.md";                  DestDir: "{app}"; Components: program; Flags: ignoreversion
#ifdef Qnn
; Qualcomm's AI runtime, for the upscaler (see ABOUT-QNN.txt)
Source: "{#Qnn}\QnnHtp.dll";                  DestDir: "{app}\qnn"; Components: program; Flags: ignoreversion
Source: "{#Qnn}\QnnSystem.dll";               DestDir: "{app}\qnn"; Components: program; Flags: ignoreversion
Source: "{#Qnn}\QnnHtpV73Stub.dll";           DestDir: "{app}\qnn"; Components: program; Flags: ignoreversion
Source: "{#QnnSkel}\libQnnHtpV73Skel.so";     DestDir: "{app}\qnn"; Components: program; Flags: ignoreversion
Source: "{#QnnSkel}\libqnnhtpv73.cat";        DestDir: "{app}\qnn"; Components: program; Flags: ignoreversion
Source: "ABOUT-QNN.txt";                      DestDir: "{app}\qnn"; Components: program; Flags: ignoreversion
#endif
#ifdef Upscaler
; the upscaler: QuickSRNet-Large x4 as a compiled NPU graph (see LICENSE-QuickSRNet.txt)
Source: "{#Upscaler}"; DestName: "quicksrnetlarge_512x512_ctx_qnn.bin"; DestDir: "{app}\upscaler"; Components: program; Flags: ignoreversion
Source: "LICENSE-QuickSRNet.txt";             DestDir: "{app}\upscaler"; Components: program; Flags: ignoreversion
#endif
; the picture model: downloaded during the installation and checked. A file that is there already is kept (a newer
; version installed over this one does not fetch 5.8 GB again).
Source: "{#ModelRepo}/z-image-turbo-ultrareal06-q4_0.gguf"; DestName: "z-image-turbo-ultrareal06-q4_0.gguf"; DestDir: "{app}\models"; \
  Hash: "18c0ce3fb54562d9cf7bc97279441ab86c0493881e3310e0b8a9dc21520a7a58"; ExternalSize: 3465750400; \
  Components: model; Flags: external download ignoreversion onlyifdoesntexist
Source: "{#ModelRepo}/qwen3-4b-zimage-q4_0.gguf"; DestName: "qwen3-4b-zimage-q4_0.gguf"; DestDir: "{app}\models"; \
  Hash: "959e546a24f79870c7fb7ef5a8b7f33d2f8568d9622746d446656eb340dcfa3a"; ExternalSize: 2369546656; \
  Components: model; Flags: external download ignoreversion onlyifdoesntexist
Source: "{#Taef1Repo}/diffusion_pytorch_model.safetensors"; DestName: "diffusion_pytorch_model.safetensors"; DestDir: "{app}\models\taef1"; \
  Hash: "47a6c2bff850da04b267cab70fe3553fef57255eb9a8e76852baa0a87850e54d"; ExternalSize: 9848636; \
  Components: model; Flags: external download ignoreversion onlyifdoesntexist

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\bin\pulsex-zimage.exe"; WorkingDir: "{app}\bin"; IconFilename: "{app}\bin\pulsex-zimage.ico"
Name: "{autodesktop}\{#AppName}";  Filename: "{app}\bin\pulsex-zimage.exe"; WorkingDir: "{app}\bin"; IconFilename: "{app}\bin\pulsex-zimage.ico"; Tasks: desktopicon

[Run]
Filename: "{app}\bin\pulsex-zimage.exe"; Description: "{cm:LaunchProgram,{#AppName}}"; WorkingDir: "{app}\bin"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; what the first picture wrote beside the model (the model in the NPU's own layout). Anything else in models\ that
; this setup did not put there is the user's and stays - then the folder stays too.
Type: files; Name: "{app}\models\*.hexpack"
Type: dirifempty; Name: "{app}\models\taef1"
Type: dirifempty; Name: "{app}\models"
Type: dirifempty; Name: "{app}"

[Code]
// plain: what the NPU driver accepts in the path of the folder it loads from
function IsPlain(const S: String): Boolean;
var
  I: Integer;
begin
  Result := True;
  for I := 1 to Length(S) do
    if Ord(S[I]) > 126 then begin
      Result := False;
      Exit;
    end;
end;

function DefaultDir(Param: String): String;
begin
  Result := ExpandConstant('{autopf}\{#AppName}');
  if not IsPlain(Result) then
    Result := ExpandConstant('{sd}\{#AppName}');
end;

function NextButtonClick(CurPageID: Integer): Boolean;
begin
  Result := True;
  if (CurPageID = wpSelectDir) and not IsPlain(WizardDirValue) then begin
    SuppressibleMsgBox(CustomMessage('NotPlain'), mbError, MB_OK, IDOK);
    Result := False;
  end;
end;

// the same rule for an installation that shows no pages (/SILENT /DIR=...)
function PrepareToInstall(var NeedsRestart: Boolean): String;
begin
  Result := '';
  if not IsPlain(WizardDirValue) then
    Result := CustomMessage('NotPlain');
end;

function InitializeSetup(): Boolean;
var
  Code: Integer;
begin
  Result := True;
  if not (FileExists(ExpandConstant('{sys}\vcruntime140.dll')) and FileExists(ExpandConstant('{sys}\msvcp140.dll'))
          and FileExists(ExpandConstant('{sys}\vcomp140.dll'))) then
    if SuppressibleMsgBox(CustomMessage('NeedVC'), mbConfirmation, MB_YESNO, IDNO) = IDYES then
      ShellExec('open', 'https://aka.ms/vs/17/release/vc_redist.arm64.exe', '', '', SW_SHOWNORMAL, ewNoWait, Code);
end;
