@echo off
REM Bygger zimage.exe - startaren som kor zimage.ps1 och bar ikonen.
REM Motorn ar zimage.ps1 + zimage.py; den har filen andras nastan aldrig.
setlocal
call "D:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsarm64.bat" >nul 2>&1
set LLVM=D:\Tools\llvm\bin
cd /d "%~dp0"
"%LLVM%\llvm-rc.exe" /FO zimage.res zimage.rc || exit /b 1
"%LLVM%\clang.exe" --target=arm64-pc-windows-msvc -O2 -municode ^
    zimage_launch.c zimage.res -o zimage.exe ^
    -lkernel32 -lshell32 -luser32 -lpsapi -ladvapi32 || exit /b 1
del zimage.res 2>nul
echo [bld_cli] zimage.exe klar
