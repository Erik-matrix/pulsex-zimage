@call "D:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsarm64.bat" >nul 2>&1
cd /d "%~dp0"
cl /nologo /O2 /LD cdsptrace.c /Fe:cdsptrace.dll kernel32.lib >build.log 2>&1
echo EXIT=%errorlevel%
