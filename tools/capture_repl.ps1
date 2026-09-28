# Riktig skarmbild av repl-fonstret for README:n (09-28). Kor PulseX i en neutral katalog med en
# demokonfig (ingen standby-rensning => ingen admin-tipsrad), skriver en prompt med SendKeys, vantar
# tills bilden ar sparad och fotograferar BARA fonstret (GetWindowRect + CopyFromScreen).
#   pwsh -File tools\capture_repl.ps1 <ut.png>
param([string] $OutPng)
$ErrorActionPreference = 'Stop'
$root = 'C:\PulseCore\PulseX\zimage_dit'
$demo = 'D:\PulseX-demo'
New-Item -ItemType Directory -Force $demo | Out-Null
$cfg = Get-Content "$root\zimage.json" -Raw | ConvertFrom-Json
$cfg.defaults.purge_on_start = 'off'
$cfg.defaults.purge_on_exit  = 'off'
$cfg | ConvertTo-Json -Depth 5 | Set-Content "$demo\zimage.json" -Encoding utf8

Add-Type @'
using System; using System.Runtime.InteropServices; using System.Text;
public static class W {
  [StructLayout(LayoutKind.Sequential)] public struct R { public int L, T, Rt, B; }
  [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr h, out R r);
  [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
  [DllImport("user32.dll")] public static extern IntPtr GetForegroundWindow();
  [DllImport("user32.dll")] public static extern bool MoveWindow(IntPtr h, int x, int y, int w, int hh, bool rp);
  [DllImport("user32.dll")] public static extern IntPtr FindWindow(string c, string t);
  [DllImport("dwmapi.dll")] public static extern int DwmGetWindowAttribute(IntPtr h, int a, out R r, int s);
}
'@
Add-Type -AssemblyName System.Drawing, System.Windows.Forms

$env:ZIMAGE_CONFIG = "$demo\zimage.json"
# Fonstret kan vara Windows Terminal (standardterminal) - leta via processernas fonstertitel.
function Find-PulseX { Get-Process | Where-Object { $_.MainWindowTitle -eq 'PulseX' -and $_.MainWindowHandle -ne 0 } | Select-Object -First 1 }
if (-not (Find-PulseX)) { Start-Process "$root\zimage.exe" -ArgumentList 'repl' -WorkingDirectory $demo }
$p = $null
for ($i = 0; $i -lt 150 -and -not $p; $i++) { Start-Sleep -Milliseconds 200; $p = Find-PulseX }
if (-not $p) { throw 'hittade inget PulseX-fonster' }
$h = $p.MainWindowHandle
[W]::MoveWindow($h, 60, 60, 1180, 760, $true) | Out-Null
for ($i = 0; $i -lt 150; $i++) {
    Start-Sleep -Milliseconds 200
    if (Get-NetTCPConnection -LocalPort 8099 -State Listen -EA SilentlyContinue) { break }
}
Start-Sleep 2

function Send([string] $keys) {
    [W]::SetForegroundWindow($h) | Out-Null
    Start-Sleep -Milliseconds 300
    if ([W]::GetForegroundWindow() -ne $h) { throw 'PulseX-fonstret fick inte fokus - avbryter (inga tangenter skickade)' }
    [System.Windows.Forms.SendKeys]::SendWait($keys)
}
Send 'a lighthouse on a rocky coast at dusk{ENTER}'
# vanta tills bilden ar sparad
$deadline = (Get-Date).AddSeconds(90)
while (-not (Get-ChildItem $demo -Filter 'zimage_*.jpg' -EA SilentlyContinue) -and (Get-Date) -lt $deadline) { Start-Sleep 1 }
Start-Sleep 2

# fotografera fonstret (DWM-ramen = den synliga, utan skugga)
$r = New-Object W+R
[W]::DwmGetWindowAttribute($h, 9, [ref] $r, 16) | Out-Null
$w = $r.Rt - $r.L; $hh = $r.B - $r.T
$bmp = New-Object System.Drawing.Bitmap $w, $hh
$g = [System.Drawing.Graphics]::FromImage($bmp)
[W]::SetForegroundWindow($h) | Out-Null; Start-Sleep -Milliseconds 400
$g.CopyFromScreen($r.L, $r.T, 0, 0, $bmp.Size)
$bmp.Save($OutPng, [System.Drawing.Imaging.ImageFormat]::Png)
$g.Dispose(); $bmp.Dispose()
"sparad $OutPng ($w x $hh)"

# stang sessionen snyggt (Ctrl+D)
Send '^d'
Start-Sleep 5
Get-Process llama-server, zimage-dit-stream -EA SilentlyContinue | Select-Object Name, Id
