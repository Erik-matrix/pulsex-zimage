# zimage.ps1: slutstadningen efter make (och stop) utan Get-Counter.
# Matt 09-27: Get-StandbyMB (Get-Counter) 1,0-1,4 s STYCK, anropad fore OCH efter, plus en fast
# paus pa 400 ms aven nar ingen process fanns att doda = ~3 s efter att bilden redan var skriven.
# Nu: zimage.exe --drop-cache mater sjalv (GetPerformanceInfo) och skriver "dropped <MB>";
# pausen bara om nagot faktiskt dodades. Get-StandbyMB finns kvar for `status`.
# Filen ar ASCII.
import io
P = r"C:\PulseCore\PulseX\zimage_dit\zimage.ps1"
s = io.open(P, encoding="ascii", newline="").read()
nl = "\r\n" if "\r\n" in s else "\n"
assert "dropped (" not in s


def rep(old, new):
    global s
    old, new = old.replace("\n", nl), new.replace("\n", nl)
    assert s.count(old) == 1, (s.count(old), old[:70])
    s = s.replace(old, new)


rep("""function Clear-ModelCache {
    $exe = Join-Path $Root 'zimage.exe'
    if (Test-Path $exe) { & $exe --drop-cache $Dit $Encoder | Out-Null }
}""", """# Returns the MB the drop released (zimage.exe measures it itself), or -1.
function Clear-ModelCache {
    $exe = Join-Path $Root 'zimage.exe'
    if (-not (Test-Path $exe)) { return -1 }
    $line = & $exe --drop-cache $Dit $Encoder | Select-String '^dropped (-?\\d+)' | Select-Object -Last 1
    if ($line) { return [int] $line.Matches[0].Groups[1].Value }
    return -1
}""")

rep("""function Stop-All {
    Get-Process llama-server, zimage-dit-stream -EA SilentlyContinue | Stop-Process -Force
    Start-Sleep -Milliseconds 400
    Clear-ModelCache
}""", """# Returns the MB dropped from the cache (-1 if unknown). The pause is only needed when a
# process was actually killed - it used to be paid after every make, with nothing to kill.
function Stop-All {
    $procs = Get-Process llama-server, zimage-dit-stream -EA SilentlyContinue
    if ($procs) { $procs | Stop-Process -Force; Start-Sleep -Milliseconds 400 }
    return (Clear-ModelCache)
}

function Write-Dropped([int] $mb) {
    if ($mb -ge 0) { W ("NPU memory freed, {0} MB dropped from cache." -f $mb) DarkGray }
    else { W 'NPU memory freed, models dropped from cache.' DarkGray }
}""")

rep("""        $before = Get-StandbyMB
        Stop-All
        $after = Get-StandbyMB
        if ($before -ge 0 -and $after -ge 0) {
            W ("[zimage] cleaned up: NPU memory freed, {0} MB dropped from cache." -f ($before - $after)) DarkGray
        } else {
            W '[zimage] cleaned up: NPU memory freed, models dropped from cache.' DarkGray
        }""", """        $mb = Stop-All
        W '[zimage] cleaned up: ' DarkGray -NoNL
        Write-Dropped $mb""")

rep("""        if (-not $Keep) { Stop-All }""", """        if (-not $Keep) { $null = Stop-All }""")

rep("""        $before = Get-StandbyMB
        Stop-All
        $after = Get-StandbyMB
        W '[zimage] stopped. ' Green -NoNL
        if ($before -ge 0 -and $after -ge 0) { W ("NPU memory freed, {0} MB dropped from cache." -f ($before - $after)) }
        else { W 'NPU memory freed, models dropped from cache.' }""", """        $mb = Stop-All
        W '[zimage] stopped. ' Green -NoNL
        Write-Dropped $mb""")

io.open(P, "w", encoding="ascii", newline="").write(s)
print("OK")
