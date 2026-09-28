# 09-28 ("rensningen bra nar appen startar - och nar den stangs, jag gor bada for att ZImage
# ska fa eget utrymme"): systemomfattande standby-rensning (samma mekanism som PulseCore Memory,
# zimage.exe --purge [full]) vid repl-START och vid repl-avslut / zimage stop.
# Kraver admin. INGEN kvarliggande konfiguration (ingen schemalagd uppgift): kor PulseX som
# administrator (genvagen) sa rensar den sjalv; annars toms som forr bara vara modellfiler.
# zimage.json defaults.purge_on_start / purge_on_exit: standby | full | off.
import io, json
PS = r"C:\PulseCore\PulseX\zimage_dit\zimage.ps1"
PY = r"C:\PulseCore\PulseX\zimage_dit\zimage.py"
JS = [r"C:\PulseCore\PulseX\zimage_dit\zimage.json", r"C:\PulseCore\PulseX\zimage_dit\zimage.example.json"]

p = io.open(PS, encoding="ascii", newline="").read()
nl = "\r\n" if "\r\n" in p else "\n"


def rp(old, new):
    global p
    old, new = old.replace("\n", nl), new.replace("\n", nl)
    if p.count(old) != 1:
        raise SystemExit("ps1: ankare %d ggr: %r" % (p.count(old), old[:70]))
    p = p.replace(old, new)


# riktad tomning: aven ESRGAN-kontexten
rp("""    $line = & $exe --drop-cache $Dit $Encoder | Select-String '^dropped (-?\\d+)' | Select-Object -Last 1""",
   """    $files = @($Dit, $Encoder)
    $esrBin = $Esrgan -replace '\\.wrap\\.onnx$', ''
    if (Test-Path $esrBin) { $files += $esrBin }
    $line = & $exe --drop-cache @files | Select-String '^dropped (-?\\d+)' | Select-Object -Last 1""")

rp("""function Write-Dropped([int] $mb) {""", """function Test-Admin {
    $id = [Security.Principal.WindowsIdentity]::GetCurrent()
    return ([Security.Principal.WindowsPrincipal] $id).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

# Empties Windows' whole standby cache ("Cached" in Task Manager) - the same call PulseCore Memory
# makes. Only possible when PulseX runs as administrator; nothing is installed or left behind.
# $key = 'purge_on_start' | 'purge_on_exit'; value standby (default) | full | off.
$script:PurgeTipShown = $false
function Invoke-Purge([string] $key) {
    $mode = [string] $D.$key
    if (-not $mode) { $mode = 'standby' }
    if ($mode -eq 'off') { return }
    if (-not (Test-Admin)) {
        if (-not $script:PurgeTipShown) {
            W "  tip: run PulseX as administrator to also empty Windows' standby cache on start and exit" DarkGray
            W '       (shortcut: Properties > Shortcut > Advanced > Run as administrator).' DarkGray
            $script:PurgeTipShown = $true
        }
        return
    }
    $pa = @('--purge'); if ($mode -eq 'full') { $pa += 'full' }
    $line = & (Join-Path $Root 'zimage.exe') @pa | Select-Object -Last 1
    if ($line -match '^purged (-?\\d+)') {
        W ('  Windows standby cache emptied: {0:N1} GB given back.' -f ([int] $Matches[1] / 1024.0)) DarkGray
    }
}

function Write-Dropped([int] $mb) {""")

rp("""    if ($script:CleanupOnExit) {
        $mb = Stop-All
        W '  cleaned up: ' DarkGray -NoNL
        Write-Dropped $mb
    }""", """    if ($script:CleanupOnExit) {
        $mb = Stop-All
        W '  cleaned up: ' DarkGray -NoNL
        Write-Dropped $mb
        Invoke-Purge 'purge_on_exit'
    }""")

rp("""    'stop'   {
        $mb = Stop-All
        W '  stopped. ' Green -NoNL
        Write-Dropped $mb
    }""", """    'stop'   {
        $mb = Stop-All
        W '  stopped. ' Green -NoNL
        Write-Dropped $mb
        Invoke-Purge 'purge_on_exit'
    }""")

# repl: rensa FORE modellerna laddas, sa PulseX far eget utrymme
rp("""    'repl' {
        $script:Transient     = $false""", """    'repl' {
        Invoke-Purge 'purge_on_start'
        $script:Transient     = $false""")
io.open(PS, "w", encoding="ascii", newline="").write(p)

for J in JS:
    j = json.load(open(J, encoding="utf-8"))
    nd = {}
    for k, v in j["defaults"].items():
        nd[k] = v
        if k == "unload_after_s":
            nd["purge_on_start"] = "standby"
            nd["purge_on_exit"] = "standby"
    j["defaults"] = nd
    if "_help" in j:
        j["_help"]["defaults.purge_on_start"] = ("REPL start: empty Windows' whole standby cache ('Cached') before the models load. "
                                               "standby | full (also trim every process's working set, like PulseCore Memory) | off. "
                                               "Only when PulseX runs as administrator; otherwise only PulseX's own model files are dropped.")
        j["_help"]["defaults.purge_on_exit"] = "the same on 'zimage stop' and when the REPL closes"
    io.open(J, "w", encoding="utf-8", newline="\n").write(json.dumps(j, indent=2, ensure_ascii=False) + "\n")

s = io.open(PY, encoding="utf-8", newline="").read()
old = '"unload_after_s": 60}'
assert s.count(old) == 1
s = s.replace(old, '"unload_after_s": 60, "purge_on_start": "standby", "purge_on_exit": "standby"}')
io.open(PY, "w", encoding="utf-8", newline="").write(s)
print("OK")
