<#
    zimage - text to image, on the NPU.

    Thin CLI in front of the engine, same split as GenieX: this front handles the user and
    the encoder server's lifecycle, the engine (zimage.py + zimage-dit-stream.exe) does the
    work. Paths and defaults live in zimage.json next to this file; options given on the
    command line win over the file.

    NOTE: all user-facing text is ENGLISH and ASCII (PowerShell 5.1 reads BOM-less files as
    ANSI). This is a tool meant to be released.
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)] [string] $Command,
    [Parameter(Position = 1, ValueFromRemainingArguments = $true)] [string[]] $Rest,

    # 0 / '' = "not given": the value then comes from zimage.json.
    [Alias('o')] [string] $Out = '',
    [ValidatePattern('^(|0|512|1024|480p|720p|[0-9]+x[0-9]+)$')] [string] $Size = '',
    [ValidateRange(0, 50)]   [int] $Steps = 0,
    [int] $Seed = -1,
    [ValidateRange(0, 100)]  [int] $Quality = 0,
    [ValidateSet('', 'taef1', 'full')] [string] $Vae = '',
    [ValidateSet('', 'cpu', 'npu')] [string] $EncoderOn = '',
    [Alias('Up')] [switch] $Upscale,
    [Alias('NoUp')] [switch] $NoUpscale,
    [ValidateSet('', 'quicksrnet', 'esrgan')] [string] $Upscaler = '',
    [switch] $Enrich,
    [string] $Title = '',
    [string] $Subtitle = '',
    [ValidateSet('top', 'bottom')] [string] $TextPos = 'bottom',
    [switch] $Cascade,
    [ValidateRange(1, 3)] [int] $Tail = 2,
    [switch] $Keep,
    [switch] $Pause,
    [switch] $NoServer,
    [Alias('ShowTimings')] [switch] $Timings
)

$ErrorActionPreference = 'Stop'
$Root    = $PSScriptRoot
# The window says PulseX; the command stays "zimage" (09-27).
try { $Host.UI.RawUI.WindowTitle = 'PulseX' } catch {}
$Version = '0.9'

# ---- configuration (zimage.json) -------------------------------------------------------------
# Same rules as zimage.py: relative paths are resolved against the json's folder and {models}
# expands to paths.models. ZIMAGE_CONFIG points somewhere else.
$ConfigFile = if ($env:ZIMAGE_CONFIG) { $env:ZIMAGE_CONFIG } else { Join-Path $Root 'zimage.json' }
if (-not (Test-Path $ConfigFile)) {
    Write-Host "PulseX: config file not found: $ConfigFile" -ForegroundColor Red
    Write-Host "  copy zimage.example.json to zimage.json next to zimage.ps1 and set the paths for this machine." -ForegroundColor Yellow
    exit 1
}
$Cfg = Get-Content $ConfigFile -Raw | ConvertFrom-Json
function Resolve-CfgPath([string] $p, [string] $models) {
    $p = $p.Replace('{models}', $models).Replace('/', '\')
    if ([System.IO.Path]::IsPathRooted($p)) { return $p }
    return [System.IO.Path]::GetFullPath((Join-Path (Split-Path $ConfigFile -Parent) $p))
}
$Models  = Resolve-CfgPath $Cfg.paths.models ''
$Dit     = Resolve-CfgPath $Cfg.paths.dit $Models
$Encoder = Resolve-CfgPath $Cfg.paths.encoder $Models
$VaeFull = Resolve-CfgPath $Cfg.paths.vae_full $Models
$Taef1   = Resolve-CfgPath $Cfg.paths.taef1 $Models
$Bin     = Resolve-CfgPath $Cfg.paths.bin $Models
# 10-01: the NPU driver only loads from a plain-ASCII folder; zimage.py copies such a bin/ once to ProgramData
if ($Bin -match '[^\x00-\x7F]') { $ab = & python (Join-Path $Root 'zimage.py') --ascii-bin 2>$null; if ($LASTEXITCODE -eq 0 -and $ab) { $Bin = "$ab".Trim() } }
$Esrgan  = Resolve-CfgPath $Cfg.paths.esrgan_x4 $Models
$Qsr     = if ($Cfg.paths.quicksrnet_x4) { Resolve-CfgPath $Cfg.paths.quicksrnet_x4 $Models } else { '' }   # 10-01: optional
$UpEngine = if ($Upscaler) { $Upscaler } elseif ($Cfg.defaults.upscale_engine) { [string] $Cfg.defaults.upscale_engine } else { 'quicksrnet' }
if ($UpEngine -eq 'quicksrnet' -and -not ($Qsr -and (Test-Path $Qsr))) { $UpEngine = 'esrgan' }
$UpName  = if ($UpEngine -eq 'quicksrnet') { 'QuickSRNet' } else { 'Real-ESRGAN' }
$QnnRt   = Resolve-CfgPath $Cfg.paths.qnn_runtime $Models
$Port    = [int] $Cfg.server.port
$D       = $Cfg.defaults
if (-not $Out)       { $Out       = [string] $D.out }
if (-not $Size -or $Size -eq '0') { $Size = [string] $D.size }
if (-not $Steps)     { $Steps     = [int] $D.steps }
if ($Seed -lt 0)     { $Seed      = [int] $D.seed }
if (-not $Quality)   { $Quality   = [int] $D.quality }
if (-not $Vae)       { $Vae       = [string] $D.vae }
if (-not $EncoderOn) { $EncoderOn = if ($D.encoder) { [string] $D.encoder } else { 'cpu' } }
$UpscaleTo = [int] $D.upscale_to   # 1024; 2048 works but is held back for a later release
if (-not $PSBoundParameters.ContainsKey('Upscale') -and $D.upscale) { $Upscale = [switch]::new($true) }
if ($NoUpscale) { $Upscale = [switch]::new($false) }

# Colour scheme, after GenieX: cyan headings, green commands and prompt, yellow options.
function W([string] $t, [string] $c = 'Gray', [switch] $NoNL) {
    if ($NoNL) { Write-Host $t -ForegroundColor $c -NoNewline } else { Write-Host $t -ForegroundColor $c }
}

function Show-Rows([array] $rows, [string] $colour, [int] $w, [int] $w2 = 0) {
    foreach ($r in $rows) {
        W ('  {0}' -f $r[0].PadRight($w)) $colour -NoNL
        W $(if ($w2) { $r[1].PadRight($w2) } else { $r[1] }) -NoNL
        if ($r.Count -gt 2 -and $r[2]) { W ('  ' + $r[2]) DarkGray } else { W '' }
    }
}

function Show-Usage {
    W ''
    W '  PulseX' White -NoNL; W " $Version - turns a sentence into a picture, on this PC's NPU."
    W '  No internet, no GPU: a text encoder, an image model and a decoder run locally.' DarkGray

    W ''; W 'Quick start' Cyan
    W '  zimage make "a red fox in an autumn forest"' Green
    W '    -> zimage_0001.jpg in the current folder, then _0002 ... - nothing is overwritten.' DarkGray
    W '  zimage repl' Green
    W '    -> interactive: type one prompt after another. Fastest way to try ideas.' DarkGray

    W ''; W 'Which mode should I use?' Cyan
    Show-Rows @(
        @('(default)',       "512, then $UpName x4 -> 1024",                   '~12 s'),
        @('-NoUp',           'plain 512 - fastest, for trying prompts',         '~11 s'),
        @('-Size 1024',      'the real thing - most detail, for keepers',       '~46 s')) Green 16 48
    W '  Times are for "zimage make" (everything starts cold). In "zimage repl" everything' DarkGray
    W '  stays loaded: from the 2nd image on, 512 takes ~10 s and 512 + upscale ~12 s.' DarkGray
    W '  The same prompt + seed gives the same composition at every size.' DarkGray

    W ''; W 'Commands' Cyan
    Show-Rows @(
        @('make "..."', 'make one image and exit'),
        @('text <img>', 'put -Title / -Subtitle on an existing image (no new image)'),
        @('repl',       'interactive mode - models stay loaded between images'),
        @('serve',      'start the text encoder and leave it running'),
        @('stop',       'stop everything, free NPU memory, drop the model cache'),
        @('status',     'what is running, and how much memory is cached'),
        @('config',     'paths, defaults and where images go (from zimage.json)'),
        @('version',    'show the version'),
        @('help',       'this text')) Green 12

    W ''; W 'Options' Cyan
    W '  what you get' DarkGray
    Show-Rows @(
        @('-Out <file>',        'exact file name (.jpg/.png); without it: next free', "zimage_NNNN.jpg"),
        @('-Size 512|1024|480p', 'resolution; 480p = 848 x 480 (wide, no upscaler)', "now $Size"),
        @('-NoUp',              'keep the plain 512 image (no x4 upscaler)',         $(if ($Upscale) { 'upscaler now on' } else { 'upscaler now off' })),
        @('-Upscaler <name>',   'quicksrnet (fast, default) or esrgan (Real-ESRGAN)', "now $UpEngine"),
        @('-Seed <n>',          'another number = another image for the same text',  "now $Seed"),
        @('-Enrich',            'add material words (skin pores, wet sand ...)',     $(if ($D.enrich) { 'now on' } else { 'now off' })),
        @('-Quality <1-100>',   'JPEG quality',                                      "now $Quality")) Yellow 22 50
    W '  text on the picture (spelled right - the model itself cannot write)' DarkGray
    Show-Rows @(
        @('-Title "..."',       'big line of text on the picture',                   ''),
        @('-Subtitle "..."',    'smaller line under it',                             ''),
        @('-TextPos top|bottom', 'where the text goes',                              "now $TextPos")) Yellow 22 50
    W '  how it is made (rarely needed)' DarkGray
    Show-Rows @(
        @('-Steps <n>',         'denoising steps; the model is trained for 4',       "now $Steps"),
        @('-Vae taef1|full',    'decoder: taef1 is fast, full is the reference',     "now $Vae"),
        @('-EncoderOn npu|cpu', 'text encoder: npu loads per prompt while you type; cpu stays warm', "now $EncoderOn"),
        @('-Cascade',           '1024 via a 512 pass (faster, experimental)',        ''),
        @('-Tail <n>',          'steps the -Cascade refine stage re-runs',           "now $Tail")) Yellow 22 50
    W '  housekeeping' DarkGray
    Show-Rows @(
        @('-Keep',              'keep the text encoder loaded after make',           ''),
        @('-Timings',           'print where the time goes, phase by phase',         ''),
        @('-Pause',             'wait for Enter before closing (for shortcuts)',     ''),
        @('-NoServer',          'run without the warm encoder (slower, fallback)',   '')) Yellow 22 50

    W ''; W 'Examples' Cyan
    W '  zimage make "a lighthouse at dusk"' Green
    W '  zimage make "a lighthouse at dusk" -Out lighthouse.jpg' Green
    W '  zimage make "a lighthouse at dusk" -Title "Open House" -Subtitle "Saturday 10-14"' Green
    W '  zimage text lighthouse.jpg -Title "Open House" -TextPos top' Green
    W '  zimage make "a lighthouse at dusk" -Size 1024 -Seed 7' Green
    W '  zimage repl -NoUp' Green
    W ''
    W "Settings live in $ConfigFile - options above override them for one run." DarkGray
    W ''
}

# A refused connection to localhost costs ~2 s on Windows (TCP retries the SYN), so asking
# /health while nothing listens cost 2.0 s before every start AND made the ready-poll tick
# every ~2 s instead of 100 ms. The listener table answers in ~1 ms; HTTP only once it is open.
function Test-Encoder {
    $open = [System.Net.NetworkInformation.IPGlobalProperties]::GetIPGlobalProperties().GetActiveTcpListeners() |
            Where-Object { $_.Port -eq $Port }
    if (-not $open) { return $false }
    try { $null = Invoke-RestMethod "http://127.0.0.1:$Port/health" -TimeoutSec 2 -EA Stop; $true }
    catch { $false }
}

function Start-Encoder {
    # npu: zimage.py starts zimage-encode.exe per prompt (nothing may sit on the NPU in between)
    if ($EncoderOn -eq 'npu') { return }
    if (Test-Encoder) { return }
    # Ingen gissad tid har: den varierar med om modellen ligger i cachen eller inte.
    # Mat och rapportera i stallet - en pahittad siffra som alltid ar fel irriterar mer
    # an den hjalper.
    $t0 = Get-Date
    # (tyst sedan 09-28, raden behovs inte - fel visas fortfarande)
    $env:ADSP_LIBRARY_PATH = $Bin      # without this: 0x80000406 + segfault
    $env:QWEN_EMBD_LAYER   = '35'      # cap_feats = hidden_states[-2], NOT the final norm
    $log = Join-Path $env:TEMP 'zimage-encoder'
    $encArgs = @(
            '-m', $Encoder, '--embeddings', '--pooling', 'none', '--embd-normalize', '-1',
            '-c', '512', '--host', '127.0.0.1', '--port', $Port, '--no-warmup') + $(
            if ($EncoderOn -eq 'npu') {
                # ONLY the NPU. With a GPU backend present, -ngl 99 split the layers NPU/GPU:
                # cap_feats relL2 0.155 vs the HF reference (NPU alone 0.047), ready 5.9 s vs 3.2 s.
                # Its weights are copied into NPU memory: +2.8 GB COMMITTED while it runs.
                @('-ngl', '99', '--device', 'HTP0')
            } else {
                # CPU without repacking reads the weights straight from the mapped file (the
                # page cache), so it commits ~0.5 GB instead of 2.8. Ready 1.6 s; a long prompt
                # costs ~1.2 s more than on the NPU. Default threads (8) are best.
                @('-ngl', '0', '--device', 'none', '--no-repack')
            }) + @(
            # 'auto' falls back to plain reads because HTP cannot map; forcing mmap reads the
            # file straight from the page cache: load 1.50 -> 1.03 s, ready 2.90 -> 2.40 s,
            # same embedding bit for bit.
            '--load-mode', 'mmap')
    Start-Process (Join-Path $Bin 'llama-server.exe') -WindowStyle Hidden `
        -RedirectStandardOutput "$log.log" -RedirectStandardError "$log.err" -ArgumentList $encArgs
    $deadline = (Get-Date).AddSeconds(180)
    $polls = 0
    while (-not (Test-Encoder)) {
        if ((Get-Date) -gt $deadline) { throw "the text encoder never came up - see $log.err" }
        # Poll often, print rarely: at 700 ms the poll alone cost ~0.5 s of the 3.4 s start.
        Start-Sleep -Milliseconds 100
    }

}

function Get-StandbyMB {
    try { [int]((Get-Counter '\Memory\Standby Cache Normal Priority Bytes' -EA Stop).CounterSamples[0].CookedValue / 1MB) }
    catch { -1 }
}

# Models are memory-mapped, so their pages stay in Windows' standby list after the process
# exits. They count as AVAILABLE, but they show as "Cached" and are not needed once we are
# done. Measured: 6285 -> 717 MB standby, and it costs nothing - the models live on C: (SSD).
# Returns the MB the drop released (zimage.exe measures it itself), or -1.
function Clear-ModelCache {
    $exe = Join-Path $Root 'zimage.exe'
    if (-not (Test-Path $exe)) { return -1 }
    $files = @($Dit, $Encoder)
    $esrBin = $Esrgan -replace '\.wrap\.onnx$', ''
    if (Test-Path $esrBin) { $files += $esrBin }
    $line = & $exe --drop-cache @files | Select-String '^dropped (-?\d+)' | Select-Object -Last 1
    if ($line) { return [int] $line.Matches[0].Groups[1].Value }
    return -1
}

# Returns the MB dropped from the cache (-1 if unknown). The pause is only needed when a
# process was actually killed - it used to be paid after every make, with nothing to kill.
function Stop-All {
    $procs = Get-Process llama-server, zimage-dit-stream -EA SilentlyContinue
    if ($procs) { $procs | Stop-Process -Force; Start-Sleep -Milliseconds 400 }
    return (Clear-ModelCache)
}

function Test-Admin {
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
    if ($line -match '^purged (-?\d+)') {
        # tyst (09-28); resultatet ar inget att agera pa
    }
}

function Write-Dropped([int] $mb) {
    if ($mb -ge 0) { W ("NPU memory freed, {0} MB dropped from cache." -f $mb) DarkGray }
    else { W 'NPU memory freed, models dropped from cache.' DarkGray }
}

function Hold-Window { if ($Pause) { W ''; W 'Press Enter to close...' DarkGray; $null = Read-Host } }

# -Cascade renders the base at 512 and then re-runs the TAIL of the sigma schedule at
# 1024 on the upscaled, re-noised latent. Measured 91.0 s vs 109.7 s native, image at
# least as good. NOTE: it saves TIME, not memory - the 1024 stage still loads the whole
# model and allocates the same 4116-token activations.
function Invoke-Cascade([string] $prompt) {
    if ($Size -ne '1024') {
        W 'Note: -Cascade only helps at -Size 1024; rendering natively.' DarkGray
        return $false
    }
    Start-Encoder
    $env:ZI_VAE = $Vae
    # Out-Host, not the pipeline: anything a PowerShell function writes becomes part of its
    # RETURN VALUE, so without this the engine's progress lines are swallowed by the
    # `if (Invoke-Cascade ...)` that calls it - the CLI then works in complete silence.
    & python (Join-Path $Root 'cascade2.py') $prompt -o $Out --base 512 --final 1024 `
        --tail $Tail --seed $Seed 2>&1 | Out-Host
    return $true
}

function Invoke-Engine([string[]] $engineArgs) {
    $env:ZI_VAE = $Vae
    $env:ZI_ENCODER_ON = $EncoderOn       # also for the no-server fallback in zimage.py
    # One-shot mode releases the encoder as soon as it has done its job. Measured: 1.9 GB
    # private held for 110 s of a 1024 run while it is used for 0.9. In serve/repl the whole
    # point is that it stays, so the flag is not set there.
    $env:ZI_ENCODER_TRANSIENT = if ($script:Transient) { '1' } else { '0' }
    if ($NoServer) { $env:ZIMAGE_SERVER = 'http://127.0.0.1:1' } else { Start-Encoder }
    & python (Join-Path $Root 'zimage.py') @engineArgs
    $rc = $LASTEXITCODE
    if ($script:CleanupOnExit) {
        $mb = Stop-All
        W '  cleaned up: ' DarkGray -NoNL
        Write-Dropped $mb
        Invoke-Purge 'purge_on_exit'
    }
    Hold-Window
    exit $rc
}

$common = @('--size', $Size, '--steps', $Steps, '--seed', $Seed, '-q', $Quality, '--upscale-to', $UpscaleTo)
if ($PSBoundParameters.ContainsKey('Out')) { $common += @('-o', $Out) }
if ($Upscale) { $common += '--upscale' } else { $common += '--no-upscale' }
if ($Upscaler) { $common += @('--upscaler', $Upscaler) }
if ($Enrich)  { $common += '--enrich' }
$textArgs = @()
if ($Title)    { $textArgs += @('--title', $Title) }
if ($Subtitle) { $textArgs += @('--subtitle', $Subtitle) }
$textArgs += @('--text-pos', $TextPos)
$common += $textArgs
if ($Timings) { $common += '-v' }

switch ($Command) {
    'make' {
        $p = ($Rest -join ' ').Trim()
        if (-not $p) {
            W 'zimage make needs a prompt - the words describing the picture, in quotes:' Yellow
            W '  zimage make "a red fox in an autumn forest"' Green
            exit 1
        }
        $script:Transient = -not $Keep
        if ($Cascade -and (Invoke-Cascade $p)) {
            if (-not $Keep) { $null = Stop-All }
            Hold-Window
            exit 0
        }
        Invoke-Engine (@($p) + $common)
    }
    'text' {
        $img = ($Rest -join ' ').Trim()
        if (-not $img -or -not (Test-Path $img)) {
            W 'zimage text needs an existing image and the text to put on it:' Yellow
            W '  zimage text photo.jpg -Title "Open House" -Subtitle "Saturday 10-14"' Green
            exit 1
        }
        if (-not $Title -and -not $Subtitle) { W 'Give -Title and/or -Subtitle.' Yellow; exit 1 }
        $a = @('--add-text', (Resolve-Path $img).Path, '-q', $Quality) + $textArgs
        if ($PSBoundParameters.ContainsKey('Out')) { $a += @('-o', $Out) }
        & python (Join-Path $Root 'zimage.py') @a
        exit $LASTEXITCODE
    }
    'repl' {
        Invoke-Purge 'purge_on_start'
        $script:Transient     = $false
        $script:CleanupOnExit = -not $Keep
        Invoke-Engine ($common + '--serve')
    }
    'serve'  {
        if ($EncoderOn -eq 'npu') { W '  the NPU text encoder starts per prompt - nothing to keep running (use -EncoderOn cpu for a warm server).' DarkGray; exit 0 }
        Start-Encoder; W "  the text encoder is listening on 127.0.0.1:$Port" Green
    }
    'stop'   {
        $mb = Stop-All
        W '  stopped. ' Green -NoNL
        Write-Dropped $mb
        Invoke-Purge 'purge_on_exit'
    }
    'status' {
        W ''
        W '  text encoder  ' Green -NoNL
        W "port $Port  " -NoNL
        if (Test-Encoder) { W 'running' Green } else { W 'not running' DarkGray }
        foreach ($n in @(@('llama-server', 'encoder process'), @('zimage-dit-stream', 'image model process'))) {
            W ('  {0,-14}' -f $n[1]) Green -NoNL
            W ("{0} running" -f @(Get-Process $n[0] -EA SilentlyContinue).Count)
        }
        W '  cached        ' Green -NoNL
        W ("{0} MB in the standby list  " -f (Get-StandbyMB)) -NoNL
        W '("zimage stop" drops it)' DarkGray
        W ''
    }
    'config' {
        W ''
        W 'Config file' Cyan
        W "  $ConfigFile"
        W ''
        W 'Files' Cyan
        foreach ($kv in @(@('image model', $Dit), @('text encoder', $Encoder), @('decoder taef1', $Taef1),
                          @('decoder full', $VaeFull), @('upscaler QSR', $(if ($Qsr) { $Qsr } else { '(not set)' })),
                          @('upscaler ESR', $Esrgan), @('QNN runtime', $QnnRt),
                          @('binaries', $Bin))) {
            W ('  {0,-15}' -f $kv[0]) Green -NoNL
            W $kv[1] -NoNL
            if (Test-Path $kv[1]) { W '  [ok]' DarkGray } else { W '  [MISSING]' Red }
        }
        W ''
        W 'Where images go' Cyan
        W '  make          ' Green -NoNL; W ("-Out, relative to the current folder: {0}" -f (Get-Location))
        W '  repl          ' Green -NoNL; W ("{0}_0001{1}, _0002 ... in the current folder" -f [IO.Path]::GetFileNameWithoutExtension($Out), [IO.Path]::GetExtension($Out))
        W '  shortcut      ' Green -NoNL; W (Join-Path $env:USERPROFILE 'Pictures\zimage')
        W ''
        W 'Defaults (from the config file)' Cyan
        W ("  size {0} | upscale {1} -> {2} ({9}) | steps {3} | seed {4} | quality {5} | decoder {6} | encoder {7} | port {8}" -f `
            $D.size, $(if ($D.upscale) { 'on' } else { 'off' }), $D.upscale_to, $D.steps, $D.seed, $D.quality, $D.vae, $EncoderOn, $Port, $UpName)
        W ''
    }
    'version' { W "PulseX $Version" White -NoNL; W "  (Z-Image-Turbo Q4_0 + Qwen3-4B Q4_0 + $UpName x4, ggml-hexagon / HTP0)" DarkGray }
    { $_ -in 'help', '-h', '--help', '-?', '/?', '', $null } { Show-Usage }
    default {
        # "zimage \"a cat\"" without a verb should also work
        $script:Transient = -not $Keep
        Invoke-Engine (@((@($Command) + $Rest) -join ' ') + $common)
    }
}
