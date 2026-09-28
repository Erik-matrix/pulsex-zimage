# zimage.ps1 + zimage.json: uppskalning forval vid 512 (-NoUp for att slippa), -Title/-Subtitle/-TextPos,
# nytt kommando `text` (text pa en befintlig bild), hjalptexten uppdaterad. ASCII.
import io, json
P = r"C:\PulseCore\PulseX\zimage_dit\zimage.ps1"
J = r"C:\PulseCore\PulseX\zimage_dit\zimage.json"
s = io.open(P, encoding="ascii", newline="").read()
nl = "\r\n" if "\r\n" in s else "\n"
assert "NoUpscale" not in s


def rep(old, new, count=1):
    global s
    old, new = old.replace("\n", nl), new.replace("\n", nl)
    n = s.count(old)
    if n != count:
        raise SystemExit("ankare %d ggr (vill %d): %r" % (n, count, old[:80]))
    s = s.replace(old, new)


rep("""    [Alias('Up')] [switch] $Upscale,""", """    [Alias('Up')] [switch] $Upscale,
    [Alias('NoUp')] [switch] $NoUpscale,
    [string] $Title = '',
    [string] $Subtitle = '',
    [ValidateSet('top', 'bottom')] [string] $TextPos = 'bottom',""")
rep("""if (-not $PSBoundParameters.ContainsKey('Upscale') -and $D.upscale) { $Upscale = [switch]::new($true) }""",
    """if (-not $PSBoundParameters.ContainsKey('Upscale') -and $D.upscale) { $Upscale = [switch]::new($true) }
if ($NoUpscale) { $Upscale = [switch]::new($false) }""")

# hjalptexten
rep("""    Show-Rows @(
        @('512',             'try prompts - fast, and already good',            '~16 s'),
        @('512 -Upscale',    '512, then Real-ESRGAN x4 -> 1024 (old ZImage)',   '~19 s'),
        @('-Size 1024',      'the real thing - most detail, for keepers',       '~46 s')) Green 16 48""",
"""    Show-Rows @(
        @('(default)',       '512, then Real-ESRGAN x4 -> 1024 (old ZImage)',   '~19 s'),
        @('-NoUp',           'plain 512 - fastest, for trying prompts',         '~16 s'),
        @('-Size 1024',      'the real thing - most detail, for keepers',       '~46 s')) Green 16 48""")
rep("""    W '  stays loaded: from the 2nd image on, 512 takes ~10 s and 512 -Upscale ~12 s.' DarkGray""",
    """    W '  stays loaded: from the 2nd image on, 512 takes ~10 s and 512 + upscale ~12 s.' DarkGray""")
rep("""        @('make "..."', 'make one image and exit'),""",
    """        @('make "..."', 'make one image and exit'),
        @('text <img>', 'put -Title / -Subtitle on an existing image (no new image)'),""")
rep("""        @('-Upscale  (-Up)',    'with -Size 512: sharpen to a larger size (x4)',     $(if ($Upscale) { 'now on' } else { 'now off' })),""",
    """        @('-NoUp',              'keep the plain 512 image (no x4 upscaler)',         $(if ($Upscale) { 'upscaler now on' } else { 'upscaler now off' })),""")
rep("""        @('-Quality <1-100>',   'JPEG quality',                                      "now $Quality")) Yellow 22 50""",
    """        @('-Quality <1-100>',   'JPEG quality',                                      "now $Quality")) Yellow 22 50
    W '  text on the picture (spelled right - the model itself cannot write)' DarkGray
    Show-Rows @(
        @('-Title "..."',       'big line of text on the picture',                   ''),
        @('-Subtitle "..."',    'smaller line under it',                             ''),
        @('-TextPos top|bottom', 'where the text goes',                              "now $TextPos")) Yellow 22 50""")
rep("""    W '  zimage make "a lighthouse at dusk" -Upscale -Out lighthouse.jpg' Green""",
    """    W '  zimage make "a lighthouse at dusk" -Out lighthouse.jpg' Green
    W '  zimage make "a lighthouse at dusk" -Title "Open House" -Subtitle "Saturday 10-14"' Green
    W '  zimage text lighthouse.jpg -Title "Open House" -TextPos top' Green""")
rep("""    W '  zimage repl -Upscale' Green""", """    W '  zimage repl -NoUp' Green""")

# argumenten till motorn
rep("""if ($Upscale) { $common += '--upscale' }""",
    """if ($Upscale) { $common += '--upscale' } else { $common += '--no-upscale' }
$textArgs = @()
if ($Title)    { $textArgs += @('--title', $Title) }
if ($Subtitle) { $textArgs += @('--subtitle', $Subtitle) }
$textArgs += @('--text-pos', $TextPos)
$common += $textArgs""")

# text-kommandot
rep("""    'repl' {""", """    'text' {
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
    'repl' {""")
io.open(P, "w", encoding="ascii", newline="").write(s)

j = json.load(open(J, encoding="utf-8"))
j["defaults"]["upscale"] = True
io.open(J, "w", encoding="utf-8", newline="\n").write(json.dumps(j, indent=2, ensure_ascii=False) + "\n")
print("OK")
