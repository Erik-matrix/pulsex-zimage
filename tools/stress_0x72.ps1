# Stresstest for det intermittenta `dspqueue_read failed: 0x72` (AEE_ECONNREFUSED = DSP-PD:n dog).
# Hypotes: kraschen kraver att encoderns llama-server (egen NPU-session) LEVER under DiT-korningen,
# som pa forrmiddagen 09-27 (alla 3 kraschar) - `make` dodar den annars fore DiT.
#   pwsh -File tools\stress_0x72.ps1 -N 7 [-Keep]      (-Keep = encodern lever, annars som make)
# All utdata till FIL: via ett ror arver llama-server rorets handtag och loopen vantar for evigt.
param([int] $N = 7, [switch] $Keep, [string] $Tag = 'keep')
$D = 'D:\prov\2026-09-27_0x72'
New-Item -ItemType Directory -Force $D | Out-Null
Set-Location 'C:\PulseCore\PulseX\zimage_dit'
$sum = "$D\$Tag.summary.txt"
"start $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz') N=$N Keep=$Keep" | Out-File -Append $sum
for ($i = 1; $i -le $N; $i++) {
    $env:ZI_ENGINE_LOG = "$D\${Tag}_$i.log"
    $t0 = Get-Date
    if ($Keep) {
        & .\zimage.exe make 'a photo of a red fox in autumn forest' -Out "$D\${Tag}_$i.jpg" -Size 1024 -Keep *> "$D\${Tag}_$i.out"
    } else {
        & .\zimage.exe make 'a photo of a red fox in autumn forest' -Out "$D\${Tag}_$i.jpg" -Size 1024 *> "$D\${Tag}_$i.out"
    }
    $env:ZI_ENGINE_LOG = $null
    $ok  = Test-Path "$D\${Tag}_$i.jpg"
    $dq  = if (Test-Path "$D\${Tag}_$i.log") { (Select-String -Path "$D\${Tag}_$i.log" -Pattern 'dspqueue_read failed' | Select-Object -First 1).Line } else { 'ingen motorlogg' }
    $srv = (Get-Process llama-server -EA SilentlyContinue | Measure-Object).Count
    "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') {0} bild={1} encoder-levande={2} tid={3:N1}s {4}" -f $i, $ok, $srv, ((Get-Date) - $t0).TotalSeconds, $dq | Out-File -Append $sum
}
"slut $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')" | Out-File -Append $sum
