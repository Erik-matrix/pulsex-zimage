# Som stress_0x72.ps1 -Keep, men med encodern UNDER BELASTNING (tools/hammer_encoder.py) hela
# DiT-korningen: tva NPU-sessioner som faktiskt tavlar om VTCM/HMX.
param([int] $N = 3, [string] $Tag = 'hammer')
$D = 'D:\prov\2026-09-27_0x72'
New-Item -ItemType Directory -Force $D | Out-Null
Set-Location 'C:\PulseCore\PulseX\zimage_dit'
$sum = "$D\$Tag.summary.txt"
"start $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz') N=$N" | Out-File -Append $sum
for ($i = 1; $i -le $N; $i++) {
    $h = Start-Process python -ArgumentList @('tools\hammer_encoder.py', '70', "$D\${Tag}_$i.hammer.txt") -PassThru -WindowStyle Hidden
    Start-Sleep -Seconds 2
    $env:ZI_ENGINE_LOG = "$D\${Tag}_$i.log"
    $t0 = Get-Date
    & .\zimage.exe make 'a photo of a red fox in autumn forest' -Out "$D\${Tag}_$i.jpg" -Size 1024 -Keep *> "$D\${Tag}_$i.out"
    $env:ZI_ENGINE_LOG = $null
    $h.WaitForExit(90000) | Out-Null
    $ok  = Test-Path "$D\${Tag}_$i.jpg"
    $dq  = if (Test-Path "$D\${Tag}_$i.log") { (Select-String -Path "$D\${Tag}_$i.log" -Pattern 'dspqueue_read failed' | Select-Object -First 1).Line } else { 'ingen motorlogg' }
    $hk  = (Get-Content "$D\${Tag}_$i.hammer.txt" -EA SilentlyContinue | Select-Object -Last 1)
    "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') {0} bild={1} tid={2:N1}s | hammare: {3} | {4}" -f $i, $ok, ((Get-Date) - $t0).TotalSeconds, $hk, $dq | Out-File -Append $sum
}
"slut $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')" | Out-File -Append $sum
