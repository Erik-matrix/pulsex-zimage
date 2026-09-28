# Provtar systemets COMMITTADE minne och varje relevant process PRIVATA minne (= dess commit) tva ganger
# i sekunden tills filen <stop> finns. Skriver CSV: tid, system_commit_MB, och en kolumn per process.
#   pwsh -File tools\mem_sampler.ps1 <ut.csv> <stoppfil>
param([string] $OutCsv, [string] $StopFile)
$names = @('python', 'zimage-dit-stream', 'llama-server', 'taef1_decode', 'powershell', 'pwsh')
$pc = New-Object System.Diagnostics.PerformanceCounter('Memory', 'Committed Bytes')
$null = $pc.NextValue()
"t_s,system_commit_MB," + ($names -join '_MB,') + "_MB" | Out-File $OutCsv -Encoding ascii
$t0 = Get-Date
while (-not (Test-Path $StopFile)) {
    $row = @(('{0:N1}' -f ((Get-Date) - $t0).TotalSeconds).Replace(',', '.'), [int]($pc.NextValue() / 1MB))
    foreach ($n in $names) {
        $sum = (Get-Process $n -EA SilentlyContinue | Measure-Object PrivateMemorySize64 -Sum).Sum
        $row += [int]($sum / 1MB)
    }
    ($row -join ',') | Out-File $OutCsv -Append -Encoding ascii
    Start-Sleep -Milliseconds 500
}
