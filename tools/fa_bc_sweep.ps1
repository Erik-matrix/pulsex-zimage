# Flash Br/Bc-svep med den residenta K/V-layoutens lediga VTCM (se ggml-hexagon/tools_local/fa_rkv_vtcm.py).
# En 1024-bild, ETT steg, per-op-profil; varje konfiguration N ganger, interleavat, med 512/704 (forval)
# som kand kontroll i samma svep. Vittnet ar motorns egen "fa-tiling"-rad for S=4113.
#   pwsh -File tools\fa_bc_sweep.ps1 [-N 2]
param([int] $N = 2, [string[]] $Configs = @('512x704', '512x832', '576x832', '640x704', '640x832', '768x704'))
$ErrorActionPreference = 'Stop'
$Configs = @($Configs | ForEach-Object { $_ -split ',' } | Where-Object { $_ })   # -File ger EN strang
$Out = 'D:\prov\2026-09-27_fabc'
New-Item -ItemType Directory -Force $Out | Out-Null
$Z = 'C:\PulseCore\PulseX\zimage_dit'
$env:GGML_HEXAGON_PROFILE = '1'
try {
    for ($i = 1; $i -le $N; $i++) {
        foreach ($c in $Configs) {
            $br, $bc = $c.Split('x')
            $env:GGML_HEXAGON_FA_BR = $br
            $env:GGML_HEXAGON_FA_BC = $bc
            $tag = "${c}_$i"
            $env:ZI_ENGINE_LOG = "$Out\engine_$tag.log"
            & "$Z\zimage.exe" make 'a photo of a red fox in autumn forest' -Out "$Out\$tag.jpg" -Size 1024 -Steps 1 2>&1 | Out-Null
            $fa  = (& python "$Z\tools\prof_summary.py" "$Out\engine_$tag.log" 2>$null | Select-String '^FLASH' | Select-Object -First 1)
            $wit = (Select-String -Path "$Out\engine_$tag.log" -Pattern 'fa-tiling qo 4113' | Select-Object -First 1)
            $ovr = (Select-String -Path "$Out\engine_$tag.log" -Pattern 'fa-override' | Select-Object -First 1)
            $md5 = if (Test-Path "$Out\$tag.jpg") { (Get-FileHash -Algorithm MD5 "$Out\$tag.jpg").Hash.Substring(0, 6).ToLower() } else { 'INGEN BILD' }
            "$tag | $fa | md5 $md5" | Tee-Object -Append "$Out\sweep.log"
            if ($wit) { "    " + ($wit.Line -replace '.*fa-tiling ', '') | Tee-Object -Append "$Out\sweep.log" }
            if ($ovr) { "    " + $ovr.Line | Tee-Object -Append "$Out\sweep.log" }
        }
    }
} finally {
    foreach ($v in 'GGML_HEXAGON_PROFILE', 'GGML_HEXAGON_FA_BR', 'GGML_HEXAGON_FA_BC', 'ZI_ENGINE_LOG') { Set-Item "env:$v" $null -EA SilentlyContinue }
    'SVEP KLART' | Tee-Object -Append "$Out\sweep.log"
}
