#!/bin/sh
# Sveper flash-attn Bc och mater ut flash-tid + total DSP-tid per val.
# Br lamnas pa kostnadsmodellens 256 (g_br=256 => row_vecs=4 = n_tradar).
BIN=/c/PulseCore/PulseX/ggml-hexagon/build-wos/bin
OUT=/d/prov/2026-09-25_fa-bc-sweep
mkdir -p "$OUT"
GGUF='D:\prov\2026-09-22_zimage-gguf\z-image-turbo-q8_0.gguf'
DIR='D:\prov\2026-09-24_fresh-golden\stream_in'

echo "exe  $(stat -c %y $BIN/zimage-dit-stream.exe)"
echo "dll  $(stat -c %y $BIN/ggml-hexagon.dll)"
echo "so   $(stat -c %y $BIN/libggml-htp-v73.so)"
echo
printf "%6s %10s %6s %8s %11s %11s\n" Bc kv_blocks pad% "flash_ms" "mul_mat_ms" "batch_ms"

for BC in 0 1024 832 704 576 448; do
  log="$OUT/bc_$BC.log"
  ADSP_LIBRARY_PATH='C:\PulseCore\PulseX\ggml-hexagon\build-wos\bin' \
  GGML_HEXAGON_PROFILE=1 GGML_HEXAGON_FA_BC=$BC \
  ZI_FLASH=1 ZI_STEPS=1 ZI_DUMP=0 \
  "$BIN/zimage-dit-stream.exe" "$GGUF" "$DIR" HTP0 > "$log" 2>&1
  python - "$log" "$BC" <<'PY'
import re,sys
log,bc=sys.argv[1],sys.argv[2]
tot={}; n={}; til=None
for ln in open(log,encoding="utf8",errors="ignore"):
    if "fa-tiling" in ln and " qo 4109 " in ln and til is None:
        m=re.search(r"Br (\d+) Bc (\d+) g_br (\d+) kv_blocks (\d+)",ln)
        if m: til=tuple(int(x) for x in m.groups())
    if "profile-op" not in ln: continue
    m=re.search(r"profile-op (\S+)\|.*?usec (\d+)",ln)
    if not m: continue
    o=m.group(1).split("|")[0]
    tot[o]=tot.get(o,0)+int(m.group(2)); n[o]=n.get(o,0)+1
if til is None:
    print(f"{bc:>6} {'?':>10} {'?':>6} {'MISSLYCKAD':>8}"); raise SystemExit
Br,Bc,g_br,kvb=til
pad=100.0*(kvb*Bc)/4109-100
print(f"{Bc:6d} {kvb:10d} {pad:5.1f}% {tot.get('FLASH_ATTN_EXT',0)/1000:8.0f} {tot.get('MUL_MAT',0)/1000:11.0f} {tot.get('OPBATCH',0)/1000:11.0f}")
PY
done
