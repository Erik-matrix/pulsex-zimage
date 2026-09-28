#!/bin/sh
# Kalibreringssvep for flash-tilingen. Skriver en CSV med DEN FAKTISKT VALDA (Br,Bc)
# ur fa-tiling-raden, sa rader dar VTCM avvisade overriden kan sorteras bort.
BIN=/c/PulseCore/PulseX/ggml-hexagon/build-wos/bin
OUT=/d/prov/2026-09-25_fa-bc-sweep/calib
mkdir -p "$OUT"
CSV="$OUT/calib.csv"
echo "req_br,req_bc,br,bc,g_br,kv_blocks,sfm_thr,vecs_per_t,vtcm,flash_us" > "$CSV"
echo "exe $(stat -c %y $BIN/zimage-dit-stream.exe)"
echo "so  $(stat -c %y $BIN/libggml-htp-v73.so)"

for BR in 256 384 512 640 768 1024; do
for BC in 256 448 576 704 832 1024 1280; do
  log="$OUT/c_${BR}_${BC}.log"
  ADSP_LIBRARY_PATH='C:\PulseCore\PulseX\ggml-hexagon\build-wos\bin' \
  GGML_HEXAGON_PROFILE=1 GGML_HEXAGON_FA_BR=$BR GGML_HEXAGON_FA_BC=$BC \
  ZI_FLASH=1 ZI_STEPS=1 ZI_DUMP=0 \
  "$BIN/zimage-dit-stream.exe" \
    'D:\prov\2026-09-22_zimage-gguf\z-image-turbo-q8_0.gguf' \
    'D:\prov\2026-09-24_fresh-golden\stream_in' HTP0 > "$log" 2>&1
  python - "$log" "$BR" "$BC" "$CSV" <<'PY'
import re,sys
log,rbr,rbc,csv=sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4]
til=None; us=0
for ln in open(log,encoding="utf8",errors="ignore"):
    if til is None and "fa-tiling" in ln and " qo 4109 " in ln:
        m=re.search(r"Br (\d+) Bc (\d+) g_br (\d+) kv_blocks (\d+) pipe \d+ n_thr \d+ sfm_thr (\d+) vecs_per_t (\d+) vtcm (\d+)",ln)
        if m: til=m.groups()
    m=re.search(r"profile-op FLASH_ATTN_EXT\|.*?usec (\d+)",ln)
    if m: us+=int(m.group(1))
if til is None:
    print(f"  {rbr}x{rbc}: MISSLYCKAD")
else:
    open(csv,"a").write(f"{rbr},{rbc},{','.join(til)},{us}\n")
    ok = "ok " if (til[0]==rbr and til[1]==rbc) else "VTCM"
    print(f"  {rbr:>4}x{rbc:<5} {ok} -> Br {til[0]:>4} Bc {til[1]:>5} kvb {til[3]:>2}  flash {us/1000:6.0f} ms")
PY
done
done
echo "klart -> $CSV"
