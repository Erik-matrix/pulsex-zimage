#!/bin/sh
# En flash-matning. $1 = etikett, $2 = antal korningar (forval 3), resten = extra env.
# Skriver en rad per korning: flash_us mul_mat_us, sedan median.
# Spridning N>=3 enligt [[npu-runs-are-nondeterministic-validate-with-spread]].
BIN=/c/PulseCore/PulseX/ggml-hexagon/build-wos/bin
DIT='C:\PulseCore\models\zImage\z-image-turbo-q4_0.gguf'
IN='D:\prov\2026-09-24_fresh-golden\stream_in'
OUT=/d/prov/2026-09-26_flash
mkdir -p "$OUT"
LBL="${1:-base}"; N="${2:-3}"

echo "== $LBL =="
echo "   so  $(stat -c %y $BIN/libggml-htp-v73.so | cut -c1-19)"
echo "   exe $(stat -c %y $BIN/zimage-dit-stream.exe | cut -c1-19)"
i=1
while [ "$i" -le "$N" ]; do
  log="$OUT/${LBL}_$i.log"
  ADSP_LIBRARY_PATH='C:\PulseCore\PulseX\ggml-hexagon\build-wos\bin' \
  GGML_HEXAGON_PROFILE=1 ZI_FLASH=1 ZI_STEPS=1 ZI_DUMP=0 \
  "$BIN/zimage-dit-stream.exe" "$DIT" "$IN" HTP0 > "$log" 2>&1
  awk -v n="$i" '
    /profile-op FLASH_ATTN_EXT\|/ { if (match($0, /usec [0-9]+/)) f += substr($0, RSTART+5, RLENGTH-5) }
    /profile-op MUL_MAT\|/        { if (match($0, /usec [0-9]+/)) m += substr($0, RSTART+5, RLENGTH-5) }
    END { printf "   %d  flash %7d us   mul_mat %7d us\n", n, f, m }' "$log"
  i=$((i+1))
done
