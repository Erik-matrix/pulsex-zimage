# 09-28: DiT-compute per bildform. Samma prompt (cap_raw ur en fardig katalog), olika storlekar;
# skiljer "antal token" fran "form" (kvadrat mot bred, jamnt/udda Wt) och flash-kaklingens granser.
#   python tools/shape_probe.py <katalog med cap_raw.bin> <storlek> [storlek ...]
import os, re, subprocess, sys, time
from pathlib import Path
import numpy as np

sys.argv, args = sys.argv[:1], sys.argv[1:]
import zimage as z

src = Path(args[0])
cap = np.fromfile(src / "cap_raw.bin", dtype=np.float32).reshape(-1, z.CAPD)
env = {**os.environ, "ADSP_LIBRARY_PATH": str(z.BIN), "ZI_FLASH": "1", "ZI_DUMP": "0", "ZI_STEPS": "4",
       "ZI_SHIFT": "1", "GGML_HEXAGON_PD_DUMP": "1"}
for sz in args[1:]:
    px = z.parse_size(sz)
    wd = Path(r"D:/prov/2026-09-28_ring/shape_" + z.size_str(px))
    wd.mkdir(parents=True, exist_ok=True)
    su = z.build_inputs(cap, px, 7, wd)
    t0 = time.perf_counter()
    r = subprocess.run([str(z.BIN / "zimage-dit-stream.exe"), str(z.DIT), str(wd), "HTP0"], capture_output=True,
                       text=True, env=env)
    dt = time.perf_counter() - t0
    m = re.search(r"compute ([\d.]+)", r.stderr)
    comp = float(m.group(1)) if m else float("nan")
    print("%-9s %5d token | process %5.1f s | compute %5.1f s | %.2f ms/token/steg | rc %d" % (
        z.size_str(px), su, dt, comp, comp * 1000 / su / 4, r.returncode), flush=True)
