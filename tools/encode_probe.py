# 09-28: provar zimage-encode.exe (encodern som kortlivad NPU-process) mot facit fran den gamla
# encoder-servern (D:/prov/2026-09-28_encoder/ref_*.f32, ref.json). Mater laddtid, kodtid, exitkod
# och committed fore/under/efter.   python tools/encode_probe.py [HTP0|none]
import ctypes, json, subprocess, sys, time
from pathlib import Path
import numpy as np

EXE = r"C:/PulseCore/PulseX/ggml-hexagon/build-wos/bin/zimage-encode.exe"
MODEL = r"C:/PulseCore/models/zImage/qwen3-4b-zimage-q4_0.gguf"
D = Path(r"D:/prov/2026-09-28_encoder")
dev = sys.argv[1] if len(sys.argv) > 1 else "HTP0"


def commit_mb():
    class _PI(ctypes.Structure):
        _fields_ = ([("cb", ctypes.c_ulong)]
                    + [(k, ctypes.c_size_t) for k in ("CommitTotal", "CommitLimit", "CommitPeak", "PhysicalTotal",
                       "PhysicalAvailable", "SystemCache", "KernelTotal", "KernelPaged", "KernelNonpaged", "PageSize")]
                    + [(k, ctypes.c_ulong) for k in ("HandleCount", "ProcessCount", "ThreadCount")])
    p = _PI(); p.cb = ctypes.sizeof(p)
    ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(p), p.cb)
    return p.CommitTotal * p.PageSize / 2**20


ref = json.load(open(D / "ref.json", encoding="utf-8"))
env = dict(__import__("os").environ, ADSP_LIBRARY_PATH=r"C:\PulseCore\PulseX\ggml-hexagon\build-wos\bin")
for i, r in enumerate(ref):
    c0 = commit_mb()
    t0 = time.perf_counter()
    p = subprocess.Popen([EXE, MODEL, dev], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         text=True, env=env)
    line = p.stdout.readline().strip()
    t_ready = time.perf_counter() - t0
    c1 = commit_mb()
    if line != "READY":
        print(i, "fel:", line, p.stderr.read()[-400:]); continue
    tf, of = D / ("txt_%d.txt" % i), D / ("enc_%d.f32" % i)
    tf.write_bytes(r["template"].encode("utf-8"))
    t1 = time.perf_counter()
    p.stdin.write("%s\t%s\n" % (tf, of)); p.stdin.flush()
    ans = p.stdout.readline().strip()
    t_enc = time.perf_counter() - t1
    p.stdin.close()
    rc = p.wait(timeout=30)
    t_exit = time.perf_counter() - t1 - t_enc
    time.sleep(0.5)
    c2 = commit_mb()
    a = np.fromfile(D / ("ref_%d.f32" % i), dtype=np.float32)
    b = np.fromfile(of, dtype=np.float32) if of.exists() else np.zeros(1, np.float32)
    ok = a.size == b.size
    cos = float(a @ b / np.linalg.norm(a) / np.linalg.norm(b)) if ok else float("nan")
    rel = float(np.linalg.norm(a - b) / np.linalg.norm(a)) if ok else float("nan")
    print("%d %-26s %s | redo %.2f s, kodning %.2f s, avslut %.2f s rc=%d | commit laddad %+.0f MB, efter %+.0f MB | "
          "cos %.5f relL2 %.4f (%d tok)" % (i, r["prompt"][:26].encode("ascii", "replace").decode(), ans, t_ready,
                                             t_enc, t_exit, rc, c1 - c0, c2 - c0, cos, rel, b.size // 2560))
