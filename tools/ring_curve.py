# 09-28: kurvan tid/minne mot viktbudget (ZI_WEIGHT_BUDGET_MB, buffertringen).
# Forbered en katalog med zimage.py --keep (samma prompt/seed), kor sedan DiT-motorn direkt per
# budget. Mater: DiT-vaggtid, [budget]-raderna, systemets committed-topp over forlaget och
# processens toppcommit, samt latentens avvikelse mot en obegransad korning. Obegransad kors TVA
# ganger: HTP0 ar inte bitdeterministisk, sa spridningen dem emellan ar golvet (kand-frisk kontroll).
#   python tools/ring_curve.py <workdir> <outjson> <budget,budget,...> [slots]
import ctypes, json, os, subprocess, sys, threading, time
from ctypes import wintypes
from pathlib import Path
import numpy as np

BIN = Path(r"C:/PulseCore/PulseX/ggml-hexagon/build-wos/bin")
DIT = Path(r"C:/PulseCore/models/zImage/z-image-turbo-q4_0.gguf")
wd, outj = Path(sys.argv[1]), Path(sys.argv[2])
budgets = [int(b) for b in sys.argv[3].split(",")]
slots = sys.argv[4] if len(sys.argv) > 4 else "2"


class _PI(ctypes.Structure):
    _fields_ = ([("cb", ctypes.c_ulong)]
                + [(k, ctypes.c_size_t) for k in ("CommitTotal", "CommitLimit", "CommitPeak", "PhysicalTotal",
                   "PhysicalAvailable", "SystemCache", "KernelTotal", "KernelPaged", "KernelNonpaged", "PageSize")]
                + [(k, ctypes.c_ulong) for k in ("HandleCount", "ProcessCount", "ThreadCount")])


def commit_mb():
    p = _PI(); p.cb = ctypes.sizeof(p)
    ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(p), p.cb)
    return p.CommitTotal * p.PageSize / 2**20


class _PMC(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + \
               [(k, ctypes.c_size_t) for k in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage",
                "PeakPagefileUsage")]


def proc_peak_mb(h):
    m = _PMC(); m.cb = ctypes.sizeof(m)
    ctypes.windll.psapi.GetProcessMemoryInfo(wintypes.HANDLE(h), ctypes.byref(m), m.cb)
    return m.PeakPagefileUsage / 2**20


def run(budget):
    env = {**os.environ, "ADSP_LIBRARY_PATH": str(BIN), "ZI_FLASH": "1", "ZI_DUMP": "0", "ZI_STEPS": "4",
           "ZI_SHIFT": "1", "GGML_HEXAGON_PD_DUMP": "1", "ZI_RING_SLOTS": slots,
           "ZI_WEIGHT_BUDGET_MB": "" if budget < 0 else str(budget)}
    (wd / "lat_out.f32").unlink(missing_ok=True)
    c0 = commit_mb()
    peak = [c0]
    stop = threading.Event()

    def sample():
        while not stop.is_set():
            peak[0] = max(peak[0], commit_mb())
            time.sleep(0.05)
    th = threading.Thread(target=sample, daemon=True); th.start()
    t0 = time.perf_counter()
    p = subprocess.Popen([str(BIN / "zimage-dit-stream.exe"), str(DIT), str(wd), "HTP0"],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    err = []
    te = threading.Thread(target=lambda: err.extend(p.stderr), daemon=True); te.start()
    t_lat = None
    for line in p.stdout:
        if line.startswith("[resident] klar") and t_lat is None:
            t_lat = time.perf_counter() - t0
    rc = p.wait()
    t_all = time.perf_counter() - t0
    te.join()
    ppk = proc_peak_mb(p._handle)
    stop.set(); th.join()
    bud = [l.split("[budget]")[1].strip() for l in err if "[budget]" in l and "fas:" not in l]
    lat = np.fromfile(wd / "lat_out.f32", dtype=np.float32) if rc == 0 else None
    return dict(budget=budget, rc=rc, t_lat=t_lat, t_all=t_all, commit_peak_delta=peak[0] - c0,
                proc_peak=ppk, budget_lines=bud, tail="".join(err[-6:]) if rc else ""), lat


res, ref = [], None
TAG = os.environ.get("RING_TAG", "")
if os.environ.get("RING_REF"):          # extern referens (t.ex. korning utan packcache)
    ref = np.load(os.environ["RING_REF"]).astype(np.float32)
order = [budgets[0]] + budgets  # forsta (obegransad) tva ganger: kontroll
for i, b in enumerate(order):
    r, lat = run(b)
    if lat is not None:
        if ref is None:
            ref = lat.copy()
        else:
            r["relL2"] = float(np.linalg.norm(lat - ref) / np.linalg.norm(ref))
            r["maxabs"] = float(np.abs(lat - ref).max())
        np.save(wd / ("lat_%sb%s_%d.npy" % (TAG, b, i)), lat)
    res.append(r)
    print("budget %6s rc=%d | DiT %.1f s (process %.1f s) | commit +%.0f MB (process topp %.0f MB) | relL2 %s" % (
        "inf" if b < 0 else b, r["rc"], r["t_lat"] or -1, r["t_all"], r["commit_peak_delta"], r["proc_peak"],
        "%.2e" % r["relL2"] if "relL2" in r else "-"), flush=True)
    for l in r["budget_lines"]:
        if l.startswith(("ring:", "minne:")) or " block |" in l or l.startswith("vikter:"):
            print("      " + l, flush=True)
    if r["rc"]:
        print(r["tail"]); break
    json.dump(res, open(outj, "w"), indent=1)
    time.sleep(2)
