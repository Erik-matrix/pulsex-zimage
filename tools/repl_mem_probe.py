# 09-28: REPL-provet med minneskurva. Styr `zimage.py --serve` via stdin som en anvandare:
# bild 1 -> vila -> bild 2 -> vila -> :q. Samplar systemets committed och cached var 100 ms
# och markerar varje handelse. Encodern forstartas INTE vid prompten (ZI_ENCODER_PRESTART=0),
# sa viloperioderna visar det du ser nar du inte skriver.
#   python tools/repl_mem_probe.py <config.json> <utkatalog> <etikett> "<rad1>" "<rad2>" [vila_s]
import ctypes, json, os, subprocess, sys, threading, time
from pathlib import Path

cfg, outd, tag, line1, line2 = sys.argv[1:6]
cfg = str(Path(cfg).resolve())   # REPL:n startar i utkatalogen
rest = float(sys.argv[6]) if len(sys.argv) > 6 else 10.0
outd = Path(outd)
outd.mkdir(parents=True, exist_ok=True)


class _PI(ctypes.Structure):
    _fields_ = ([("cb", ctypes.c_ulong)]
                + [(k, ctypes.c_size_t) for k in ("CommitTotal", "CommitLimit", "CommitPeak", "PhysicalTotal",
                   "PhysicalAvailable", "SystemCache", "KernelTotal", "KernelPaged", "KernelNonpaged", "PageSize")]
                + [(k, ctypes.c_ulong) for k in ("HandleCount", "ProcessCount", "ThreadCount")])


def mem():
    p = _PI(); p.cb = ctypes.sizeof(p)
    ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(p), p.cb)
    return p.CommitTotal * p.PageSize / 2**30, p.SystemCache * p.PageSize / 2**30


T0 = time.perf_counter()
samples, marks, out = [], [], []
stop = threading.Event()


def sampler():
    while not stop.is_set():
        c, k = mem()
        samples.append((time.perf_counter() - T0, c, k))
        time.sleep(0.1)


def mark(what):
    c, k = mem()
    marks.append((time.perf_counter() - T0, what, c, k))
    print("%6.1f s  %-26s committed %5.2f GB  cached %5.2f GB" % (marks[-1][0], what, c, k), flush=True)


threading.Thread(target=sampler, daemon=True).start()
time.sleep(2)
mark("fore start")
env = {**os.environ, "ZIMAGE_CONFIG": cfg, "ZI_ENCODER_PRESTART": "0", "PYTHONUNBUFFERED": "1"}
p = subprocess.Popen([sys.executable, str(Path(__file__).resolve().parent.parent / "zimage.py"), "--serve",
                      "-o", str(outd / (tag + ".jpg"))], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                     stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", env=env, cwd=str(outd))
saved = threading.Semaphore(0)


def reader():
    for ln in p.stdout:
        out.append(ln)
        if "saved" in ln:
            saved.release()


threading.Thread(target=reader, daemon=True).start()
time.sleep(3)
mark("repl uppe")
for i, ln in enumerate((line1, line2), 1):
    mark("skickar bild %d" % i)
    if p.poll() is not None:
        print("".join(out)); sys.exit("REPL:n dog (rc %d)" % p.returncode)
    p.stdin.write(ln + "\n"); p.stdin.flush()
    if not saved.acquire(timeout=300):
        print("".join(out[-20:])); p.kill(); sys.exit(1)
    mark("bild %d sparad" % i)
    time.sleep(rest)
    mark("vila %.0f s efter bild %d" % (rest, i))
p.stdin.write(":q\n"); p.stdin.flush()
p.wait(timeout=120)
mark("repl avslutad")
time.sleep(3)
mark("3 s efter avslut")
stop.set()

# toppar per fas (mellan markeringarna)
print()
for (ta, wa, *_), (tb, wb, *_) in zip(marks, marks[1:]):
    seg = [s for s in samples if ta <= s[0] <= tb]
    if seg:
        print("  %-26s -> %-26s topp committed %5.2f GB, cached %5.2f GB" % (
            wa, wb, max(s[1] for s in seg), max(s[2] for s in seg)))
print()
for ln in out:
    if any(k in ln for k in ("[1/", "[2/", "[3/", "saved", "idle", "bye", "Error", "Trace")):
        print("  | " + ln.rstrip())
json.dump({"samples": samples, "marks": marks}, open(outd / (tag + "_mem.json"), "w"))
