# Summarises a GGML_HEXAGON_PROFILE=1 engine log (ZI_ENGINE_LOG): time per op type, the
# MUL_MAT split by weight shape, the DSP clock seen per op, and op-time vs batch-time.
import collections
import re
import sys

path = sys.argv[1]
ops = collections.defaultdict(lambda: [0, 0])          # op -> [usec, count]
mm = collections.defaultdict(lambda: [0, 0])           # weight shape -> [usec, count]
mhz = collections.Counter()
batch_us = 0
n_batch = 0

# [\w+]+, not \w+: fused ops are named e.g. "RMS_NORM+MUL" and \w+ silently dropped them.
rx = re.compile(r"profile-op ([\w+]+)\|([^|]*)\|([^|]*)\|([^|]*)\|[^|]*\|([^|]*)\|usec (\d+) cycles (\d+) start \d+ mhz ([\d.]+)")
for line in open(path, encoding="utf-8", errors="replace"):
    m = rx.search(line)
    if not m:
        continue
    op, names, dims, types, kp, us, cyc, hz = m.groups()
    us = int(us)
    if op == "OPBATCH":
        batch_us += us
        n_batch += 1
        continue
    ops[op][0] += us
    ops[op][1] += 1
    mhz[round(float(hz) / 50) * 50] += us
    if op == "MUL_MAT":
        w = dims.split(" x ")[0]
        mm[w + "  " + kp.split(" vtcm")[0].strip()][0] += us
        mm[w + "  " + kp.split(" vtcm")[0].strip()][1] += 1

tot = sum(v[0] for v in ops.values())
print("op-tid totalt %.2f s | OPBATCH-tid %.2f s i %d batchar" % (tot / 1e6, batch_us / 1e6, n_batch))
print()
print("%-14s %9s %6s %7s %9s" % ("op", "s", "andel", "antal", "ms/anrop"))
for op, (us, n) in sorted(ops.items(), key=lambda kv: -kv[1][0]):
    print("%-14s %9.2f %5.1f%% %7d %9.2f" % (op, us / 1e6, 100.0 * us / tot, n, us / 1e3 / n))
print()
print("MUL_MAT per viktform (K:M) och karna:")
mtot = ops["MUL_MAT"][0]
for w, (us, n) in sorted(mm.items(), key=lambda kv: -kv[1][0])[:10]:
    print("  %-34s %8.2f s %5.1f%% av MUL_MAT  %5d anrop  %7.2f ms/anrop" % (w, us / 1e6, 100.0 * us / mtot, n, us / 1e3 / n))
print()
print("DSP-klocka viktad med op-tid (MHz, avrundat till 50):")
for hz, us in sorted(mhz.items()):
    print("  %5d MHz  %5.1f%% av op-tiden" % (hz, 100.0 * us / tot))
