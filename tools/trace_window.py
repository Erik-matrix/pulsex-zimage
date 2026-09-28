# Fonsteranalys av en GGML_HEXAGON_PROFILE=3-logg (ZI_ENGINE_LOG): for varje op som matchar filtret
# summeras varje trads sparhandelser INOM op:ens cykelfonster [start, start+cycles) (mod 2^32).
# Loggordning per batch: profile-op-rader (en per op) -> OPBATCH-rad -> trace-evt-rader.
#   python tools/trace_window.py <logg> <op> [viktform-substring]      t.ex.  MUL_MAT 10240:3840
# Utdata: per trad och handelse: total tid (cykler) inne i fonstren, och andel av fonstertiden.
# ⚠ HTP_TRACE_EVT_DMA ar push->pop (flygtid, overlappar berakning), INTE vantan.
import collections
import re
import sys

path, want_op = sys.argv[1], sys.argv[2]
want_shape = sys.argv[3] if len(sys.argv) > 3 else None
M32 = 1 << 32

rx_op = re.compile(r"profile-op ([\w+]+)\|([^|]*)\|([^|]*)\|[^|]*\|[^|]*\|([^|]*)\|usec (\d+) cycles (\d+) start (\d+)")
rx_ev = re.compile(r"trace-evt (\S+): thread (\d+) info (\d+) (start|stop) (\d+)")

windows_total = 0
per = collections.defaultdict(int)          # (thread, evt) -> cycles inside windows
n_windows = 0


def flush(batch_ops, events):
    global windows_total, n_windows
    wins = []
    for op, dims, kp, cyc, st in batch_ops:
        if op != want_op:
            continue
        if want_shape and not dims.split(" x ")[0].startswith(want_shape):
            continue
        wins.append((st, st + cyc))
    if not wins:
        return
    open_ = {}
    ivs = collections.defaultdict(list)     # (thread, evt) -> [(a, b)]
    for name, th, info, kind, c in events:
        k = (th, name)
        if kind == "start":
            open_[k] = c
        elif k in open_:
            a = open_.pop(k)
            b = c if c >= a else c + M32
            ivs[k].append((a, b))
    for (w0, w1) in wins:
        n_windows += 1
        windows_total += w1 - w0
        for k, lst in ivs.items():
            for (a, b) in lst:
                # fonstret och intervallen ligger i samma 32-bitarsdomän; justera for omslag
                for off in (0, M32, -M32):
                    lo, hi = max(a + off, w0), min(b + off, w1)
                    if hi > lo:
                        per[k] += hi - lo


batch_ops, events, in_events = [], [], False
for line in open(path, encoding="utf-8", errors="replace"):
    m = rx_op.search(line)
    if m:
        op, names, dims, kp, us, cyc, st = m.groups()
        if in_events:                         # ny batch borjar
            flush(batch_ops, events)
            batch_ops, events, in_events = [], [], False
        if op == "OPBATCH":
            in_events = True
            continue
        batch_ops.append((op, dims, kp, int(cyc), int(st)))
        continue
    e = rx_ev.search(line)
    if e:
        in_events = True
        events.append((e.group(1), int(e.group(2)), int(e.group(3)), e.group(4), int(e.group(5))))
flush(batch_ops, events)

print("%d fonster (%s %s), totalt %.1f M cykler" % (n_windows, want_op, want_shape or "", windows_total / 1e6))
for (th, name), c in sorted(per.items(), key=lambda kv: (kv[0][0], -kv[1])):
    print("  trad %2d  %-14s %9.1f M cykler  %5.1f %%" % (th, name, c / 1e6, 100.0 * c / max(windows_total, 1)))
