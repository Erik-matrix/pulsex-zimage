# Provkorning av gamla ZImage-receptet for 512-laget: 512 -> esrgan_x4 (QNN-kontext via ORT QNN EP,
# QNN 2.46-runtime) -> 2048 -> Lanczos -> 1024. Mater sessionsstart, inferens, och FARGTROHET mot en
# Lanczos-uppskalning av samma bild (sr_x2-selen hade per-kanal affint fargstick, r = 0,96).
#   python tools/esrgan_x4_probe.py <in512.jpg> <utkatalog>
import os, sys, time
import numpy as np
from PIL import Image

SDXL = r"C:\PulseCore\models\sdxl-qnn"
QNN246 = r"C:\PulseCore\tools\qnn\2.46-runtime"
src, outd = sys.argv[1], sys.argv[2]
os.makedirs(outd, exist_ok=True)

t0 = time.perf_counter()
import onnxruntime as ort
try:
    ort.disable_telemetry_events()
except Exception:
    pass
for d in (QNN246, os.path.join(os.path.dirname(ort.__file__), "capi")):
    if os.path.isdir(d):
        os.add_dll_directory(d)
so = ort.SessionOptions()
so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
sess = ort.InferenceSession(os.path.join(SDXL, "esrgan_x4.bin.wrap.onnx"), so, providers=["QNNExecutionProvider"],
                            provider_options=[{"backend_path": os.path.join(QNN246, "QnnHtp.dll")}])
t_sess = time.perf_counter() - t0

im = Image.open(src).convert("RGB")
assert im.size == (512, 512), im.size
x = np.ascontiguousarray((np.asarray(im, dtype=np.float32) / 255.0).transpose(2, 0, 1)[None])
times = []
for _ in range(3):
    t1 = time.perf_counter()
    y = sess.run(None, {"image": x})[0]
    times.append(time.perf_counter() - t1)
a = np.clip(y[0].transpose(1, 2, 0), 0, 1)
big = Image.fromarray(np.rint(a * 255).astype(np.uint8), "RGB")
t2 = time.perf_counter()
out1024 = big.resize((1024, 1024), Image.LANCZOS)
t_lz = time.perf_counter() - t2
big.save(os.path.join(outd, "esr_2048.png"))
out1024.save(os.path.join(outd, "esr_1024.jpg"), quality=95)
lz = im.resize((1024, 1024), Image.LANCZOS)
lz.save(os.path.join(outd, "lanczos_1024.jpg"), quality=95)

e = np.asarray(out1024, dtype=np.float64); l = np.asarray(lz, dtype=np.float64)
print("session %.2f s | inferens %s s | lanczos 2048->1024 %.2f s" % (t_sess, " / ".join("%.3f" % t for t in times), t_lz))
for c, n in enumerate("RGB"):
    ec, lc = e[..., c].ravel(), l[..., c].ravel()
    k, m = np.polyfit(lc, ec, 1)
    r = np.corrcoef(lc, ec)[0, 1]
    print("  %s: medel esr %.1f lanczos %.1f | anpassning esr = %.3f*lz %+.1f | r %.4f" % (n, ec.mean(), lc.mean(), k, m, r))
gy, gx = np.gradient(e.mean(axis=2)); hy, hx = np.gradient(l.mean(axis=2))
print("  hogfrekvens (medel |grad|): esr %.2f  lanczos %.2f" % (np.hypot(gx, gy).mean(), np.hypot(hx, hy).mean()))
