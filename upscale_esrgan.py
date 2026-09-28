# -*- coding: utf-8 -*-
# Minimal ESRGAN-uppskalare pa NPU via ORT-QNN. esrgan256: 256x256 -> 1024x1024.
# Provar forkompilerad EPContext forst (snabb start), faller tillbaka pa esrgan.onnx.
import os, sys, time
import numpy as np
from PIL import Image

D      = r"C:\PulseCore\models\zImage\esrgan256-onnx-w8a16"
QNNDIR = r"C:\PulseCore\tools\qnn\2.46-runtime"
src, dst = sys.argv[1], sys.argv[2]

import onnxruntime as ort
try: ort.disable_telemetry_events()
except Exception: pass
for d in (QNNDIR, os.path.join(os.path.dirname(ort.__file__), "capi")):
    if os.path.isdir(d): os.add_dll_directory(d)

so = ort.SessionOptions(); so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
sess = None
for cand, vad in ((os.path.join(D, "esrgan_ctx.onnx"), "forkompilerad kontext"),
                  (os.path.join(D, "esrgan.onnx"),      "okompilerad onnx")):
    if not os.path.exists(cand): continue
    try:
        t0 = time.perf_counter()
        sess = ort.InferenceSession(cand, so, providers=["QNNExecutionProvider"],
                                    provider_options=[{"backend_path": os.path.join(QNNDIR, "QnnHtp.dll")}])
        print(f"[esrgan] {vad}: session {time.perf_counter()-t0:.2f} s"); break
    except Exception as e:
        print(f"[esrgan] {vad} gick inte: {str(e)[:160]}")
if sess is None: sys.exit("ingen laddbar graf")

# w8a16: kontexten vill ha uint16. Skalorna LASES ur esrgan.onnx (aldrig hardkodade -
# fel skala ger FEL FARGER, inte en krasch).
import onnx
m = onnx.load(os.path.join(D, "esrgan.onnx"), load_external_data=False)
init = {}
for t in m.graph.initializer:            # vikterna ligger externt i esrgan.data och
    try: init[t.name] = onnx.numpy_helper.to_array(t)   # kan inte lasas - skalorna kan
    except Exception: pass
i0, o0 = m.graph.input[0].name, m.graph.output[0].name
dq = next(n for n in m.graph.node if n.op_type == "DequantizeLinear" and n.input[0] == i0)
qz = next(n for n in m.graph.node if n.op_type == "QuantizeLinear" and n.output[0] == o0)
si, zi = float(init[dq.input[1]]), float(init[dq.input[2]])
so_, zo = float(init[qz.input[1]]), float(init[qz.input[2]])
print(f"[esrgan] kvant in s={si:.6g} zp={zi:g} | ut s={so_:.6g} zp={zo:g}")

im = Image.open(src).convert("RGB")
if im.size != (256, 256): im = im.resize((256, 256), Image.LANCZOS)
x = (np.asarray(im, np.float32) / 255.0).transpose(2, 0, 1)[None]
q = np.clip(np.rint(x / si) + zi, 0, 65535).astype(np.uint16)
iname = sess.get_inputs()[0].name
t0 = time.perf_counter(); y = sess.run(None, {iname: q})[0]; dt = time.perf_counter() - t0
y = np.clip((y.astype(np.float32) - zo) * so_, 0.0, 1.0)[0].transpose(1, 2, 0)
Image.fromarray((y * 255 + 0.5).astype(np.uint8)).save(dst, quality=95)
print(f"[esrgan] {im.size} -> {y.shape[1]}x{y.shape[0]} pa {dt*1000:.0f} ms -> {dst}")
