# sr_compare_1001.py — 2026-10-01: QuickSRNet against Real-ESRGAN as the 512 -> 1024 upscaler of Z-Image. SAME 512 inputs (the A/B run's cat + sailor, UltraReal 0.6, seed 1234), so only the upscaler differs.
#   A  esrgan d=0.5  : today's default (Real-ESRGAN x4 -> 2048 -> Lanczos 1024, mixed 50/50 with a Lanczos of the 512)
#   B  qsr    d=0.5  : QuickSRNet-Large w8a8 x4 (LTX's upscaler since 09-30), the same mix
#   C  qsr    d=1.0  : QuickSRNet alone
#   L  lanczos       : plain enlargement (reference for colour and for "how much detail was added")
# Measured: session open, inference (mean of 5 after a warm-up), Laplace variance (detail), a noise estimate on flat
# areas (grain), per-channel colour against the Lanczos. Crops side by side for the eye: fur, whiskers, eye, beard, hand.
import os, sys, time, json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "ltx_video"))
os.environ.setdefault("ZIMAGE_CONFIG", str(HERE / "zimage.json"))
import zimage as Z                       # esrgan_session(), QNN_RT (reads zimage.json; main() is not run)
import upscale_frames as U               # qsr_model(), qsr_session()

SRC = Path(r"D:\prov\2026-09-30_zimage_ab_bin")
OUT = Path(r"D:\prov\2026-10-01_zimage_sr")
OUT.mkdir(parents=True, exist_ok=True)
IMGS = {"cat": SRC / "cat_512_v2_0.png", "sailor": SRC / "sailor_512_v2_0.png"}
T = 1024


def lanczos(im):
    return np.asarray(Image.fromarray(im, "RGB").resize((T, T), Image.LANCZOS), dtype=np.float32)


def finish(big01, im, d):
    """big01: 2048 x 2048 x 3 float 0..1 -> 1024, mixed with the Lanczos of the 512 like zimage.upscale_x4"""
    sr = np.asarray(Image.fromarray(np.rint(np.clip(big01, 0, 1) * 255).astype(np.uint8), "RGB").resize((T, T), Image.LANCZOS),
                    dtype=np.float32)
    return np.clip(np.rint(d * sr + (1 - d) * lanczos(im)), 0, 255).astype(np.uint8)


def lapvar(a):
    g = a.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
    lap = -4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    return float(lap.var())


def grain(a, ref):
    """high-frequency energy on the FLATTEST 20 % of 16x16 blocks of the reference (sky, blur): added noise there
    is grain, not detail"""
    g = a.astype(np.float32).mean(2); r = ref.astype(np.float32).mean(2)
    hp = g - np.asarray(Image.fromarray(g).resize((T // 4, T // 4), Image.BOX).resize((T, T), Image.BILINEAR))
    bs = 16; nb = T // bs
    rv = r.reshape(nb, bs, nb, bs).std(axis=(1, 3)); hv = (hp ** 2).reshape(nb, bs, nb, bs).mean(axis=(1, 3))
    flat = rv <= np.quantile(rv, 0.2)
    return float(np.sqrt(hv[flat].mean()))


def timed(fn, n=5):
    fn()                                              # warm-up
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); y = fn(); ts.append(time.perf_counter() - t0)
    return y, float(np.mean(ts)), float(np.min(ts))


res = {}
# --- ESRGAN (today) ---
t0 = time.perf_counter(); esr = Z.esrgan_session(); t_open_esr = time.perf_counter() - t0
# --- QuickSRNet 512x512 ---
base = U.qsr_base() if hasattr(U, "qsr_base") else Path(r"C:\PulseCore\models\sr-qnn\qsrl_w8a8\quicksrnetlarge_288x512.onnx")
qm = U.qsr_model(base, 512, 512)
t0 = time.perf_counter(); qs = U.qsr_session(qm, Z.QNN_RT); t_open_qsr = time.perf_counter() - t0
print("session open: esrgan %.2f s, quicksrnet %.2f s" % (t_open_esr, t_open_qsr), flush=True)

for name, p in IMGS.items():
    im = np.asarray(Image.open(p).convert("RGB"))
    x = np.ascontiguousarray((im.astype(np.float32) / 255.0).transpose(2, 0, 1)[None])
    y_e, te, te_min = timed(lambda: esr.run(None, {"image": x})[0])
    big_e = y_e[0].transpose(1, 2, 0)
    y_q, tq, tq_min = timed(lambda: U.qsr_frame(qs, im))
    L = np.clip(np.rint(lanczos(im)), 0, 255).astype(np.uint8)
    V = {"A_esrgan_d05": finish(big_e, im, 0.5), "B_qsr_d05": finish(y_q, im, 0.5), "C_qsr_d10": finish(y_q, im, 1.0),
         "L_lanczos": L}
    r = {"infer_s": {"esrgan": round(te, 3), "esrgan_min": round(te_min, 3), "quicksrnet": round(tq, 4), "quicksrnet_min": round(tq_min, 4)}}
    for k, a in V.items():
        Image.fromarray(a).save(OUT / ("%s_%s.png" % (name, k)))
        ch = [round(float(a[..., c].mean() / max(L[..., c].mean(), 1e-6)), 4) for c in range(3)]
        r[k] = {"lapvar": round(lapvar(a), 1), "grain": round(grain(a, L), 3), "colour_vs_lanczos": ch}
    res[name] = r
    print(name, json.dumps(r), flush=True)

# crops side by side (2x nearest so pixels show): rows = regions, columns = A, B, C, L
CROPS = {"cat": [("fur", (430, 230, 630, 430)), ("whiskers", (170, 620, 370, 820)), ("eye", (330, 420, 490, 540))],
         "sailor": [("beard", (430, 300, 630, 460)), ("hand", (250, 470, 410, 630)), ("face", (420, 170, 620, 330))]}
cols = ["A_esrgan_d05", "B_qsr_d05", "C_qsr_d10", "L_lanczos"]
heads = ["ESRGAN 0.5 (today)", "QuickSRNet 0.5", "QuickSRNet 1.0", "Lanczos only"]
for name, crops in CROPS.items():
    tiles = []
    for label, box in crops:
        row = [Image.open(OUT / ("%s_%s.png" % (name, c))).crop(box) for c in cols]
        w, h = row[0].size
        row = [t.resize((w * 2, h * 2), Image.NEAREST) for t in row]
        tiles.append((label, row))
    W2, H2 = tiles[0][1][0].size
    sheet = Image.new("RGB", (4 * W2 + 5 * 8, len(tiles) * (H2 + 8) + 40), (24, 24, 28))
    d = ImageDraw.Draw(sheet)
    for j, hname in enumerate(heads):
        d.text((8 + j * (W2 + 8), 12), hname, fill=(230, 230, 230))
    for i, (label, row) in enumerate(tiles):
        for j, t in enumerate(row):
            sheet.paste(t, (8 + j * (W2 + 8), 36 + i * (H2 + 8)))
    sheet.save(OUT / ("%s_crops.png" % name))
res["session_open_s"] = {"esrgan": round(t_open_esr, 2), "quicksrnet": round(t_open_qsr, 2)}
(OUT / "sr_result.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
print("DONE")
