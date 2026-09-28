# -*- coding: utf-8 -*-
# KAND-FRISK KONTROLL: encoda riktiga bilder med zImage-VAE:ns encoder och mat
# latentens hogfrekvens-innehall med EXAKT samma matt som pa vara DiT-latenter.
# Fragan: ar ~28% hf normalt for en naturlig bild, eller ar vara latenter anomala?
import sys, json, struct
import numpy as np, torch, torch.nn.functional as F
from PIL import Image

VF = r"C:\PulseCore\models\zImage\vae\ae.safetensors"
SCALE, SHIFT, G, EPS = 0.3611, 0.1159, 32, 1e-6

with open(VF, "rb") as fh:
    n = struct.unpack("<Q", fh.read(8))[0]; hdr = json.loads(fh.read(n)); base = 8 + n
    W = {}
    for k, m in hdr.items():
        if k == "__metadata__": continue
        o0, o1 = m["data_offsets"]; fh.seek(base + o0)
        dt = {'F32': np.float32, 'F16': np.float16}[m["dtype"]]
        W[k] = torch.from_numpy(np.frombuffer(fh.read(o1 - o0), dtype=dt).reshape(m["shape"]).astype(np.float32).copy())

def has(k): return k in W
def conv(x, p, stride=1, pad=1): return F.conv2d(x, W[p + ".weight"], W[p + ".bias"], stride=stride, padding=pad)
def gn(x, p): return F.group_norm(x, G, W[p + ".weight"], W[p + ".bias"], EPS)

def resnet(x, p):
    h = conv(F.silu(gn(x, p + ".norm1")), p + ".conv1")
    h = conv(F.silu(gn(h, p + ".norm2")), p + ".conv2")
    s = conv(x, p + ".nin_shortcut", pad=0) if has(p + ".nin_shortcut.weight") else x
    return s + h

def attn(x, p):
    B, C, H, Wd = x.shape; h = gn(x, p + ".norm")
    q = conv(h, p + ".q", pad=0).reshape(B, C, H * Wd).permute(0, 2, 1)
    k = conv(h, p + ".k", pad=0).reshape(B, C, H * Wd)
    v = conv(h, p + ".v", pad=0).reshape(B, C, H * Wd).permute(0, 2, 1)
    a = torch.softmax((q @ k) * (C ** -0.5), -1)
    o = (a @ v).permute(0, 2, 1).reshape(B, C, H, Wd)
    return x + conv(o, p + ".proj_out", pad=0)

def encode(img):                                   # img [1,3,H,W] i [-1,1]
    h = conv(img, "encoder.conv_in")
    for lvl in range(4):
        for b in range(2):
            h = resnet(h, f"encoder.down.{lvl}.block.{b}")
        dp = f"encoder.down.{lvl}.downsample.conv"
        if has(dp + ".weight"):                    # asymmetrisk pad (0,1,0,1) som i ldm
            h = F.conv2d(F.pad(h, (0, 1, 0, 1)), W[dp + ".weight"], W[dp + ".bias"], stride=2, padding=0)
    h = resnet(h, "encoder.mid.block_1"); h = attn(h, "encoder.mid.attn_1"); h = resnet(h, "encoder.mid.block_2")
    h = conv(F.silu(gn(h, "encoder.norm_out")), "encoder.conv_out")
    if has("quant_conv.weight"):
        h = F.conv2d(h, W["quant_conv.weight"], W["quant_conv.bias"])
    mean = h[:, :16]                               # deterministisk: ta mean, ej sample
    return (mean - SHIFT) * SCALE                  # diffusers-konvention

def box3(z):
    p = np.pad(z, ((0, 0), (1, 1), (1, 1)), mode='edge'); s = np.zeros_like(z)
    for dy in range(3):
        for dx in range(3): s += p[:, dy:dy + z.shape[1], dx:dx + z.shape[2]]
    return s / 9.0

def stats(lat, name):
    hf = lat - box3(lat)
    a = lat[:, :, :-1].ravel(); b = lat[:, :, 1:].ravel()
    print(f"{name:34s} hf={hf.std()/lat.std()*100:5.1f}%  granne-korr={np.corrcoef(a,b)[0,1]:.4f}  std={lat.std():.3f}")

for path in sys.argv[1:]:
    im = Image.open(path).convert("RGB")
    s = min(im.size); im = im.crop(((im.width-s)//2, (im.height-s)//2, (im.width+s)//2, (im.height+s)//2)).resize((512, 512), Image.LANCZOS)
    x = torch.from_numpy(np.asarray(im, np.float32) / 127.5 - 1.0).permute(2, 0, 1)[None]
    with torch.no_grad():
        lat = encode(x)[0].numpy().astype(np.float32)
    stats(lat, path.split("\\")[-1][:32])
