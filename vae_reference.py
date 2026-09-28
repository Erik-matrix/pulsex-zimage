# -*- coding: utf-8 -*-
# VAE-decode-referens: ldm/flux-decoder byggd manuellt i torch ur ae.safetensors (ldm-nycklar).
# Dumpar mellansteg + bild för ggml-verifiering.
import numpy as np, torch, torch.nn.functional as F, struct, json
from pathlib import Path

VF = r"C:\PulseCore\models\zImage\vae\ae.safetensors"
OUT = Path(r"D:\prov\2026-09-24_fresh-golden\vae"); OUT.mkdir(parents=True, exist_ok=True)
SCALE, SHIFT, G, EPS = 0.3611, 0.1159, 32, 1e-6

# ladda safetensors
with open(VF,"rb") as fh:
    n=struct.unpack("<Q",fh.read(8))[0]; hdr=json.loads(fh.read(n)); base=8+n
    W={}
    for k,m in hdr.items():
        if k=="__metadata__": continue
        o0,o1=m["data_offsets"]; fh.seek(base+o0)
        dt={'F32':np.float32,'F16':np.float16}[m["dtype"]]
        W[k]=torch.from_numpy(np.frombuffer(fh.read(o1-o0),dtype=dt).reshape(m["shape"]).astype(np.float32).copy())
def w(k): return W[k]

def conv(x,pfx,k=3): return F.conv2d(x, w(pfx+".weight"), w(pfx+".bias"), padding=k//2)
def gn(x,pfx): return F.group_norm(x, G, w(pfx+".weight"), w(pfx+".bias"), EPS)
def silu(x): return F.silu(x)

def resnet(x,pfx):
    h=conv(silu(gn(x,pfx+".norm1")), pfx+".conv1", 3)
    h=conv(silu(gn(h,pfx+".norm2")), pfx+".conv2", 3)
    s=x if (pfx+".nin_shortcut.weight") not in W else conv(x, pfx+".nin_shortcut", 1)
    return s+h

def attn(x,pfx):
    B,C,H,Wd=x.shape; h=gn(x,pfx+".norm")
    q=conv(h,pfx+".q",1).reshape(B,C,H*Wd).permute(0,2,1)  # [B,HW,C]
    k=conv(h,pfx+".k",1).reshape(B,C,H*Wd)                 # [B,C,HW]
    v=conv(h,pfx+".v",1).reshape(B,C,H*Wd).permute(0,2,1)  # [B,HW,C]
    a=torch.softmax((q@k)*(C**-0.5), dim=-1)               # [B,HW,HW]
    o=(a@v).permute(0,2,1).reshape(B,C,H,Wd)
    return x+conv(o,pfx+".proj_out",1)

def upsample(x,pfx):
    x=F.interpolate(x, scale_factor=2, mode="nearest")
    return conv(x, pfx+".upsample.conv", 3)

lat=np.fromfile(r"D:\prov\2026-09-24_fresh-golden\latent_final_HTP0.f32",dtype=np.float32).reshape(1,16,8,8)
z=torch.from_numpy(lat)/SCALE + SHIFT
z.numpy().astype(np.float32).tofile(OUT/"z_scaled.f32")

stg={}
with torch.no_grad():
    h=conv(z,"decoder.conv_in",3); stg["conv_in"]=h
    h=resnet(h,"decoder.mid.block_1"); h=attn(h,"decoder.mid.attn_1"); h=resnet(h,"decoder.mid.block_2"); stg["mid"]=h
    for lvl in (3,2,1,0):
        for b in range(3): h=resnet(h,f"decoder.up.{lvl}.block.{b}")
        if lvl!=0: h=upsample(h,f"decoder.up.{lvl}")
        stg[f"up{lvl}"]=h
    h=silu(gn(h,"decoder.norm_out")); img=conv(h,"decoder.conv_out",3); stg["img"]=img
for k,v in stg.items():
    v.numpy().astype(np.float32).tofile(OUT/f"golden_{k}.f32")
    print(f"golden_{k} {tuple(v.shape)} std={float(v.std()):.4f} maxabs={float(v.abs().max()):.3f}")
im=((img[0].clamp(-1,1)+1)/2*255).round().byte().permute(1,2,0).numpy()
from PIL import Image; Image.fromarray(im).save(OUT/"image.png")
print(f"[ok] bild {im.shape} -> {OUT}\\image.png  img min/max {float(img.min()):.3f}/{float(img.max()):.3f}")
