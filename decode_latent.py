# -*- coding: utf-8 -*-
# Decoda en godtycklig latent [16,H,H] -> PNG via torch ldm-VAE (samma som vae_reference).
import sys, numpy as np, torch, torch.nn.functional as F, struct, json
from pathlib import Path
from PIL import Image

LATF = sys.argv[1] if len(sys.argv) > 1 else r"D:\prov\2026-09-24_fresh-golden\latent_sharp_32.f32"
OUTP = sys.argv[2] if len(sys.argv) > 2 else r"D:\prov\2026-09-24_fresh-golden\image_sharp.png"
VF   = r"C:\PulseCore\models\zImage\vae\ae.safetensors"
SCALE, SHIFT, G, EPS, INCH = 0.3611, 0.1159, 32, 1e-6, 16

with open(VF,"rb") as fh:
    n=struct.unpack("<Q",fh.read(8))[0]; hdr=json.loads(fh.read(n)); base=8+n
    W={}
    for k,m in hdr.items():
        if k=="__metadata__": continue
        o0,o1=m["data_offsets"]; fh.seek(base+o0)
        dt={'F32':np.float32,'F16':np.float16}[m["dtype"]]
        W[k]=torch.from_numpy(np.frombuffer(fh.read(o1-o0),dtype=dt).reshape(m["shape"]).astype(np.float32).copy())
def w(k): return W[k]
def conv(x,p,k=3): return F.conv2d(x,w(p+".weight"),w(p+".bias"),padding=k//2)
def gn(x,p): return F.group_norm(x,G,w(p+".weight"),w(p+".bias"),EPS)
def resnet(x,p):
    h=conv(F.silu(gn(x,p+".norm1")),p+".conv1",3); h=conv(F.silu(gn(h,p+".norm2")),p+".conv2",3)
    s=x if (p+".nin_shortcut.weight") not in W else conv(x,p+".nin_shortcut",1)
    return s+h
def attn(x,p):
    B,C,H,Wd=x.shape; h=gn(x,p+".norm")
    q=conv(h,p+".q",1).reshape(B,C,H*Wd).permute(0,2,1); k=conv(h,p+".k",1).reshape(B,C,H*Wd); v=conv(h,p+".v",1).reshape(B,C,H*Wd).permute(0,2,1)
    a=torch.softmax((q@k)*(C**-0.5),-1); o=(a@v).permute(0,2,1).reshape(B,C,H,Wd)
    return x+conv(o,p+".proj_out",1)
def up(x,p): return conv(F.interpolate(x,scale_factor=2,mode="nearest"),p+".upsample.conv",3)

lat=np.fromfile(LATF,dtype=np.float32); H=int(round((lat.size//INCH)**0.5))
z=torch.from_numpy(lat.reshape(1,INCH,H,H))/SCALE+SHIFT
print(f"[decode] latent {z.shape} -> bild {H*8}px")
with torch.no_grad():
    h=conv(z,"decoder.conv_in",3)
    h=resnet(h,"decoder.mid.block_1"); h=attn(h,"decoder.mid.attn_1"); h=resnet(h,"decoder.mid.block_2")
    for lvl in (3,2,1,0):
        for b in range(3): h=resnet(h,f"decoder.up.{lvl}.block.{b}")
        if lvl!=0: h=up(h,f"decoder.up.{lvl}")
    img=conv(F.silu(gn(h,"decoder.norm_out")),"decoder.conv_out",3)
im=((img[0].clamp(-1,1)+1)/2*255).round().byte().permute(1,2,0).numpy()
Image.fromarray(im).save(OUTP)
print(f"[ok] {im.shape} min/max {float(img.min()):.3f}/{float(img.max()):.3f} -> {OUTP}")
