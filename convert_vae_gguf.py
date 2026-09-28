# -*- coding: utf-8 -*-
# ae.safetensors (ldm-decoder) -> GGUF. Conv-kärnor F16 (ggml_conv_2d), norm/bias/attn F32.
import numpy as np, struct, json, gguf
VF  = r"C:\PulseCore\models\zImage\vae\ae.safetensors"
DST = r"D:\prov\2026-09-24_fresh-golden\vae\vae-decoder.gguf"

with open(VF,"rb") as fh:
    n=struct.unpack("<Q",fh.read(8))[0]; hdr=json.loads(fh.read(n)); base=8+n
    def load(k):
        m=hdr[k]; o0,o1=m["data_offsets"]; fh.seek(base+o0)
        dt={'F32':np.float32,'F16':np.float16}[m["dtype"]]
        return np.frombuffer(fh.read(o1-o0),dtype=dt).reshape(m["shape"]).astype(np.float32).copy()
    keys=[k for k in hdr if k!="__metadata__" and k.startswith("decoder.")]
    W={k:load(k) for k in keys}

wr=gguf.GGUFWriter(DST,"vae-decoder")
for k,v in W.items():
    # conv-kärnor [O,I,kh,kw] -> F16; övrigt F32
    if v.ndim==4 and v.shape[2]>1:  # 3x3-conv
        wr.add_tensor(k, v.astype(np.float16))
    else:
        wr.add_tensor(k, v.astype(np.float32))
wr.write_header_to_file(); wr.write_kv_data_to_file(); wr.write_tensors_to_file(); wr.close()
print(f"[ok] {len(W)} tensorer -> {DST}")
print("conv_in", W["decoder.conv_in.weight"].shape, "conv_out", W["decoder.conv_out.weight"].shape)
