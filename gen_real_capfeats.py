# -*- coding: utf-8 -*-
# Steg 1 av färskt golden: RIKTIG cap_feats = stock HF Qwen3-4B hidden_states[-2]
# (diffusers exakta mall), sparad som f32 [seq,2560] till D:\prov för ref_full.
# ⛔ 2026-09-26: VIKTERNA I D:\models\qwen3-4b-stock AR RADERADE (8 GB, beslutat).
#    Tokenizern + config ligger kvar (16 MB), sa promptmallen fungerar - men
#    AutoModelForCausalLM nedan FALLER tills vikterna hamtas hem igen fran HuggingFace
#    (unsloth/Qwen3-4B eller Qwen/Qwen3-4B, samma stock-modell).
#    Skalet: orakelets jobb ar gjort - llama.cpp-vagen ar MATT mot detta facit till
#    cos 0,999984, och den egenskapen andras inte per prompt. Hamta hem vikterna bara
#    om du behover ett FARSKT oberoende facit. GenieX duger INTE som ersattning:
#    GenieX ar llama.cpp, alltsa samma implementation vi validerar = cirkulart.
import os, time, numpy as np, torch
os.environ.setdefault("HF_HUB_OFFLINE","1"); os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
from transformers import AutoModelForCausalLM, AutoTokenizer
SRC = r"D:\models\qwen3-4b-stock"
OUT = r"D:\prov\2026-09-24_fresh-golden"
os.makedirs(OUT, exist_ok=True)
PROMPT = "a photo of a cat"

tok = AutoTokenizer.from_pretrained(SRC)
s = tok.apply_chat_template([{"role":"user","content":PROMPT}], tokenize=False,
                            add_generation_prompt=True, enable_thinking=True)
ids = tok(s, return_tensors="pt").input_ids
print("[tok] seq =", ids.shape[1], flush=True)

t0=time.perf_counter()
model = AutoModelForCausalLM.from_pretrained(SRC, dtype=torch.float16, low_cpu_mem_usage=True).eval()
print(f"[load] {time.perf_counter()-t0:.0f}s", flush=True)
with torch.no_grad():
    h2 = model(ids, output_hidden_states=True).hidden_states[-2][0].float().numpy()  # [seq,2560]
print(f"[penult] shape={h2.shape} std={h2.std():.3f} maxabs={np.abs(h2).max():.1f}", flush=True)
h2.astype(np.float32).tofile(os.path.join(OUT, "real_cap_feats.f32"))
np.array([h2.shape[0]], dtype=np.int32).tofile(os.path.join(OUT, "scap.i32"))
print(f"[ok] sparad -> {OUT}\\real_cap_feats.f32  (Scap={h2.shape[0]})", flush=True)
