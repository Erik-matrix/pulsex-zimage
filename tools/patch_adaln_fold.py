# ZI_ADALN_FOLD: fold the four adaLN multiplies into the RMS-norm weights.
#   rms(x)*an1*smsa  ==  rms(x)*(an1*smsa)      (and likewise an2*gmsa, fn1*smlp, fn2*gmlp)
# Four [3840,S] MULs per block (63 MB each) become four [3840] MULs (15 KB each), and the
# remaining MUL fuses into ggml-hexagon's RMS_NORM_MUL like the plain norm weight already does.
#
# Graph ORDER matters: the fusion needs RMS_NORM immediately followed by its MUL. A DFS of
# mul(rms_norm(x), mul(an1, smsa)) would put the small MUL in between and break it, so
# build_block collects the weight products in `pre`, and run_block expands them into the
# graph BEFORE x_out (already-visited nodes are skipped in the later DFS).
import io

P = r"C:\PulseCore\PulseX\zimage_dit\zimage_stream.cpp"
s = io.open(P, encoding="utf-8", newline="").read()
NL = chr(10)


def rep(old, new):
    global s
    if s.count(old) != 1:
        raise SystemExit("anchor count %d: %r" % (s.count(old), old[:80]))
    s = s.replace(old, new)


# 1) signature: optional pre-node list
rep("                                  ggml_tensor ** stage_dbg = nullptr) {" + NL,
    "                                  ggml_tensor ** stage_dbg = nullptr,"  + NL +
    "                                  std::vector<ggml_tensor *> * pre = nullptr) {" + NL +
    "    // ZI_ADALN_FOLD (forval 1): se tools/patch_adaln_fold.py for resonemanget." + NL +
    "    static const int ZI_ADALN_FOLD = []{ const char * e = getenv(\"ZI_ADALN_FOLD\"); return e ? atoi(e) : 1; }();" + NL)

# helper used below: weight product, registered so it lands in the graph before the norm
HELPER = ("    auto fold = [&](ggml_tensor * nw, ggml_tensor * mw) {" + NL +
          "        ggml_tensor * p = ggml_mul(ctx, nw, mw);   // [DIM] x [DIM]" + NL +
          "        if (pre) pre->push_back(p);" + NL +
          "        return p;" + NL +
          "    };" + NL +
          "    const bool do_fold = mod && ZI_ADALN_FOLD && pre;" + NL + NL)

# 2) attention-branch input norm
rep("    ggml_tensor * h = rms_norm_w(ctx, x, w.an1);" + NL,
    HELPER +
    "    ggml_tensor * h = do_fold ? ggml_mul(ctx, ggml_rms_norm(ctx, x, EPS), fold(w.an1, smsa))" + NL +
    "                              : rms_norm_w(ctx, x, w.an1);" + NL)
rep("    if (mod) h = ggml_mul(ctx, h, smsa);" + NL,
    "    if (mod && !do_fold) h = ggml_mul(ctx, h, smsa);" + NL)

# 3) attention output gate
rep("    ggml_tensor * aon = rms_norm_w(ctx, aop, w.an2);" + NL,
    "    ggml_tensor * aon = do_fold ? ggml_mul(ctx, ggml_rms_norm(ctx, aop, EPS), fold(w.an2, gmsa))" + NL +
    "                                : rms_norm_w(ctx, aop, w.an2);" + NL)
rep("    x = mod ? ggml_add(ctx, x, ggml_mul(ctx, aon, gmsa)) : ggml_add(ctx, x, aon);" + NL,
    "    x = (mod && !do_fold) ? ggml_add(ctx, x, ggml_mul(ctx, aon, gmsa)) : ggml_add(ctx, x, aon);" + NL)

# 4) FFN input norm
rep("    ggml_tensor * hf = rms_norm_w(ctx, x, w.fn1);" + NL,
    "    ggml_tensor * hf = do_fold ? ggml_mul(ctx, ggml_rms_norm(ctx, x, EPS), fold(w.fn1, smlp))" + NL +
    "                               : rms_norm_w(ctx, x, w.fn1);" + NL)
rep("    if (mod) hf = ggml_mul(ctx, hf, smlp);" + NL,
    "    if (mod && !do_fold) hf = ggml_mul(ctx, hf, smlp);" + NL)

# 5) FFN output gate
rep("    ggml_tensor * w2n = rms_norm_w(ctx, w2o, w.fn2);" + NL,
    "    ggml_tensor * w2n = do_fold ? ggml_mul(ctx, ggml_rms_norm(ctx, w2o, EPS), fold(w.fn2, gmlp))" + NL +
    "                                : rms_norm_w(ctx, w2o, w.fn2);" + NL)
rep("    x = mod ? ggml_add(ctx, x, ggml_mul(ctx, w2n, gmlp)) : ggml_add(ctx, x, w2n);" + NL,
    "    x = (mod && !do_fold) ? ggml_add(ctx, x, ggml_mul(ctx, w2n, gmlp)) : ggml_add(ctx, x, w2n);" + NL)

# 6) run_block: collect and expand the pre-nodes first
i = s.index("    ggml_tensor * x_out = build_block(ctx, x_g, i0, i1, i2, i_all, ff, adaln_t, w, mod,")
j = s.index(");", i) + 2
call = s[i:j]
if "pre_nodes" in call:
    raise SystemExit("already patched")
s = s[:i] + "    std::vector<ggml_tensor *> pre_nodes;" + NL + call[:-2] + ", &pre_nodes);" + s[j:]
rep("    ggml_build_forward_expand(gf, x_out);" + NL,
    "    for (auto * t : pre_nodes) ggml_build_forward_expand(gf, t);   // FORE x_out, se ZI_ADALN_FOLD" + NL +
    "    ggml_build_forward_expand(gf, x_out);" + NL)

io.open(P, "w", encoding="utf-8", newline="").write(s)
print("OK")
