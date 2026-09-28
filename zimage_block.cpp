// Single-block forward pass of the z-image-turbo DiT (AdaLN-Zero transformer
// block), built as a plain ggml graph so it can dispatch to any ggml backend
// (CPU/HTP/OpenCL). This is a correctness-first prototype: it loads one block's
// weights from our converted GGUF file, runs the graph, and is meant to be
// diffed against ref_block.py's independent numpy implementation.
//
// The 3-axis RoPE used by this model has no ggml_rope equivalent (three
// independent id axes, each with its own frequency sub-range), so the
// rotation is built from plain elementwise ops using precomputed cos/sin
// tables (computed host-side, identical to dit_full_engine.cpp's rope_fc()).

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static const int DIM = 3840, NH = 30, HD = 128, HID = 10240;
static const int HALF = 64;
static const float EPS = 1e-5f;

template <typename T>
static std::vector<T> read_bin(const std::string & path) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", path.c_str());
        exit(1);
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<T> v(sz / sizeof(T));
    size_t got = fread(v.data(), 1, sz, f);
    (void) got;
    fclose(f);
    return v;
}

struct BlockWeights {
    ggml_tensor * an1 = nullptr, * an2 = nullptr, * fn1 = nullptr, * fn2 = nullptr;
    ggml_tensor * qn = nullptr, * kn = nullptr;
    ggml_tensor * wqkv = nullptr, * wo = nullptr;
    ggml_tensor * w1 = nullptr, * w3 = nullptr, * w2 = nullptr;
    ggml_tensor * ada_w = nullptr, * ada_b = nullptr;
};

// deinterleave [HD,NH,S] into even/odd channel-pairs along dim0, i.e.
// x0[i] = t[2*i], x1[i] = t[2*i+1] for i in [0,HALF) -- matches the host's
// interleaved (2i,2i+1) RoPE pairing.
static const int AXES[3] = {32, 48, 48};
static const float THETA = 256.0f;

// 3-axis RoPE via three native ggml_rope_ext calls, one per axis, on disjoint
// contiguous channel ranges of the head dim (interleaved/GPT-J pairing, which
// is exactly GGML_ROPE_TYPE_NORMAL). Each axis restarts its own frequency
// series (theta^(-2j/dd)), matching dit_full_engine.cpp's rope_fc(). This
// replaces an earlier hand-rolled deinterleave-via-strided-view + concat
// trick that computed correctly on CPU but gave wrong results on the
// Hexagon HTP backend (confirmed by tapping the pre/post-rope tensors).
static ggml_tensor * apply_rope(ggml_context * ctx, ggml_tensor * t, ggml_tensor * ids0, ggml_tensor * ids1, ggml_tensor * ids2) {
    ggml_tensor * ids[3] = {ids0, ids1, ids2};
    ggml_tensor * parts[3];
    int64_t off = 0;
    for (int ax = 0; ax < 3; ax++) {
        int64_t dd = AXES[ax];
        ggml_tensor * v = ggml_view_3d(ctx, t, dd, t->ne[1], t->ne[2], t->nb[1], t->nb[2], off * ggml_element_size(t));
        parts[ax] = ggml_rope_ext(ctx, v, ids[ax], nullptr, (int) dd, GGML_ROPE_TYPE_NORMAL, 0,
                                   THETA, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        off += dd;
    }
    return ggml_concat(ctx, ggml_concat(ctx, parts[0], parts[1], 0), parts[2], 0);
}

static ggml_tensor * rms_norm_w(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w) {
    return ggml_mul(ctx, ggml_rms_norm(ctx, x, EPS), w);
}

// one AdaLN-Zero DiT block. x: [DIM,S]. ids0/1/2: [S] I32, per-axis RoPE positions. adaln: [256] or null (mod=false).
// dbg, if non-null, is filled with taps: [0]=q after rope [1]=scores [2]=attn (pre-Wo) [3]=x after attn-half
// dbg[4..6] = w1o, w3o, inter (SwiGLU stage, FFN half)
static ggml_tensor * build_block(ggml_context * ctx, ggml_tensor * x, ggml_tensor * ids0, ggml_tensor * ids1, ggml_tensor * ids2,
                                  ggml_tensor * adaln, const BlockWeights & w, bool mod, ggml_tensor ** dbg = nullptr) {
    const int64_t S = x->ne[1];

    ggml_tensor * smsa = nullptr, * gmsa = nullptr, * smlp = nullptr, * gmlp = nullptr;
    if (mod) {
        ggml_tensor * m = ggml_mul_mat(ctx, w.ada_w, adaln);      // [4*DIM]
        m = ggml_add(ctx, m, w.ada_b);
        ggml_tensor * m_smsa = ggml_view_1d(ctx, m, DIM, 0 * DIM * sizeof(float));
        ggml_tensor * m_gmsa = ggml_view_1d(ctx, m, DIM, 1 * DIM * sizeof(float));
        ggml_tensor * m_smlp = ggml_view_1d(ctx, m, DIM, 2 * DIM * sizeof(float));
        ggml_tensor * m_gmlp = ggml_view_1d(ctx, m, DIM, 3 * DIM * sizeof(float));
        smsa = ggml_scale_bias(ctx, m_smsa, 1.0f, 1.0f);
        gmsa = ggml_tanh(ctx, m_gmsa);
        smlp = ggml_scale_bias(ctx, m_smlp, 1.0f, 1.0f);
        gmlp = ggml_tanh(ctx, m_gmlp);
    }

    // attention half
    ggml_tensor * h = rms_norm_w(ctx, x, w.an1);
    if (mod) h = ggml_mul(ctx, h, smsa);
    ggml_tensor * qkv = ggml_mul_mat(ctx, w.wqkv, h); // [3*DIM, S]

    ggml_tensor * q = ggml_cont(ctx, ggml_view_2d(ctx, qkv, DIM, S, qkv->nb[1], 0 * DIM * sizeof(float)));
    ggml_tensor * k = ggml_cont(ctx, ggml_view_2d(ctx, qkv, DIM, S, qkv->nb[1], 1 * DIM * sizeof(float)));
    ggml_tensor * v = ggml_cont(ctx, ggml_view_2d(ctx, qkv, DIM, S, qkv->nb[1], 2 * DIM * sizeof(float)));
    q = ggml_reshape_3d(ctx, q, HD, NH, S);
    k = ggml_reshape_3d(ctx, k, HD, NH, S);
    v = ggml_reshape_3d(ctx, v, HD, NH, S);

    ggml_tensor * qn3 = ggml_reshape_3d(ctx, w.qn, HD, 1, 1);
    ggml_tensor * kn3 = ggml_reshape_3d(ctx, w.kn, HD, 1, 1);
    q = ggml_mul(ctx, ggml_rms_norm(ctx, q, EPS), qn3);
    k = ggml_mul(ctx, ggml_rms_norm(ctx, k, EPS), kn3);
    if (dbg) dbg[8] = ggml_cont(ctx, q);
    q = apply_rope(ctx, q, ids0, ids1, ids2);
    k = apply_rope(ctx, k, ids0, ids1, ids2);
    if (dbg) dbg[0] = ggml_cont(ctx, q);

    // [HD,NH,S] -> [HD,S,NH] for per-head batched matmul
    q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
    k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
    v = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));

    const float scl = 1.0f / sqrtf((float) HD);
    ggml_tensor * scores = ggml_mul_mat(ctx, k, q);           // [S(key),S(query),NH]
    scores = ggml_scale(ctx, scores, scl);
    scores = ggml_soft_max(ctx, scores);
    if (dbg) dbg[1] = ggml_cont(ctx, scores);
    ggml_tensor * v_t = ggml_cont(ctx, ggml_permute(ctx, v, 1, 0, 2, 3)); // [S,HD,NH]
    ggml_tensor * attn = ggml_mul_mat(ctx, v_t, scores);       // [HD,S(query),NH]
    attn = ggml_cont(ctx, ggml_permute(ctx, attn, 0, 2, 1, 3)); // [HD,NH,S]
    attn = ggml_reshape_2d(ctx, attn, DIM, S);
    if (dbg) dbg[2] = ggml_cont(ctx, attn);

    ggml_tensor * aop = ggml_mul_mat(ctx, w.wo, attn);
    ggml_tensor * aon = rms_norm_w(ctx, aop, w.an2);
    x = mod ? ggml_add(ctx, x, ggml_mul(ctx, aon, gmsa)) : ggml_add(ctx, x, aon);
    if (dbg) dbg[3] = ggml_cont(ctx, x);

    // ffn half
    ggml_tensor * hf = rms_norm_w(ctx, x, w.fn1);
    if (mod) hf = ggml_mul(ctx, hf, smlp);
    ggml_tensor * w1o = ggml_mul_mat(ctx, w.w1, hf);
    ggml_tensor * w3o = ggml_mul_mat(ctx, w.w3, hf);
    if (dbg) dbg[4] = ggml_cont(ctx, w1o);
    if (dbg) dbg[5] = ggml_cont(ctx, w3o);
    ggml_tensor * inter = ggml_mul(ctx, ggml_silu(ctx, w1o), w3o);
    if (dbg) dbg[6] = ggml_cont(ctx, inter);
    ggml_tensor * w2o = ggml_mul_mat(ctx, w.w2, inter);
    if (dbg) dbg[7] = ggml_cont(ctx, w2o);
    ggml_tensor * w2n = rms_norm_w(ctx, w2o, w.fn2);
    x = mod ? ggml_add(ctx, x, ggml_mul(ctx, w2n, gmlp)) : ggml_add(ctx, x, w2n);

    return x;
}

int main(int argc, char ** argv) {
    std::string gguf_path = argc > 1 ? argv[1] : "D:\\prov\\2026-09-22_zimage-gguf\\z-image-turbo.gguf";
    std::string block_dir = argc > 2 ? argv[2] : "D:\\prov\\2026-09-22_zimage-gguf\\block_test";
    std::string dev_name  = argc > 3 ? argv[3] : "CPU";

    // meta.txt: pfx \n S \n mod
    std::string pfx; int64_t S; int mod_i;
    {
        FILE * f = fopen((block_dir + "\\meta.txt").c_str(), "r");
        if (!f) { fprintf(stderr, "cannot open meta.txt\n"); return 1; }
        char buf[256];
        fgets(buf, sizeof(buf), f); buf[strcspn(buf, "\n")] = 0; pfx = buf;
        fgets(buf, sizeof(buf), f); S = atoll(buf);
        fgets(buf, sizeof(buf), f); mod_i = atoi(buf);
        fclose(f);
    }
    bool mod = mod_i != 0;
    printf("[block] pfx=%s S=%lld mod=%d dev=%s\n", pfx.c_str(), (long long) S, mod, dev_name.c_str());

    ggml_backend_load_all();
    ggml_backend_dev_t dev = ggml_backend_dev_by_name(dev_name.c_str());
    if (!dev) { fprintf(stderr, "device '%s' not found\n", dev_name.c_str()); return 1; }
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    if (!backend) { fprintf(stderr, "failed to init backend '%s'\n", dev_name.c_str()); return 1; }
    printf("[block] backend: %s\n", ggml_backend_dev_description(dev));

    // load the whole gguf into a CPU-backed context with data, then copy the
    // block's tensors + shapes into our own graph context sized to hold them.
    ggml_context * ctx_data = nullptr;
    gguf_init_params gp = { /*.no_alloc =*/ false, /*.ctx =*/ &ctx_data };
    gguf_context * gctx = gguf_init_from_file(gguf_path.c_str(), gp);
    if (!gctx) { fprintf(stderr, "failed to load %s\n", gguf_path.c_str()); return 1; }

    auto get = [&](const std::string & name) -> ggml_tensor * {
        ggml_tensor * t = ggml_get_tensor(ctx_data, name.c_str());
        if (!t) { fprintf(stderr, "missing tensor %s\n", name.c_str()); exit(1); }
        return t;
    };

    BlockWeights w;
    w.an1 = get(pfx + ".attention_norm1.weight");
    w.an2 = get(pfx + ".attention_norm2.weight");
    w.fn1 = get(pfx + ".ffn_norm1.weight");
    w.fn2 = get(pfx + ".ffn_norm2.weight");
    w.qn  = get(pfx + ".attention.q_norm.weight");
    w.kn  = get(pfx + ".attention.k_norm.weight");
    w.wqkv = get(pfx + ".attention.qkv.weight");
    w.wo   = get(pfx + ".attention.out.weight");
    w.w1 = get(pfx + ".feed_forward.w1.weight");
    w.w3 = get(pfx + ".feed_forward.w3.weight");
    w.w2 = get(pfx + ".feed_forward.w2.weight");
    if (mod) {
        w.ada_w = get(pfx + ".adaLN_modulation.0.weight");
        w.ada_b = get(pfx + ".adaLN_modulation.0.bias");
    }

    // graph context: no_alloc, tensors allocated on the target backend below.
    size_t n_tensors = 64;
    ggml_init_params cparams = { ggml_tensor_overhead() * n_tensors + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(cparams);

    ggml_tensor * x_in    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, DIM, S);
    ggml_tensor * ids0_in = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, S);
    ggml_tensor * ids1_in = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, S);
    ggml_tensor * ids2_in = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, S);
    ggml_tensor * adaln_in = mod ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 256) : nullptr;
    ggml_set_name(x_in, "x_in"); ggml_set_input(x_in);
    ggml_set_name(ids0_in, "ids0_in"); ggml_set_input(ids0_in);
    ggml_set_name(ids1_in, "ids1_in"); ggml_set_input(ids1_in);
    ggml_set_name(ids2_in, "ids2_in"); ggml_set_input(ids2_in);
    if (mod) { ggml_set_name(adaln_in, "adaln_in"); ggml_set_input(adaln_in); }

    // re-declare the weight tensors inside ctx (no_alloc) so they participate
    // in this graph's own backend buffer, then copy the values in after alloc.
    auto redecl = [&](ggml_tensor * src) -> ggml_tensor * {
        ggml_tensor * t = ggml_dup_tensor(ctx, src);
        ggml_set_name(t, src->name);
        return t;
    };
    BlockWeights wg;
    wg.an1 = redecl(w.an1); wg.an2 = redecl(w.an2);
    wg.fn1 = redecl(w.fn1); wg.fn2 = redecl(w.fn2);
    wg.qn = redecl(w.qn); wg.kn = redecl(w.kn);
    wg.wqkv = redecl(w.wqkv); wg.wo = redecl(w.wo);
    wg.w1 = redecl(w.w1); wg.w3 = redecl(w.w3); wg.w2 = redecl(w.w2);
    if (mod) { wg.ada_w = redecl(w.ada_w); wg.ada_b = redecl(w.ada_b); }

    ggml_tensor * dbg[9] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    ggml_tensor * x_out = build_block(ctx, x_in, ids0_in, ids1_in, ids2_in, adaln_in, wg, mod, dbg);
    ggml_set_output(x_out);
    for (auto * t : dbg) if (t) ggml_set_output(t);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) { fprintf(stderr, "failed to allocate backend buffer\n"); return 1; }

    auto upload_from = [&](ggml_tensor * dst, ggml_tensor * src) {
        ggml_backend_tensor_set(dst, src->data, 0, ggml_nbytes(src));
    };
    upload_from(wg.an1, w.an1); upload_from(wg.an2, w.an2);
    upload_from(wg.fn1, w.fn1); upload_from(wg.fn2, w.fn2);
    upload_from(wg.qn, w.qn); upload_from(wg.kn, w.kn);
    upload_from(wg.wqkv, w.wqkv); upload_from(wg.wo, w.wo);
    upload_from(wg.w1, w.w1); upload_from(wg.w3, w.w3); upload_from(wg.w2, w.w2);
    if (mod) { upload_from(wg.ada_w, w.ada_w); upload_from(wg.ada_b, w.ada_b); }

    std::vector<float> x0 = read_bin<float>(block_dir + "\\x0.bin");
    std::vector<int32_t> ids0 = read_bin<int32_t>(block_dir + "\\ids0.bin");
    std::vector<int32_t> ids1 = read_bin<int32_t>(block_dir + "\\ids1.bin");
    std::vector<int32_t> ids2 = read_bin<int32_t>(block_dir + "\\ids2.bin");
    ggml_backend_tensor_set(x_in, x0.data(), 0, x0.size() * sizeof(float));
    ggml_backend_tensor_set(ids0_in, ids0.data(), 0, ids0.size() * sizeof(int32_t));
    ggml_backend_tensor_set(ids1_in, ids1.data(), 0, ids1.size() * sizeof(int32_t));
    ggml_backend_tensor_set(ids2_in, ids2.data(), 0, ids2.size() * sizeof(int32_t));
    if (mod) {
        std::vector<float> adaln = read_bin<float>(block_dir + "\\adaln.bin");
        ggml_backend_tensor_set(adaln_in, adaln.data(), 0, adaln.size() * sizeof(float));
    }

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 512, false);
    for (auto * t : dbg) if (t) ggml_build_forward_expand(gf, t);
    ggml_build_forward_expand(gf, x_out);

    ggml_status st = ggml_backend_graph_compute(backend, gf);
    if (st != GGML_STATUS_SUCCESS) { fprintf(stderr, "compute failed: %d\n", st); return 1; }

    std::vector<float> out(DIM * S);
    ggml_backend_tensor_get(x_out, out.data(), 0, out.size() * sizeof(float));

    FILE * fo = fopen((block_dir + "\\ggml_out.bin").c_str(), "wb");
    fwrite(out.data(), sizeof(float), out.size(), fo);
    fclose(fo);

    double mx = 0, mean = 0;
    for (float v : out) { mean += v; if (fabsf(v) > mx) mx = fabsf(v); }
    mean /= out.size();
    printf("[block] wrote ggml_out.bin  mean=%.6f max_abs=%.6f\n", mean, mx);

    const char * dbg_names[9] = {"dbg_q_rope", "dbg_scores", "dbg_attn", "dbg_x_after_attn",
                                  "dbg_w1o", "dbg_w3o", "dbg_inter", "dbg_w2o", "dbg_q_prerope"};
    for (int i = 0; i < 9; i++) {
        if (!dbg[i]) continue;
        size_t n = ggml_nelements(dbg[i]);
        std::vector<float> d(n);
        ggml_backend_tensor_get(dbg[i], d.data(), 0, n * sizeof(float));
        FILE * f = fopen((block_dir + "\\" + dbg_names[i] + ".bin").c_str(), "wb");
        fwrite(d.data(), sizeof(float), n, f);
        fclose(f);
        int nnan = 0, ninf = 0; float mx = 0;
        for (float v : d) { if (v != v) nnan++; else if (!std::isfinite(v)) ninf++; else if (fabsf(v) > mx) mx = fabsf(v); }
        printf("[block] %s: n=%zu nan=%d inf=%d max_finite=%.4g\n", dbg_names[i], n, nnan, ninf, mx);
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_free(ctx_data);
    gguf_free(gctx);
    ggml_backend_free(backend);
    return 0;
}
