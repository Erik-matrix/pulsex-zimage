// Full z-image-turbo DiT block chain as a plain ggml graph:
// cap_embedder(+RMSNorm) -> context_refiner x2 (mod=false)
// x_embedder              -> noise_refiner x2   (mod=true)
// concat -> layers x30 (mod=true)
// Reuses the single-block primitives verified in zimage_block.cpp (same
// build_block/apply_rope, now proven correct on HTP0 there) and chains them,
// to check whether correctness holds across many blocks back to back.

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
static const float EPS = 1e-5f;
static const int AXES[3] = {32, 48, 48};
static const float THETA = 256.0f;
static const int NL = 30, NR = 2, CAPD = 2560, PATCH_IN = 64;

template <typename T>
static std::vector<T> read_bin(const std::string & path) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); exit(1); }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<T> v(sz / sizeof(T));
    size_t got = fread(v.data(), 1, sz, f); (void) got;
    fclose(f);
    return v;
}

struct BlockWeights {
    ggml_tensor * an1, * an2, * fn1, * fn2, * qn, * kn, * wqkv, * wo, * w1, * w3, * w2, * ada_w, * ada_b;
};

static ggml_tensor * apply_rope(ggml_context * ctx, ggml_tensor * t, ggml_tensor * ids0, ggml_tensor * ids1, ggml_tensor * ids2) {
    ggml_tensor * ids[3] = {ids0, ids1, ids2};
    ggml_tensor * parts[3];
    int64_t off = 0;
    for (int ax = 0; ax < 3; ax++) {
        int64_t dd = AXES[ax];
        ggml_tensor * v = ggml_view_3d(ctx, t, dd, t->ne[1], t->ne[2], t->nb[1], t->nb[2], off * ggml_element_size(t));
        parts[ax] = ggml_rope_ext(ctx, v, ids[ax], nullptr, (int) dd, GGML_ROPE_TYPE_NORMAL, 0, THETA, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        off += dd;
    }
    return ggml_concat(ctx, ggml_concat(ctx, parts[0], parts[1], 0), parts[2], 0);
}

static ggml_tensor * rms_norm_w(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w) {
    return ggml_mul(ctx, ggml_rms_norm(ctx, x, EPS), w);
}

static ggml_tensor * build_block(ggml_context * ctx, ggml_tensor * x, ggml_tensor * ids0, ggml_tensor * ids1, ggml_tensor * ids2,
                                  ggml_tensor * adaln, const BlockWeights & w, bool mod) {
    const int64_t S = x->ne[1];
    ggml_tensor * smsa = nullptr, * gmsa = nullptr, * smlp = nullptr, * gmlp = nullptr;
    if (mod) {
        ggml_tensor * m = ggml_add(ctx, ggml_mul_mat(ctx, w.ada_w, adaln), w.ada_b);
        ggml_tensor * m_smsa = ggml_view_1d(ctx, m, DIM, 0 * DIM * sizeof(float));
        ggml_tensor * m_gmsa = ggml_view_1d(ctx, m, DIM, 1 * DIM * sizeof(float));
        ggml_tensor * m_smlp = ggml_view_1d(ctx, m, DIM, 2 * DIM * sizeof(float));
        ggml_tensor * m_gmlp = ggml_view_1d(ctx, m, DIM, 3 * DIM * sizeof(float));
        smsa = ggml_scale_bias(ctx, m_smsa, 1.0f, 1.0f);
        gmsa = ggml_tanh(ctx, m_gmsa);
        smlp = ggml_scale_bias(ctx, m_smlp, 1.0f, 1.0f);
        gmlp = ggml_tanh(ctx, m_gmlp);
    }

    ggml_tensor * h = rms_norm_w(ctx, x, w.an1);
    if (mod) h = ggml_mul(ctx, h, smsa);
    ggml_tensor * qkv = ggml_mul_mat(ctx, w.wqkv, h);

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
    q = apply_rope(ctx, q, ids0, ids1, ids2);
    k = apply_rope(ctx, k, ids0, ids1, ids2);

    q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
    k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
    v = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));

    const float scl = 1.0f / sqrtf((float) HD);
    ggml_tensor * scores = ggml_scale(ctx, ggml_mul_mat(ctx, k, q), scl);
    scores = ggml_soft_max(ctx, scores);
    ggml_tensor * v_t = ggml_cont(ctx, ggml_permute(ctx, v, 1, 0, 2, 3));
    ggml_tensor * attn = ggml_mul_mat(ctx, v_t, scores);
    attn = ggml_cont(ctx, ggml_permute(ctx, attn, 0, 2, 1, 3));
    attn = ggml_reshape_2d(ctx, attn, DIM, S);

    ggml_tensor * aop = ggml_mul_mat(ctx, w.wo, attn);
    ggml_tensor * aon = rms_norm_w(ctx, aop, w.an2);
    x = mod ? ggml_add(ctx, x, ggml_mul(ctx, aon, gmsa)) : ggml_add(ctx, x, aon);

    ggml_tensor * hf = rms_norm_w(ctx, x, w.fn1);
    if (mod) hf = ggml_mul(ctx, hf, smlp);
    ggml_tensor * w1o = ggml_mul_mat(ctx, w.w1, hf);
    ggml_tensor * w3o = ggml_mul_mat(ctx, w.w3, hf);
    ggml_tensor * inter = ggml_mul(ctx, ggml_silu(ctx, w1o), w3o);
    ggml_tensor * w2o = ggml_mul_mat(ctx, w.w2, inter);
    ggml_tensor * w2n = rms_norm_w(ctx, w2o, w.fn2);
    x = mod ? ggml_add(ctx, x, ggml_mul(ctx, w2n, gmlp)) : ggml_add(ctx, x, w2n);

    return x;
}

int main(int argc, char ** argv) {
    std::string gguf_path = argc > 1 ? argv[1] : "D:\\prov\\2026-09-22_zimage-gguf\\z-image-turbo.gguf";
    std::string dir = argc > 2 ? argv[2] : "D:\\prov\\2026-09-22_zimage-gguf\\full_test";
    std::string dev_name = argc > 3 ? argv[3] : "CPU";

    int Ht, Wt, Nimg, Scap, Su;
    {
        FILE * f = fopen((dir + "\\meta.txt").c_str(), "r");
        if (!f) { fprintf(stderr, "cannot open meta.txt\n"); return 1; }
        fscanf(f, "%d %d %d %d %d", &Ht, &Wt, &Nimg, &Scap, &Su);
        fclose(f);
    }
    printf("[full] Ht=%d Wt=%d Nimg=%d Scap=%d Su=%d dev=%s\n", Ht, Wt, Nimg, Scap, Su, dev_name.c_str());

    ggml_backend_load_all();
    ggml_backend_dev_t dev = ggml_backend_dev_by_name(dev_name.c_str());
    if (!dev) { fprintf(stderr, "device '%s' not found\n", dev_name.c_str()); return 1; }
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    printf("[full] backend: %s\n", ggml_backend_dev_description(dev));

    // metadata-only: no giant CPU-resident copy of the 12GB file. Tensor bytes
    // are read straight from disk into the backend buffer after allocation.
    gguf_init_params gp = { true, nullptr };
    gguf_context * gctx = gguf_init_from_file(gguf_path.c_str(), gp);
    if (!gctx) { fprintf(stderr, "failed to load %s\n", gguf_path.c_str()); return 1; }
    FILE * gguf_file = fopen(gguf_path.c_str(), "rb");
    if (!gguf_file) { fprintf(stderr, "cannot open %s\n", gguf_path.c_str()); return 1; }
    const size_t data_base = gguf_get_data_offset(gctx);

    // enough tensor slots for 34 blocks worth of weights + activations
    size_t n_tensors = 34 * 120 + 1000;
    ggml_init_params cparams = { ggml_tensor_overhead() * n_tensors + ggml_graph_overhead_custom(8192, false), nullptr, true };
    ggml_context * ctx = ggml_init(cparams);

    std::vector<ggml_tensor *> to_upload;
    std::vector<int64_t> to_upload_id;
    auto get = [&](const std::string & name) -> ggml_tensor * {
        int64_t id = gguf_find_tensor(gctx, name.c_str());
        if (id < 0) { fprintf(stderr, "missing tensor %s\n", name.c_str()); exit(1); }
        const int64_t * ne = gguf_get_tensor_ne(gctx, id);
        ggml_type type = gguf_get_tensor_type(gctx, id);
        ggml_tensor * t = ggml_new_tensor(ctx, type, GGML_MAX_DIMS, ne);
        ggml_set_name(t, name.c_str());
        to_upload.push_back(t);
        to_upload_id.push_back(id);
        return t;
    };
    auto load_block = [&](const std::string & pfx, bool mod) -> BlockWeights {
        BlockWeights w{};
        w.an1 = get(pfx + ".attention_norm1.weight");
        w.an2 = get(pfx + ".attention_norm2.weight");
        w.fn1 = get(pfx + ".ffn_norm1.weight");
        w.fn2 = get(pfx + ".ffn_norm2.weight");
        w.qn = get(pfx + ".attention.q_norm.weight");
        w.kn = get(pfx + ".attention.k_norm.weight");
        w.wqkv = get(pfx + ".attention.qkv.weight");
        w.wo = get(pfx + ".attention.out.weight");
        w.w1 = get(pfx + ".feed_forward.w1.weight");
        w.w3 = get(pfx + ".feed_forward.w3.weight");
        w.w2 = get(pfx + ".feed_forward.w2.weight");
        if (mod) {
            w.ada_w = get(pfx + ".adaLN_modulation.0.weight");
            w.ada_b = get(pfx + ".adaLN_modulation.0.bias");
        }
        return w;
    };

    ggml_tensor * cap_raw_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, CAPD, Scap);
    ggml_tensor * img_raw_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, PATCH_IN, Nimg);
    ggml_tensor * adaln_in = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 256);
    ggml_tensor * cap_ids0 = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, Scap);
    ggml_tensor * cap_ids1 = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, Scap);
    ggml_tensor * cap_ids2 = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, Scap);
    ggml_tensor * img_ids0 = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, Nimg);
    ggml_tensor * img_ids1 = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, Nimg);
    ggml_tensor * img_ids2 = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, Nimg);
    ggml_tensor * uni_ids0 = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, Su);
    ggml_tensor * uni_ids1 = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, Su);
    ggml_tensor * uni_ids2 = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, Su);
    for (auto t : {cap_raw_in, img_raw_in, adaln_in, cap_ids0, cap_ids1, cap_ids2, img_ids0, img_ids1, img_ids2, uni_ids0, uni_ids1, uni_ids2}) {
        ggml_set_input(t);
    }

    ggml_tensor * cap0_w = get("cap_embedder.0.weight");
    ggml_tensor * cap1_w = get("cap_embedder.1.weight");
    ggml_tensor * cap1_b = get("cap_embedder.1.bias");
    ggml_tensor * xemb_w = get("x_embedder.weight");
    ggml_tensor * xemb_b = get("x_embedder.bias");

    // load ALL weight tensor placeholders first, before any compute-graph node
    // exists, so the upcoming alloc_ctx_tensors() call only sees weights+inputs
    // (never touching intermediate activations -- those go through gallocr below).
    std::vector<BlockWeights> ctxref_w(NR), noiseref_w(NR), layer_w(NL);
    for (int r = 0; r < NR; r++) ctxref_w[r] = load_block("context_refiner." + std::to_string(r), false);
    for (int r = 0; r < NR; r++) noiseref_w[r] = load_block("noise_refiner." + std::to_string(r), true);
    for (int l = 0; l < NL; l++) layer_w[l] = load_block("layers." + std::to_string(l), true);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) { fprintf(stderr, "failed to allocate backend buffer\n"); return 1; }
    printf("[full] weights+inputs buffer: %.2f GB\n", ggml_backend_buffer_get_size(buf) / 1e9);
    {
        std::vector<uint8_t> stage;
        for (size_t i = 0; i < to_upload.size(); i++) {
            size_t off = data_base + gguf_get_tensor_offset(gctx, to_upload_id[i]);
            size_t sz = gguf_get_tensor_size(gctx, to_upload_id[i]);
            stage.resize(sz);
            _fseeki64(gguf_file, (long long) off, SEEK_SET);
            size_t got = fread(stage.data(), 1, sz, gguf_file);
            if (got != sz) { fprintf(stderr, "short read for %s\n", to_upload[i]->name); return 1; }
            ggml_backend_tensor_set(to_upload[i], stage.data(), 0, sz);
        }
        printf("[full] uploaded %zu weight tensors from disk\n", to_upload.size());
    }

    auto load_set = [&](ggml_tensor * t, const std::string & fname) {
        auto v = read_bin<float>(dir + "\\" + fname);
        ggml_backend_tensor_set(t, v.data(), 0, v.size() * sizeof(float));
    };
    auto load_set_i32 = [&](ggml_tensor * t, const std::string & fname) {
        auto v = read_bin<int32_t>(dir + "\\" + fname);
        ggml_backend_tensor_set(t, v.data(), 0, v.size() * sizeof(int32_t));
    };
    load_set(cap_raw_in, "cap_raw.bin");
    load_set(img_raw_in, "img_raw.bin");
    load_set(adaln_in, "adaln.bin");
    load_set_i32(cap_ids0, "cap_ids0.bin"); load_set_i32(cap_ids1, "cap_ids1.bin"); load_set_i32(cap_ids2, "cap_ids2.bin");
    load_set_i32(img_ids0, "img_ids0.bin"); load_set_i32(img_ids1, "img_ids1.bin"); load_set_i32(img_ids2, "img_ids2.bin");
    {
        auto i0 = read_bin<int32_t>(dir + "\\img_ids0.bin"); auto c0 = read_bin<int32_t>(dir + "\\cap_ids0.bin");
        auto i1 = read_bin<int32_t>(dir + "\\img_ids1.bin"); auto c1 = read_bin<int32_t>(dir + "\\cap_ids1.bin");
        auto i2 = read_bin<int32_t>(dir + "\\img_ids2.bin"); auto c2 = read_bin<int32_t>(dir + "\\cap_ids2.bin");
        std::vector<int32_t> u0 = i0, u1 = i1, u2 = i2;
        u0.insert(u0.end(), c0.begin(), c0.end());
        u1.insert(u1.end(), c1.begin(), c1.end());
        u2.insert(u2.end(), c2.begin(), c2.end());
        ggml_backend_tensor_set(uni_ids0, u0.data(), 0, u0.size() * sizeof(int32_t));
        ggml_backend_tensor_set(uni_ids1, u1.data(), 0, u1.size() * sizeof(int32_t));
        ggml_backend_tensor_set(uni_ids2, u2.data(), 0, u2.size() * sizeof(int32_t));
    }

    // ---- build the compute graph now (weights+inputs already have data; every
    // tensor created from here on is a fresh intermediate that gallocr will
    // allocate with lifetime-based reuse, not a permanent slot). ----
    ggml_tensor * cap = rms_norm_w(ctx, cap_raw_in, cap0_w);
    cap = ggml_add(ctx, ggml_mul_mat(ctx, cap1_w, cap), cap1_b);
    for (int r = 0; r < NR; r++) cap = build_block(ctx, cap, cap_ids0, cap_ids1, cap_ids2, nullptr, ctxref_w[r], false);

    ggml_tensor * x = ggml_add(ctx, ggml_mul_mat(ctx, xemb_w, img_raw_in), xemb_b);
    for (int r = 0; r < NR; r++) x = build_block(ctx, x, img_ids0, img_ids1, img_ids2, adaln_in, noiseref_w[r], true);

    ggml_tensor * cap_tap = ggml_cont(ctx, cap);
    ggml_tensor * x_tap = ggml_cont(ctx, x);
    ggml_set_output(cap_tap);
    ggml_set_output(x_tap);

    ggml_tensor * uni = ggml_concat(ctx, x, cap, 1);
    ggml_tensor * uni_tap = ggml_cont(ctx, uni);
    ggml_set_output(uni_tap);
    for (int l = 0; l < NL; l++) uni = build_block(ctx, uni, uni_ids0, uni_ids1, uni_ids2, adaln_in, layer_w[l], true);
    ggml_set_output(uni);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8192, false);
    ggml_build_forward_expand(gf, cap_tap);
    ggml_build_forward_expand(gf, x_tap);
    ggml_build_forward_expand(gf, uni_tap);
    ggml_build_forward_expand(gf, uni);

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(galloc, gf)) { fprintf(stderr, "gallocr failed to allocate graph\n"); return 1; }
    printf("[full] gallocr compute buffer: %.2f GB (reused across %d graph nodes)\n",
           ggml_gallocr_get_buffer_size(galloc, 0) / 1e9, ggml_graph_n_nodes(gf));

    ggml_status st = ggml_backend_graph_compute(backend, gf);
    if (st != GGML_STATUS_SUCCESS) { fprintf(stderr, "compute failed: %d\n", st); return 1; }

    auto tapstat = [&](ggml_tensor * t, const char * nm) {
        size_t n = ggml_nelements(t);
        std::vector<float> d(n);
        ggml_backend_tensor_get(t, d.data(), 0, n * sizeof(float));
        double mean = 0, mx = 0; int nnan = 0;
        for (float v : d) { if (v != v) { nnan++; continue; } mean += v; if (fabsf(v) > mx) mx = fabsf(v); }
        mean /= n;
        printf("[tap] %-10s n=%zu mean=%.5f max_abs=%.4f nan=%d\n", nm, n, mean, mx, nnan);
    };
    tapstat(cap_tap, "cap");
    tapstat(x_tap, "x");
    tapstat(uni_tap, "uni(pre-layers)");

    std::vector<float> out(DIM * Su);
    ggml_backend_tensor_get(uni, out.data(), 0, out.size() * sizeof(float));
    FILE * fo = fopen((dir + "\\ggml_out.bin").c_str(), "wb");
    fwrite(out.data(), sizeof(float), out.size(), fo);
    fclose(fo);

    double mean = 0, mx = 0;
    for (float v : out) { mean += v; if (fabsf(v) > mx) mx = fabsf(v); }
    mean /= out.size();
    printf("[full] wrote ggml_out.bin  mean=%.5f max_abs=%.4f\n", mean, mx);

    ggml_gallocr_free(galloc);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    fclose(gguf_file);
    gguf_free(gctx);
    ggml_backend_free(backend);
    return 0;
}
