// Strommad ggml VAE-decoder: z[16,LH,LH] -> bild[3,8*LH,8*LH].
//
// Den enda grafen i zimage_vae.cpp gar inte over 64px: ggml aterananvander inte minne over
// en lang graf, sa 512px ville ha 25,8 GB pa CPU och 1,2 GB i EN buffert pa HTP (over
// max_bufsize 1 GB). Samma sak som fallde DiT:n innan vi strommade den lager for lager.
//
// Har kors avkodaren i SJU etapper med var sin kontext och graf, och aktiveringen gar via
// varden emellan. Varje etapps graf ar da liten nog att allokeras.
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

static const float EPS = 1e-6f;
static const int   G   = 32;

template <typename T> static std::vector<T> read_bin(const std::string & p) {
    FILE * f = fopen(p.c_str(), "rb");
    if (!f) { fprintf(stderr, "no %s\n", p.c_str()); exit(1); }
    _fseeki64(f, 0, SEEK_END); long long sz = _ftelli64(f); _fseeki64(f, 0, SEEK_SET);
    std::vector<T> v((size_t) (sz / sizeof(T)));
    size_t got = fread(v.data(), 1, (size_t) sz, f); (void) got;
    fclose(f);
    return v;
}

struct VaeCtx {
    gguf_context * gctx;
    FILE *         gf;
    size_t         data_base;
    ggml_backend_t backend;
    ggml_gallocr_t galloc = nullptr;   // en allokator for hela korningen, inte en per etapp
};

// En etapp: bygger sin egen graf, laddar bara sina egna vikter, och lamnar tillbaka
// resultatet pa varden tillsammans med dess form.
static std::vector<float> run_stage(VaeCtx & vc, int stage,
                                    const std::vector<float> & in, int64_t W, int64_t H, int64_t C,
                                    int64_t * oW, int64_t * oH, int64_t * oC, bool verbose) {
    ggml_init_params cp = { ggml_tensor_overhead() * 512 + ggml_graph_overhead_custom(512, false), nullptr, true };
    ggml_context * ctx = ggml_init(cp);

    std::vector<std::pair<ggml_tensor *, std::string>> to_up;
    auto T = [&](const std::string & name) -> ggml_tensor * {
        int64_t id = gguf_find_tensor(vc.gctx, name.c_str());
        if (id < 0) { fprintf(stderr, "saknar %s\n", name.c_str()); exit(1); }
        ggml_tensor * t = ggml_new_tensor(ctx, gguf_get_tensor_type(vc.gctx, id), GGML_MAX_DIMS,
                                          gguf_get_tensor_ne(vc.gctx, id));
        ggml_set_name(t, name.c_str());
        to_up.push_back({ t, name });
        return t;
    };
    auto conv = [&](ggml_tensor * x, const std::string & pfx, int k) -> ggml_tensor * {
        ggml_tensor * w = T(pfx + ".weight");
        ggml_tensor * b = T(pfx + ".bias");
        ggml_tensor * y = ggml_conv_2d(ctx, w, x, 1, 1, k / 2, k / 2, 1, 1);
        return ggml_add(ctx, y, ggml_reshape_4d(ctx, b, 1, 1, y->ne[2], 1));
    };
    auto gn = [&](ggml_tensor * x, const std::string & pfx) -> ggml_tensor * {
        int64_t c = x->ne[2];
        ggml_tensor * y = ggml_group_norm(ctx, x, G, EPS);
        y = ggml_mul(ctx, y, ggml_reshape_4d(ctx, T(pfx + ".weight"), 1, 1, c, 1));
        return ggml_add(ctx, y, ggml_reshape_4d(ctx, T(pfx + ".bias"), 1, 1, c, 1));
    };
    auto resnet = [&](ggml_tensor * x, const std::string & pfx) -> ggml_tensor * {
        ggml_tensor * h = conv(ggml_silu(ctx, gn(x, pfx + ".norm1")), pfx + ".conv1", 3);
        h = conv(ggml_silu(ctx, gn(h, pfx + ".norm2")), pfx + ".conv2", 3);
        ggml_tensor * s = x;
        if (gguf_find_tensor(vc.gctx, (pfx + ".nin_shortcut.weight").c_str()) >= 0)
            s = conv(x, pfx + ".nin_shortcut", 1);
        return ggml_add(ctx, s, h);
    };
    auto attn = [&](ggml_tensor * x, const std::string & pfx) -> ggml_tensor * {
        int64_t w = x->ne[0], h_ = x->ne[1], c = x->ne[2], n = x->ne[3], hw = w * h_;
        ggml_tensor * h = gn(x, pfx + ".norm");
        auto to_chw = [&](ggml_tensor * t) {
            return ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, t, hw, c, n), 1, 0, 2, 3));
        };
        ggml_tensor * q = to_chw(conv(h, pfx + ".q", 1));
        ggml_tensor * k = to_chw(conv(h, pfx + ".k", 1));
        ggml_tensor * v = to_chw(conv(h, pfx + ".v", 1));
        ggml_tensor * sc = ggml_soft_max(ctx, ggml_scale(ctx, ggml_mul_mat(ctx, k, q), 1.0f / sqrtf((float) c)));
        ggml_tensor * vt = ggml_cont(ctx, ggml_permute(ctx, v, 1, 0, 2, 3));
        ggml_tensor * o  = ggml_cont(ctx, ggml_permute(ctx, ggml_mul_mat(ctx, vt, sc), 1, 0, 2, 3));
        o = conv(ggml_reshape_4d(ctx, o, w, h_, c, n), pfx + ".proj_out", 1);
        return ggml_add(ctx, x, o);
    };

    ggml_tensor * x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, W, H, C, 1);
    ggml_set_input(x);

    ggml_tensor * y = nullptr;
    const std::string lvl3 = "decoder.up.3", lvl2 = "decoder.up.2", lvl1 = "decoder.up.1", lvl0 = "decoder.up.0";
    auto level = [&](ggml_tensor * t, const std::string & p, bool up) {
        for (int b = 0; b < 3; b++) t = resnet(t, p + ".block." + std::to_string(b));
        if (up) t = conv(ggml_upscale(ctx, t, 2, GGML_SCALE_MODE_NEAREST), p + ".upsample.conv", 3);
        return t;
    };
    switch (stage) {
        case 0: y = conv(x, "decoder.conv_in", 3); break;
        case 1: y = resnet(x, "decoder.mid.block_1");
                y = attn(y, "decoder.mid.attn_1");
                y = resnet(y, "decoder.mid.block_2"); break;
        case 2: y = level(x, lvl3, true);  break;
        case 3: y = level(x, lvl2, true);  break;
        case 4: y = level(x, lvl1, true);  break;
        case 5: y = level(x, lvl0, false); break;
        case 6: y = conv(ggml_silu(ctx, gn(x, "decoder.norm_out")), "decoder.conv_out", 3); break;
        default: fprintf(stderr, "okand etapp %d\n", stage); exit(1);
    }
    ggml_set_output(y);

    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(ctx, vc.backend);
    if (!wbuf) { fprintf(stderr, "etapp %d: kunde inte allokera vikter\n", stage); exit(1); }
    ggml_backend_buffer_set_usage(wbuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    {
        std::vector<uint8_t> stage_buf;
        for (auto & pr : to_up) {
            int64_t id = gguf_find_tensor(vc.gctx, pr.second.c_str());
            size_t off = vc.data_base + gguf_get_tensor_offset(vc.gctx, id);
            size_t sz  = gguf_get_tensor_size(vc.gctx, id);
            stage_buf.resize(sz);
            _fseeki64(vc.gf, (long long) off, SEEK_SET);
            size_t got = fread(stage_buf.data(), 1, sz, vc.gf); (void) got;
            ggml_backend_tensor_set(pr.first, stage_buf.data(), 0, sz);
        }
    }
    ggml_backend_tensor_set(x, in.data(), 0, in.size() * sizeof(float));

    ggml_cgraph * g = ggml_new_graph_custom(ctx, 512, false);
    ggml_build_forward_expand(g, y);
    if (!vc.galloc) vc.galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(vc.backend));
    if (!ggml_gallocr_alloc_graph(vc.galloc, g)) { fprintf(stderr, "etapp %d: gallocr\n", stage); exit(1); }

    int64_t t0 = ggml_time_us();
    if (ggml_backend_graph_compute(vc.backend, g) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "etapp %d: compute\n", stage); exit(1);
    }
    int64_t t1 = ggml_time_us();

    std::vector<float> out((size_t) ggml_nelements(y));
    ggml_backend_tensor_get(y, out.data(), 0, out.size() * sizeof(float));
    *oW = y->ne[0]; *oH = y->ne[1]; *oC = y->ne[2];
    if (verbose) {
        printf("  etapp %d: [%lld,%lld,%lld] -> [%lld,%lld,%lld]  %.0f ms\n", stage,
               (long long) W, (long long) H, (long long) C, (long long) *oW, (long long) *oH, (long long) *oC,
               (t1 - t0) / 1000.0);
        fflush(stdout);
    }
    ggml_backend_buffer_free(wbuf);
    ggml_free(ctx);
    return out;
}

int main(int argc, char ** argv) {
    std::string gguf_path = argc > 1 ? argv[1] : "C:\\PulseCore\\models\\zImage\\vae-decoder.gguf";
    std::string dir       = argc > 2 ? argv[2] : ".";
    std::string dev_name  = argc > 3 ? argv[3] : "HTP0";
    const bool  verbose   = getenv("ZI_VAE_QUIET") == nullptr;

    ggml_backend_load_all();
    ggml_backend_dev_t dev = ggml_backend_dev_by_name(dev_name.c_str());
    if (!dev) { fprintf(stderr, "device %s hittades inte\n", dev_name.c_str()); return 1; }

    VaeCtx vc{};
    vc.backend = ggml_backend_dev_init(dev, nullptr);
    gguf_init_params gp = { true, nullptr };
    vc.gctx = gguf_init_from_file(gguf_path.c_str(), gp);
    if (!vc.gctx) { fprintf(stderr, "kan inte lasa %s\n", gguf_path.c_str()); return 1; }
    vc.gf = fopen(gguf_path.c_str(), "rb");
    vc.data_base = gguf_get_data_offset(vc.gctx);

    std::vector<float> h = read_bin<float>(dir + "\\z_scaled.f32");
    int64_t LH = (int64_t) (0.5 + sqrt((double) h.size() / 16.0));
    int64_t W = LH, H = LH, C = 16;
    if (verbose) printf("[vae] %s: latent %lldx%lld -> bild %lldpx\n",
                        ggml_backend_dev_description(dev), (long long) LH, (long long) LH, (long long) LH * 8);

    int64_t t0 = ggml_time_us();
    for (int s = 0; s <= 6; s++) {
        int64_t oW, oH, oC;
        h = run_stage(vc, s, h, W, H, C, &oW, &oH, &oC, verbose);
        W = oW; H = oH; C = oC;
    }
    int64_t t1 = ggml_time_us();

    FILE * f = fopen((dir + "\\got_img.f32").c_str(), "wb");
    fwrite(h.data(), sizeof(float), h.size(), f);
    fclose(f);
    double mx = 0; int nan = 0;
    for (float v : h) { mx = fmax(mx, fabs(v)); if (std::isnan(v)) nan++; }
    printf("[vae] klar: [%lld,%lld,%lld] maxabs=%.3f nan=%d  %.2f s -> got_img.f32\n",
           (long long) W, (long long) H, (long long) C, mx, nan, (t1 - t0) / 1e6);

    if (vc.galloc) ggml_gallocr_free(vc.galloc);
    fclose(vc.gf);
    gguf_free(vc.gctx);
    ggml_backend_free(vc.backend);
    return 0;
}
