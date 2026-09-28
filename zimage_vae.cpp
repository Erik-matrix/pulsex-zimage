// ggml VAE-decoder (ldm/flux-stil): z[16,8,8] -> bild[3,64,64]. Verifieras mot torch-golden.
#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>

static const float EPS = 1e-6f;
static const int G = 32;

template <typename T> static std::vector<T> read_bin(const std::string & p) {
    FILE * f = fopen(p.c_str(), "rb"); if (!f) { fprintf(stderr, "no %s\n", p.c_str()); exit(1); }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<T> v(sz / sizeof(T)); fread(v.data(), 1, sz, f); fclose(f); return v;
}

int main(int argc, char ** argv) {
    std::string gguf_path = argc > 1 ? argv[1] : "D:\\prov\\2026-09-24_fresh-golden\\vae\\vae-decoder.gguf";
    std::string dir       = argc > 2 ? argv[2] : "D:\\prov\\2026-09-24_fresh-golden\\vae";
    std::string dev_name  = argc > 3 ? argv[3] : "CPU";

    ggml_backend_load_all();
    ggml_backend_dev_t dev = ggml_backend_dev_by_name(dev_name.c_str());
    if (!dev) { fprintf(stderr, "device %s not found\n", dev_name.c_str()); return 1; }
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    printf("[vae] backend: %s\n", ggml_backend_dev_description(dev));

    gguf_init_params gp = { true, nullptr };
    gguf_context * gctx = gguf_init_from_file(gguf_path.c_str(), gp);
    if (!gctx) { fprintf(stderr, "failed gguf %s\n", gguf_path.c_str()); return 1; }
    FILE * gf = fopen(gguf_path.c_str(), "rb");
    size_t data_base = gguf_get_data_offset(gctx);

    ggml_init_params cparams = { ggml_tensor_overhead() * 1024 + ggml_graph_overhead_custom(2048, false), nullptr, true };
    ggml_context * ctx = ggml_init(cparams);

    // ---- weight loader: create tensor in ctx, remember for upload ----
    std::vector<std::pair<ggml_tensor *, std::string>> to_up;
    auto T = [&](const std::string & name) -> ggml_tensor * {
        int64_t id = gguf_find_tensor(gctx, name.c_str());
        if (id < 0) { fprintf(stderr, "missing %s\n", name.c_str()); exit(1); }
        ggml_tensor * t = ggml_new_tensor(ctx, gguf_get_tensor_type(gctx, id), GGML_MAX_DIMS, gguf_get_tensor_ne(gctx, id));
        ggml_set_name(t, name.c_str());
        to_up.push_back({ t, name });
        return t;
    };

    // input z [W,H,C,N] -- latent-sidan las ur z_scaled.f32 (kvadratisk, INCH=16)
    std::vector<float> zin = read_bin<float>(dir + "\\z_scaled.f32");
    int64_t LH = (int64_t) (0.5 + sqrt((double) zin.size() / 16.0));
    printf("[vae] latent %lldx%lld (INCH=16) -> bild %lldpx\n", (long long) LH, (long long) LH, (long long) LH * 8);
    ggml_tensor * z = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, LH, LH, 16, 1);
    ggml_set_input(z);

    // ---- op helpers ----
    auto conv = [&](ggml_tensor * x, const std::string & pfx, int k) -> ggml_tensor * {
        ggml_tensor * w = T(pfx + ".weight");
        ggml_tensor * b = T(pfx + ".bias");
        ggml_tensor * y = ggml_conv_2d(ctx, w, x, 1, 1, k / 2, k / 2, 1, 1); // [OW,OH,OC,N]
        int64_t OC = y->ne[2];
        return ggml_add(ctx, y, ggml_reshape_4d(ctx, b, 1, 1, OC, 1));
    };
    auto gn = [&](ggml_tensor * x, const std::string & pfx) -> ggml_tensor * {
        int64_t C = x->ne[2];
        ggml_tensor * y = ggml_group_norm(ctx, x, G, EPS);
        ggml_tensor * w = T(pfx + ".weight");
        ggml_tensor * b = T(pfx + ".bias");
        y = ggml_mul(ctx, y, ggml_reshape_4d(ctx, w, 1, 1, C, 1));
        return ggml_add(ctx, y, ggml_reshape_4d(ctx, b, 1, 1, C, 1));
    };
    auto resnet = [&](ggml_tensor * x, const std::string & pfx) -> ggml_tensor * {
        ggml_tensor * h = conv(ggml_silu(ctx, gn(x, pfx + ".norm1")), pfx + ".conv1", 3);
        h = conv(ggml_silu(ctx, gn(h, pfx + ".norm2")), pfx + ".conv2", 3);
        ggml_tensor * s = x;
        if (gguf_find_tensor(gctx, (pfx + ".nin_shortcut.weight").c_str()) >= 0)
            s = conv(x, pfx + ".nin_shortcut", 1);
        return ggml_add(ctx, s, h);
    };
    auto attn = [&](ggml_tensor * x, const std::string & pfx) -> ggml_tensor * {
        int64_t W = x->ne[0], H = x->ne[1], C = x->ne[2], N = x->ne[3], HW = W * H;
        ggml_tensor * h = gn(x, pfx + ".norm");
        auto to_ChwN = [&](ggml_tensor * t) { // [W,H,C,N] -> [C,HW,N]
            ggml_tensor * r = ggml_reshape_3d(ctx, t, HW, C, N);
            return ggml_cont(ctx, ggml_permute(ctx, r, 1, 0, 2, 3)); // [C,HW,N]
        };
        ggml_tensor * q = to_ChwN(conv(h, pfx + ".q", 1));
        ggml_tensor * k = to_ChwN(conv(h, pfx + ".k", 1));
        ggml_tensor * v = to_ChwN(conv(h, pfx + ".v", 1));
        ggml_tensor * sc = ggml_mul_mat(ctx, k, q);            // [HW,HW,N] (contract C)
        sc = ggml_scale(ctx, sc, 1.0f / sqrtf((float) C));
        sc = ggml_soft_max(ctx, sc);
        ggml_tensor * vt = ggml_cont(ctx, ggml_permute(ctx, v, 1, 0, 2, 3)); // [HW,C,N]
        ggml_tensor * o = ggml_mul_mat(ctx, vt, sc);           // [C,HW,N] (contract HW)
        o = ggml_cont(ctx, ggml_permute(ctx, o, 1, 0, 2, 3));  // [HW,C,N]
        o = ggml_reshape_4d(ctx, o, W, H, C, N);
        o = conv(o, pfx + ".proj_out", 1);
        return ggml_add(ctx, x, o);
    };
    auto upsample = [&](ggml_tensor * x, const std::string & pfx) -> ggml_tensor * {
        ggml_tensor * u = ggml_upscale(ctx, x, 2, GGML_SCALE_MODE_NEAREST);
        return conv(u, pfx + ".upsample.conv", 3);
    };

    // ---- decoder graph ----
    ggml_tensor * h = conv(z, "decoder.conv_in", 3);
    ggml_tensor * s_conv_in = h;
    h = resnet(h, "decoder.mid.block_1");
    h = attn(h, "decoder.mid.attn_1");
    h = resnet(h, "decoder.mid.block_2");
    ggml_tensor * s_mid = h;
    ggml_tensor * s_up[4] = {};
    for (int lvl = 3; lvl >= 0; lvl--) {
        for (int b = 0; b < 3; b++) h = resnet(h, "decoder.up." + std::to_string(lvl) + ".block." + std::to_string(b));
        if (lvl != 0) h = upsample(h, "decoder.up." + std::to_string(lvl));
        s_up[lvl] = h;
    }
    h = ggml_silu(ctx, gn(h, "decoder.norm_out"));
    ggml_tensor * img = conv(h, "decoder.conv_out", 3);       // [64,64,3,1]
    ggml_set_output(img);
    ggml_set_output(s_conv_in); ggml_set_output(s_mid);
    for (int i = 0; i < 4; i++) ggml_set_output(s_up[i]);

    // ---- alloc weights + upload ----
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    ggml_backend_buffer_set_usage(wbuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    for (auto & pr : to_up) {
        int64_t id = gguf_find_tensor(gctx, pr.second.c_str());
        size_t off = data_base + gguf_get_tensor_offset(gctx, id);
        size_t sz = gguf_get_tensor_size(gctx, id);
        std::vector<uint8_t> stage(sz);
        _fseeki64(gf, (long long) off, SEEK_SET);
        fread(stage.data(), 1, sz, gf);
        ggml_backend_tensor_set(pr.first, stage.data(), 0, sz);
    }
    ggml_backend_tensor_set(z, zin.data(), 0, zin.size() * sizeof(float));

    // ---- compute ----
    ggml_cgraph * g = ggml_new_graph_custom(ctx, 2048, false);
    ggml_build_forward_expand(g, img);
    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(ga, g);
    ggml_backend_graph_compute(backend, g);

    auto dump = [&](ggml_tensor * t, const std::string & fn) {
        std::vector<float> o(ggml_nelements(t));
        ggml_backend_tensor_get(t, o.data(), 0, o.size() * sizeof(float));
        FILE * f = fopen((dir + "\\got_" + fn + ".f32").c_str(), "wb");
        fwrite(o.data(), sizeof(float), o.size(), f); fclose(f);
        double mx = 0; for (float x : o) mx = fmax(mx, fabs(x));
        printf("  got_%s ne=[%lld,%lld,%lld,%lld] maxabs=%.3f nan=%d\n", fn.c_str(),
               (long long)t->ne[0],(long long)t->ne[1],(long long)t->ne[2],(long long)t->ne[3], mx, (int)std::isnan(o.empty()?0:o[0]));
    };
    dump(s_conv_in, "conv_in"); dump(s_mid, "mid");
    dump(s_up[3], "up3"); dump(s_up[2], "up2"); dump(s_up[1], "up1"); dump(s_up[0], "up0");
    dump(img, "img");
    printf("[vae] klar\n");
    return 0;
}
