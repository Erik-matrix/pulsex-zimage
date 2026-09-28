// zimage_encode_stream.h - the Qwen3-4B text encoder with its layers STREAMED through two NPU slots.
//
// 2026-09-28 ("Stromma aven encoderns lager genom en ring"). The llama.cpp path put all 36
// layers on the NPU at once: 1.95 GB of pinned, committed memory for the ~2 s the encoder lives.
// Here llama.cpp only tokenizes (vocab_only); the layers run one at a time in our own ggml graphs,
// the same way zimage-dit-stream runs the DiT:
//   * weights: <model>.hexpack - every layer tensor stored once in ggml-hexagon's tiled layout
//     (built on first use, keyed on the GGUF's size + mtime + pack tag), copied into a slot with
//     ggml_backend_hexagon_set_tensor_packed. Two slots of one layer each (~54 MB).
//   * only layers 0..layer-1 run: cap_feats is the INPUT of `layer` (35 = hidden_states[-2]), so
//     layer 35 and the output head are never needed.
//   * the graph mirrors llama.cpp's qwen3 graph op for op: RMS norm * w, separate q/k/v, q/k RMS
//     norm, NEOX rope (base 1e6), flash attention on f16 K/V with a causal f16 mask and f32
//     accumulation, swiglu_split, residuals. Token embeddings (q6_K) are dequantized on the host.
//   * ggml_backend_sched with HTP0 + CPU: an op the NPU cannot take falls back instead of failing.

#pragma once

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "llama.h"

#include <sys/stat.h>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace zienc {

struct PackHdr { char magic[8]; char tag[32]; uint64_t gsize, gmtime; uint32_t n, pad; };
struct PackEnt { char name[96]; uint32_t type, pad; int64_t ne[4]; uint64_t off, size; };
static_assert(sizeof(PackHdr) == 64 && sizeof(PackEnt) == 152, "packcache layout");

static double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct Layer {           // one layer's tensors (in a slot's context)
    ggml_tensor *an, *wq, *wk, *wv, *qn, *kn, *wo, *fn, *wg, *wu, *wd;
};

static const char * LAYER_TENSORS[11] = {
    "attn_norm", "attn_q", "attn_k", "attn_v", "attn_q_norm", "attn_k_norm",
    "attn_output", "ffn_norm", "ffn_gate", "ffn_up", "ffn_down",
};

struct Engine {
    gguf_context *   g   = nullptr;
    FILE *           gf  = nullptr;     // the GGUF (token embeddings; raw weights while building the pack)
    size_t           data_base = 0;
    FILE *           pk  = nullptr;     // the .hexpack
    ggml_backend_t   htp = nullptr, cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;
    bool (*get_packed)(const ggml_tensor *, void *, size_t) = nullptr;
    bool (*set_packed)(ggml_tensor *, const void *, size_t) = nullptr;
    std::vector<PackEnt> ents;          // [layer*11 + k]
    ggml_backend_buffer_t slot[2] = {};
    size_t slot_size = 0;
    int n_layer_run = 35;
    int64_t n_embd = 2560, n_head = 32, n_head_kv = 8, head_dim = 128;
    float eps = 1e-6f, rope_base = 1e6f;
    int n_ctx_orig = 40960;
    double t_copy = 0, t_read = 0, t_comp = 0;

    std::string tname(int il, int k) const { return "blk." + std::to_string(il) + "." + LAYER_TENSORS[k] + ".weight"; }

    bool open(const std::string & path, const std::string & device, int layer) {
        n_layer_run = layer;
        gguf_init_params gp = { true, nullptr };
        g = gguf_init_from_file(path.c_str(), gp);
        if (!g) return false;
        gf = fopen(path.c_str(), "rb");
        data_base = gguf_get_data_offset(g);
        auto u32 = [&](const char * k, int64_t & dst) { int64_t id = gguf_find_key(g, k); if (id >= 0) dst = gguf_get_val_u32(g, id); };
        auto f32 = [&](const char * k, float & dst) { int64_t id = gguf_find_key(g, k); if (id >= 0) dst = gguf_get_val_f32(g, id); };
        u32("qwen3.embedding_length", n_embd);
        u32("qwen3.attention.head_count", n_head);
        u32("qwen3.attention.head_count_kv", n_head_kv);
        u32("qwen3.attention.key_length", head_dim);
        f32("qwen3.attention.layer_norm_rms_epsilon", eps);
        f32("qwen3.rope.freq_base", rope_base);
        { int64_t c = n_ctx_orig; u32("qwen3.context_length", c); n_ctx_orig = (int) c; }

        ggml_backend_dev_t dev = ggml_backend_dev_by_name(device.c_str());
        if (!dev) return false;
        htp = ggml_backend_dev_init(dev, nullptr);
        cpu = ggml_backend_dev_init(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU), nullptr);
        if (!htp || !cpu) return false;
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        get_packed = (bool (*)(const ggml_tensor *, void *, size_t)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_hexagon_get_tensor_packed");
        set_packed = (bool (*)(ggml_tensor *, const void *, size_t)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_hexagon_set_tensor_packed");
        auto tag_fn = (const char * (*)(void)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_hexagon_pack_tag");
        if (!get_packed || !set_packed || !tag_fn) { fprintf(stderr, "[enc] backend has no packed-weight API\n"); return false; }

        // index: layers 0..layer-1, 11 tensors each, with their packed size
        ggml_backend_buffer_type_t bt = ggml_backend_get_default_buffer_type(htp);
        const size_t al = ggml_backend_buft_get_alignment(bt);
        size_t layer_bytes = 0;
        for (int il = 0; il < n_layer_run; il++) {
            size_t lb = 0;
            for (int k = 0; k < 11; k++) {
                const std::string nm = tname(il, k);
                int64_t id = gguf_find_tensor(g, nm.c_str());
                if (id < 0) { fprintf(stderr, "[enc] missing %s\n", nm.c_str()); return false; }
                ggml_init_params p = { ggml_tensor_overhead() * 2, nullptr, true };
                ggml_context * c = ggml_init(p);
                ggml_tensor * t = ggml_new_tensor(c, gguf_get_tensor_type(g, id), GGML_MAX_DIMS, gguf_get_tensor_ne(g, id));
                PackEnt e{};
                strncpy(e.name, nm.c_str(), sizeof(e.name) - 1);
                e.type = (uint32_t) t->type;
                for (int d = 0; d < 4; d++) e.ne[d] = t->ne[d];
                e.size = ggml_backend_buft_get_alloc_size(bt, t);
                lb += GGML_PAD(e.size, al);
                ggml_free(c);
                ents.push_back(e);
            }
            layer_bytes = lb > layer_bytes ? lb : layer_bytes;
        }
        slot_size = layer_bytes;
        const uint64_t data0 = GGML_PAD(sizeof(PackHdr) + ents.size() * sizeof(PackEnt), 4096);
        { uint64_t off = data0; for (auto & e : ents) { e.off = off; off += GGML_PAD(e.size, 4096); } }

        struct _stat64 st;
        if (_stat64(path.c_str(), &st) != 0) return false;
        PackHdr want{};
        memcpy(want.magic, "PXRP0001", 8);
        strncpy(want.tag, tag_fn(), sizeof(want.tag) - 1);
        want.gsize = (uint64_t) st.st_size; want.gmtime = (uint64_t) st.st_mtime;
        const std::string pack_path = path.substr(0, path.rfind('.')) + ".hexpack";

        bool ok = false;
        if (FILE * f = fopen(pack_path.c_str(), "rb")) {
            PackHdr h{};
            std::vector<PackEnt> got(ents.size());
            ok = fread(&h, sizeof h, 1, f) == 1 && !memcmp(&h, &want, offsetof(PackHdr, n)) && h.n == ents.size() &&
                 fread(got.data(), sizeof(PackEnt), got.size(), f) == got.size() &&
                 !memcmp(got.data(), ents.data(), got.size() * sizeof(PackEnt));
            fclose(f);
        }
        if (!ok && !build_pack(pack_path, want, bt)) return false;
        pk = fopen(pack_path.c_str(), "rb");
        if (!pk) return false;

        for (int s = 0; s < 2; s++) {
            slot[s] = ggml_backend_buft_alloc_buffer(bt, slot_size);
            if (!slot[s]) return false;
            ggml_backend_buffer_set_usage(slot[s], GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        }
        ggml_backend_t bk[2] = { htp, cpu };
        sched = ggml_backend_sched_new(bk, nullptr, 2, 2048, false, true);
        return sched != nullptr;
    }

    // Every layer tensor packed ONCE in a reused WEIGHTS buffer and written as it came out. Written to
    // .tmp and renamed only when complete: an interrupted build never looks like a valid cache.
    bool build_pack(const std::string & path, PackHdr h, ggml_backend_buffer_type_t bt) {
        const double t0 = now_s();
        const std::string tmp = path + ".tmp";
        FILE * f = fopen(tmp.c_str(), "wb");
        if (!f) return false;
        uint64_t maxsz = 0;
        for (auto & e : ents) maxsz = e.size > maxsz ? e.size : maxsz;
        ggml_backend_buffer_t b = ggml_backend_buft_alloc_buffer(bt, (size_t) maxsz);
        if (!b) { fclose(f); remove(tmp.c_str()); return false; }
        ggml_backend_buffer_set_usage(b, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        h.n = (uint32_t) ents.size();
        fwrite(&h, sizeof h, 1, f);
        fwrite(ents.data(), sizeof(PackEnt), ents.size(), f);
        std::vector<uint8_t> raw, packed;
        bool good = true;
        for (size_t i = 0; i < ents.size() && good; i++) {
            ggml_init_params p = { ggml_tensor_overhead() * 2, nullptr, true };
            ggml_context * c = ggml_init(p);
            ggml_tensor * t = ggml_new_tensor(c, (ggml_type) ents[i].type, GGML_MAX_DIMS, ents[i].ne);
            ggml_tallocr ta = ggml_tallocr_new(b);
            good = ggml_tallocr_alloc(&ta, t) == GGML_STATUS_SUCCESS;
            const int64_t id = gguf_find_tensor(g, ents[i].name);
            const size_t rsz = gguf_get_tensor_size(g, id);
            raw.resize(rsz);
            _fseeki64(gf, (long long) (data_base + gguf_get_tensor_offset(g, id)), SEEK_SET);
            good = good && fread(raw.data(), 1, rsz, gf) == rsz;
            if (good) ggml_backend_tensor_set(t, raw.data(), 0, rsz);   // the tiled repack
            packed.assign((size_t) GGML_PAD(ents[i].size, 4096), 0);
            good = good && get_packed(t, packed.data(), (size_t) ents[i].size);
            _fseeki64(f, (long long) ents[i].off, SEEK_SET);
            good = good && fwrite(packed.data(), 1, packed.size(), f) == packed.size();
            ggml_free(c);
        }
        ggml_backend_buffer_free(b);
        good = fclose(f) == 0 && good;
        if (good) remove(path.c_str());
        if (!good || rename(tmp.c_str(), path.c_str()) != 0) { remove(tmp.c_str()); return false; }
        fprintf(stderr, "[enc] pack built: %zu tensors, %.1f s -> %s\n", ents.size(), now_s() - t0, path.c_str());
        return true;
    }

    // Layer il into slot s: tensors placed linearly, packed bytes copied in.
    ggml_context * load_layer(int il, int s, Layer & L, std::vector<uint8_t> & stage) {
        ggml_init_params p = { ggml_tensor_overhead() * 16, nullptr, true };
        ggml_context * c = ggml_init(p);
        ggml_tallocr ta = ggml_tallocr_new(slot[s]);
        ggml_tensor ** dst[11] = { &L.an, &L.wq, &L.wk, &L.wv, &L.qn, &L.kn, &L.wo, &L.fn, &L.wg, &L.wu, &L.wd };
        for (int k = 0; k < 11; k++) {
            const PackEnt & e = ents[(size_t) il * 11 + k];
            ggml_tensor * t = ggml_new_tensor(c, (ggml_type) e.type, GGML_MAX_DIMS, e.ne);
            ggml_set_name(t, e.name);
            if (ggml_tallocr_alloc(&ta, t) != GGML_STATUS_SUCCESS) { ggml_free(c); return nullptr; }
            const double t0 = now_s();
            stage.resize((size_t) e.size);
            _fseeki64(pk, (long long) e.off, SEEK_SET);
            if (fread(stage.data(), 1, (size_t) e.size, pk) != e.size) { ggml_free(c); return nullptr; }
            const double t1 = now_s();
            if (!set_packed(t, stage.data(), (size_t) e.size)) { ggml_free(c); return nullptr; }
            t_read += t1 - t0; t_copy += now_s() - t1;
            *dst[k] = t;
        }
        return c;
    }

    // tokens -> input of layer n_layer_run, [n x n_embd] f32
    bool encode(const std::vector<llama_token> & tok, std::vector<float> & out) {
        const int64_t n = (int64_t) tok.size();
        // token embeddings: dequantize the rows on the host (== llama's get_rows on the CPU)
        std::vector<float> x((size_t) n * n_embd);
        {
            const int64_t id = gguf_find_tensor(g, "token_embd.weight");
            const ggml_type ty = gguf_get_tensor_type(g, id);
            const size_t rb = ggml_row_size(ty, n_embd);
            const auto * tr = ggml_get_type_traits(ty);
            std::vector<uint8_t> row(rb);
            for (int64_t i = 0; i < n; i++) {
                _fseeki64(gf, (long long) (data_base + gguf_get_tensor_offset(g, id) + (size_t) tok[i] * rb), SEEK_SET);
                if (fread(row.data(), 1, rb, gf) != rb) return false;
                if (ty == GGML_TYPE_F32) memcpy(&x[(size_t) i * n_embd], row.data(), rb);
                else tr->to_float(row.data(), &x[(size_t) i * n_embd], n_embd);
            }
        }
        // causal mask [n_kv = n, n] f16 and positions 0..n-1 - the same for every layer
        std::vector<ggml_fp16_t> mask((size_t) n * n);
        for (int64_t i = 0; i < n; i++)
            for (int64_t j = 0; j < n; j++) mask[(size_t) i * n + j] = ggml_fp32_to_fp16(j <= i ? 0.0f : -INFINITY);
        std::vector<int32_t> pos((size_t) n);
        for (int64_t i = 0; i < n; i++) pos[i] = (int32_t) i;

        std::vector<uint8_t> stage;
        std::vector<float> y((size_t) n * n_embd);
        ggml_context * lctx[2] = {};
        for (int il = 0; il < n_layer_run; il++) {
            const int s = il & 1;
            if (lctx[s]) { ggml_free(lctx[s]); lctx[s] = nullptr; }
            Layer L{};
            lctx[s] = load_layer(il, s, L, stage);
            if (!lctx[s]) { fprintf(stderr, "[enc] layer %d failed to load\n", il); return false; }

            const double tc = now_s();
            ggml_init_params p = { ggml_tensor_overhead() * 128 + ggml_graph_overhead_custom(256, false), nullptr, true };
            ggml_context * c = ggml_init(p);
            ggml_tensor * xin = ggml_new_tensor_2d(c, GGML_TYPE_F32, n_embd, n);
            ggml_tensor * tpos = ggml_new_tensor_1d(c, GGML_TYPE_I32, n);
            ggml_tensor * tmask = ggml_new_tensor_2d(c, GGML_TYPE_F16, n, n);
            ggml_set_input(xin); ggml_set_input(tpos); ggml_set_input(tmask);

            auto norm = [&](ggml_tensor * t, ggml_tensor * w) { return ggml_mul(c, ggml_rms_norm(c, t, eps), w); };
            auto rope = [&](ggml_tensor * t) {
                return ggml_rope_ext(c, t, tpos, nullptr, (int) head_dim, GGML_ROPE_TYPE_NEOX, n_ctx_orig,
                                     rope_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
            };
            ggml_tensor * cur = norm(xin, L.an);
            ggml_tensor * q = ggml_reshape_3d(c, ggml_mul_mat(c, L.wq, cur), head_dim, n_head, n);
            ggml_tensor * k = ggml_reshape_3d(c, ggml_mul_mat(c, L.wk, cur), head_dim, n_head_kv, n);
            ggml_tensor * v = ggml_reshape_3d(c, ggml_mul_mat(c, L.wv, cur), head_dim, n_head_kv, n);
            q = rope(norm(q, L.qn));
            k = rope(norm(k, L.kn));
            q = ggml_permute(c, q, 0, 2, 1, 3);
            k = ggml_cast(c, ggml_permute(c, k, 0, 2, 1, 3), GGML_TYPE_F16);
            v = ggml_cast(c, ggml_permute(c, v, 0, 2, 1, 3), GGML_TYPE_F16);
            ggml_tensor * a = ggml_flash_attn_ext(c, q, k, v, tmask, 1.0f / sqrtf((float) head_dim), 0.0f, 0.0f);
            ggml_prec_set_acc(a, GGML_PREC_F32);
            a = ggml_reshape_2d(c, a, a->ne[0] * a->ne[1], a->ne[2] * a->ne[3]);
            ggml_tensor * h = ggml_add(c, ggml_mul_mat(c, L.wo, a), xin);
            cur = norm(h, L.fn);
            cur = ggml_swiglu_split(c, ggml_mul_mat(c, L.wg, cur), ggml_mul_mat(c, L.wu, cur));
            ggml_tensor * xo = ggml_add(c, ggml_mul_mat(c, L.wd, cur), h);
            ggml_set_output(xo);

            ggml_cgraph * gr = ggml_new_graph_custom(c, 256, false);
            ggml_build_forward_expand(gr, xo);
            ggml_backend_sched_reset(sched);
            if (!ggml_backend_sched_alloc_graph(sched, gr)) { ggml_free(c); fprintf(stderr, "[enc] alloc failed\n"); return false; }
            ggml_backend_tensor_set(xin, x.data(), 0, x.size() * sizeof(float));
            ggml_backend_tensor_set(tpos, pos.data(), 0, pos.size() * sizeof(int32_t));
            ggml_backend_tensor_set(tmask, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
            if (ggml_backend_sched_graph_compute(sched, gr) != GGML_STATUS_SUCCESS) { ggml_free(c); fprintf(stderr, "[enc] compute failed at layer %d\n", il); return false; }
            ggml_backend_tensor_get(xo, y.data(), 0, y.size() * sizeof(float));
            ggml_free(c);
            x.swap(y);
            t_comp += now_s() - tc;
        }
        for (auto * lc : lctx) if (lc) ggml_free(lc);
        out.swap(x);
        return true;
    }

    void close() {
        if (sched) ggml_backend_sched_free(sched);
        for (auto & b : slot) if (b) ggml_backend_buffer_free(b);
        if (htp) ggml_backend_free(htp);
        if (cpu) ggml_backend_free(cpu);
        if (pk) fclose(pk);
        if (gf) fclose(gf);
        if (g) gguf_free(g);
    }
};

// The same READY / "<text>\t<out>" / DONE protocol as the llama.cpp path in zimage_encode.cpp.
static int run_stream(const std::string & model_path, const std::string & device, int layer) {
    const double t0 = now_s();
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;                                   // llama.cpp only tokenizes
    llama_model * vm = llama_model_load_from_file(model_path.c_str(), mp);
    Engine E;
    if (!vm || !E.open(model_path, device, layer)) {
        printf("ERROR cannot open the streaming encoder\n");
        fflush(stdout);
        E.close();
        if (vm) llama_model_free(vm);
        return 1;
    }
    const double t_ready = now_s() - t0;
    printf("READY\n");
    fflush(stdout);

    int rc = 0;
    std::string line;
    if (std::getline(std::cin, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        const size_t tab = line.find('\t');
        if (tab == std::string::npos) {
            printf("ERROR expected '<text file>\\t<output file>'\n");
            rc = 1;
        } else {
            const std::string in_path = line.substr(0, tab), out_path = line.substr(tab + 1);
            std::ifstream f(in_path, std::ios::binary);
            std::stringstream ss;
            ss << f.rdbuf();
            const std::string text = ss.str();
            const llama_vocab * vocab = llama_model_get_vocab(vm);
            std::vector<llama_token> tok(text.size() + 16);
            int n = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), tok.data(), (int32_t) tok.size(), true, true);
            std::vector<float> out;
            const double te = now_s();
            if (n <= 0) {
                printf("ERROR tokenize gave %d tokens\n", n);
                rc = 1;
            } else if (tok.resize(n), !E.encode(tok, out)) {
                printf("ERROR encode failed\n");
                rc = 1;
            } else {
                FILE * o = fopen(out_path.c_str(), "wb");
                if (!o) { printf("ERROR cannot open %s\n", out_path.c_str()); rc = 1; }
                else {
                    fwrite(out.data(), sizeof(float), out.size(), o);
                    fclose(o);
                    printf("DONE %d\n", n);
                }
            }
            if (getenv("ZI_ENC_LOG")) {
                fprintf(stderr, "[enc] ready %.2f s | encode %.2f s = read %.2f + copy %.2f + compute %.2f | slot %.1f MB x2\n",
                        t_ready, now_s() - te, E.t_read, E.t_copy, E.t_comp, E.slot_size / 1048576.0);
            }
        }
        fflush(stdout);
    }
    E.close();
    llama_model_free(vm);
    return rc;
}

}  // namespace zienc
