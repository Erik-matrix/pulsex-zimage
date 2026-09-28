// z-image-turbo DiT block chain, streamed ONE BLOCK AT A TIME: load a single
// block's weights, run it, copy the output back to host memory, free
// everything, move to the next block. Matches how dit_full_engine.cpp always
// ran (weights streamed per-layer, never all resident at once) -- much
// lighter on RAM than zimage_full.cpp's all-blocks-in-one-graph approach,
// and the natural fit for the elev (distilled student) weights later, since
// those are tiny per-layer and this loop already visits layers one by one.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <sys/stat.h>
#include <cstddef>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <mutex>
#include <thread>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <string>
#include <vector>

static const int DIM = 3840, NH = 30, HD = 128, HID = 10240;
static const float EPS = 1e-5f;
static const int AXES[3] = {32, 48, 48};
static const float THETA = 256.0f;
static const int NL = 30, NR = 2, CAPD = 2560, PATCH_IN = 64, PATCH = 2, INCH = 16;

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

// ZI_ROPE_FUSED=1 -> EN ggml_rope_multi(MSECT) i stallet for 3 rope + 2 concat.
// MSECT = sammanhangande sektioner med theta-omstart per sektion (som VISION) men
// NORMAL parning (intilliggande par), vilket ar exakt vad de tre separata
// NORMAL-ropesen gor. sections anges i PAR-enheter: 32/48/48 dims -> 16/24/24 par.
static ggml_tensor * apply_rope(ggml_context * ctx, ggml_tensor * t, ggml_tensor * ids0, ggml_tensor * ids1, ggml_tensor * ids2, ggml_tensor * ids_all, ggml_tensor * ff) {
    static const int ZI_ROPE_FUSED = []{ const char * e = getenv("ZI_ROPE_FUSED"); return e ? atoi(e) : 1; }();
    if (ZI_ROPE_FUSED && ids_all) {
        int sections[4] = { AXES[0] / 2, AXES[1] / 2, AXES[2] / 2, 0 };
        return ggml_rope_multi(ctx, t, ids_all, ff, HD, sections,
                               GGML_ROPE_TYPE_MSECT, 0, THETA, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    }
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

// stage_dbg[0..5] = x_after_attn, w1o, w3o, inter, w2o, w2n (when non-null)
static ggml_tensor * build_block(ggml_context * ctx, ggml_tensor * x, ggml_tensor * ids0, ggml_tensor * ids1, ggml_tensor * ids2,
                                  ggml_tensor * ids_all, ggml_tensor * ff,
                                  ggml_tensor * adaln, const BlockWeights & w, bool mod, ggml_tensor ** m_dbg = nullptr,
                                  ggml_tensor ** stage_dbg = nullptr,
                                  std::vector<ggml_tensor *> * pre = nullptr) {
    // ZI_ADALN_FOLD (forval 1): se tools/patch_adaln_fold.py for resonemanget.
    static const int ZI_ADALN_FOLD = []{ const char * e = getenv("ZI_ADALN_FOLD"); return e ? atoi(e) : 1; }();
    const int64_t S = x->ne[1];
    ggml_tensor * smsa = nullptr, * gmsa = nullptr, * smlp = nullptr, * gmlp = nullptr;
    if (mod) {
        ggml_tensor * m = ggml_add(ctx, ggml_mul_mat(ctx, w.ada_w, adaln), w.ada_b);
        if (m_dbg) *m_dbg = ggml_cont(ctx, m);
        ggml_tensor * m_smsa = ggml_view_1d(ctx, m, DIM, 0 * DIM * sizeof(float));
        ggml_tensor * m_gmsa = ggml_view_1d(ctx, m, DIM, 1 * DIM * sizeof(float));
        ggml_tensor * m_smlp = ggml_view_1d(ctx, m, DIM, 2 * DIM * sizeof(float));
        ggml_tensor * m_gmlp = ggml_view_1d(ctx, m, DIM, 3 * DIM * sizeof(float));
        smsa = ggml_scale_bias(ctx, m_smsa, 1.0f, 1.0f);
        gmsa = ggml_tanh(ctx, m_gmsa);
        smlp = ggml_scale_bias(ctx, m_smlp, 1.0f, 1.0f);
        gmlp = ggml_tanh(ctx, m_gmlp);
    }

    auto fold = [&](ggml_tensor * nw, ggml_tensor * mw) {
        ggml_tensor * p = ggml_mul(ctx, nw, mw);   // [DIM] x [DIM]
        if (pre) pre->push_back(p);
        return p;
    };
    const bool do_fold = mod && ZI_ADALN_FOLD && pre;

    ggml_tensor * h = do_fold ? ggml_mul(ctx, ggml_rms_norm(ctx, x, EPS), fold(w.an1, smsa))
                              : rms_norm_w(ctx, x, w.an1);
    if (stage_dbg) stage_dbg[12] = ggml_cont(ctx, h);            // h after rmsnorm (pre gate-mul)
    if (mod && !do_fold) h = ggml_mul(ctx, h, smsa);
    if (stage_dbg) stage_dbg[13] = ggml_cont(ctx, h);            // h after gate broadcast-mul
    ggml_tensor * qkv = ggml_mul_mat(ctx, w.wqkv, h);
    if (stage_dbg) stage_dbg[6] = ggml_cont(ctx, qkv);

    // ZI_QKV_FUSED (forval 1): Q/K/V som VYER rakt in i qkv i stallet for CONT-kopior.
    // Profil 1024 px fore: per block gick Q, K och V vardera genom TRE fulla pass a 63 MB --
    // CONT ur qkv (0,50 s/bild), CONT efter permute (0,64 s), CPY f32->f16 (1,79 s).
    // ggml-hexagon klarar steg: RMS_NORM kraver bara sammanhangande RADER (128 floats har),
    // och CPY-konverteringen (cpy-ops.c, cpy_thread_f32_f16_sameshape) stegar varje rad med
    // nb01/nb02/nb03. Samma matematik, farre kopior => bilden ska vara BIT-IDENTISK.
    static const int ZI_QKV_FUSED = []{ const char * e = getenv("ZI_QKV_FUSED"); return e ? atoi(e) : 2; }();
    ggml_tensor * q;
    ggml_tensor * k;
    ggml_tensor * v;
    if (ZI_QKV_FUSED) {
        // [HD, NH, S]: rader HD floats sammanhangande, huvuden HD*4 isar, tokens qkv->nb[1] isar
        q = ggml_view_3d(ctx, qkv, HD, NH, S, HD * sizeof(float), qkv->nb[1], 0 * DIM * sizeof(float));
        k = ggml_view_3d(ctx, qkv, HD, NH, S, HD * sizeof(float), qkv->nb[1], 1 * DIM * sizeof(float));
        v = ggml_view_3d(ctx, qkv, HD, NH, S, HD * sizeof(float), qkv->nb[1], 2 * DIM * sizeof(float));
    } else {
        q = ggml_cont(ctx, ggml_view_2d(ctx, qkv, DIM, S, qkv->nb[1], 0 * DIM * sizeof(float)));
        k = ggml_cont(ctx, ggml_view_2d(ctx, qkv, DIM, S, qkv->nb[1], 1 * DIM * sizeof(float)));
        v = ggml_cont(ctx, ggml_view_2d(ctx, qkv, DIM, S, qkv->nb[1], 2 * DIM * sizeof(float)));
        q = ggml_reshape_3d(ctx, q, HD, NH, S);
        k = ggml_reshape_3d(ctx, k, HD, NH, S);
        v = ggml_reshape_3d(ctx, v, HD, NH, S);
    }

    // Ingen reshape_3d: w.qn/w.kn ar redan [HD] och broadcastar lika bra. reshape_3d ar en
    // EGEN grafnod, och djupet-forst-ordningen blev RMS_NORM -> RESHAPE -> MUL -- sa
    // ggml-hexagons RMS_NORM+MUL-fusion (kraver dem intill varandra) slog aldrig till.
    ggml_tensor * qn3 = w.qn;
    ggml_tensor * kn3 = w.kn;
    q = ggml_mul(ctx, ggml_rms_norm(ctx, q, EPS), qn3);
    k = ggml_mul(ctx, ggml_rms_norm(ctx, k, EPS), kn3);
    q = apply_rope(ctx, q, ids0, ids1, ids2, ids_all, ff);
    k = apply_rope(ctx, k, ids0, ids1, ids2, ids_all, ff);
    if (stage_dbg) stage_dbg[7] = ggml_cont(ctx, q);

    const float scl = 1.0f / sqrtf((float) HD);
    // Flash-attention (streamar attention, materialiserar EJ [S,S,NH]-scores -> skalar forbi 32MiB-arg-cap).
    // ZI_FLASH=0 -> gamla manuella vagen (for A/B-jamforelse). q/k/v blir [HD,S,NH,1].
    static int USE_FLASH = []{ const char * e = getenv("ZI_FLASH"); return e ? atoi(e) : 1; }();

    // Fuserat: permute + f16-konvertering i EN CPY (ggml_cast pa en permuterad vy). Annars
    // forst en CONT till sammanhangande f32, sedan en CPY till f16 -- tva pass over 63 MB.
    const bool fuse_cast = ZI_QKV_FUSED && USE_FLASH;
    if (!fuse_cast) {
        q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
        k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
        v = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));
    }

    ggml_tensor * attn;
    if (USE_FLASH) {
        ggml_tensor * q16;
        ggml_tensor * k16;
        ggml_tensor * v16;
        if (ZI_QKV_FUSED >= 2) {
            // Nivå 2 (FORVAL): Q in som PERMUTERAD f32-VY, ingen kopia alls. Numeriskt IDENTISK med
            // niva 1: bilden traffade byte for byte ett utfall som niva 1 sjalv ocksa ger (motorn
            // har minst tva utfall for identiska indata, se ZI_SWIGLU nedan). -0,49 s op-tid. HMX-flashen har en egen
            // hmx_fa_q_prep_fp32_d4 (DK=128) som konverterar till fp16-tegel i VTCM och valjer
            // layout via q_transposed = nb[1] < nb[2]. K/V: konvertera den SAMMANHANGANDE
            // [HD,NH,S]-tensorn (snabbaste CPY-formen) och ge flash en permuterad vy -- dess
            // DMA tar godtyckligt radsteg k->nb[1] och huvudsteg k->nb[2].
            q16 = ggml_permute(ctx, q, 0, 2, 1, 3);
            // Konvertera med LANGA rader (3840) i stallet for 128: CPY-karnan har en fast kostnad
            // per rad (l2fetch, pekarrakning). Matt 1024 px: [128,30,S] 5,08 ms/anrop mot
            // [3840,S] 2,28 ms for samma datamangd. Samma konvertering per element.
            // K ar sammanhangande efter rope -> reshape_2d ar gratis. V ar en vy in i qkv, men
            // varje tokens 30 huvuden ligger sammanhangande -> view_2d med radsteg qkv->nb[1].
            // ZI_KV_F32 (forval 1): K/V in som f32-VYER, ingen CPY alls - HMX-flashens prepp
            // konverterar raderna pa plats med samma hvx_vec_f32_to_f16 som CPY-op:en anvander
            // (ggml-hexagon tools_local/patch_fa_kv_f32.py). 2 x 63 MB farre pass per block.
            static const int ZI_KV_F32 = []{ const char * e = getenv("ZI_KV_F32"); return e ? atoi(e) : 1; }();
            if (ZI_KV_F32) {
                k16 = ggml_permute(ctx, ggml_reshape_3d(ctx, k, HD, NH, S), 0, 2, 1, 3);
                v16 = ggml_permute(ctx, ggml_view_3d(ctx, qkv, HD, NH, S, HD * sizeof(float), qkv->nb[1],
                                                     2 * DIM * sizeof(float)), 0, 2, 1, 3);
            } else {
            k16 = ggml_permute(ctx, ggml_reshape_3d(ctx,
                      ggml_cast(ctx, ggml_reshape_2d(ctx, k, DIM, S), GGML_TYPE_F16), HD, NH, S), 0, 2, 1, 3);
            v16 = ggml_permute(ctx, ggml_reshape_3d(ctx,
                      ggml_cast(ctx, ggml_view_2d(ctx, qkv, DIM, S, qkv->nb[1], 2 * DIM * sizeof(float)), GGML_TYPE_F16),
                      HD, NH, S), 0, 2, 1, 3);
            }
        } else {
            q16 = fuse_cast ? ggml_cast(ctx, ggml_permute(ctx, q, 0, 2, 1, 3), GGML_TYPE_F16)
                            : ggml_cast(ctx, q, GGML_TYPE_F16);   // ggml-hexagon f16-flash-karna: allt F16
            k16 = fuse_cast ? ggml_cast(ctx, ggml_permute(ctx, k, 0, 2, 1, 3), GGML_TYPE_F16)
                            : ggml_cast(ctx, k, GGML_TYPE_F16);
            v16 = fuse_cast ? ggml_cast(ctx, ggml_permute(ctx, v, 0, 2, 1, 3), GGML_TYPE_F16)
                            : ggml_cast(ctx, v, GGML_TYPE_F16);
        }
        attn = ggml_flash_attn_ext(ctx, q16, k16, v16, NULL, scl, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
        // ZI_FA_FAST_EXP2 (forval 1): MEDVETET PRECISIONSVAL (2026-09-27) - grad-2-exp2 i
        // flash-softmaxen (ggml-hexagon tools_local/patch_fa_fast_exp2.py). op_params[4] bit 0; varden
        // tar den bara med pipelinen pa. Matt 1024: flash 86,0 -> 83,5 ms/anrop (-0,32 s/bild), CPU-fel
        // 9,04e-5 mot 8,9e-5, bild 0,92 LSB medel mot exakt - samma monster som NPU-bruset (paels,
        // kanter), ogat ser ingen skillnad. ⛔ Invikningen av s-m i exp2 (qf16-brakdel) forkastad:
        // samma op-fel men ETT ANNAT SAMPEL (11 LSB). ZI_FA_FAST_EXP2=0 = exakta vagen.
        static const int ZI_FA_FAST_EXP2 = []{ const char * e = getenv("ZI_FA_FAST_EXP2"); return e ? atoi(e) : 1; }();
        if (ZI_FA_FAST_EXP2) attn->op_params[4] |= 1;
        attn = ggml_reshape_2d(ctx, attn, DIM, S);             // [d,n_head,n_q] -> [DIM,S]
    } else {
        ggml_tensor * scores = ggml_scale(ctx, ggml_mul_mat(ctx, k, q), scl);
        scores = ggml_soft_max(ctx, scores);
        if (stage_dbg) stage_dbg[8] = ggml_cont(ctx, scores);
        ggml_tensor * v_t = ggml_cont(ctx, ggml_permute(ctx, v, 1, 0, 2, 3));
        ggml_tensor * a = ggml_mul_mat(ctx, v_t, scores);
        a = ggml_cont(ctx, ggml_permute(ctx, a, 0, 2, 1, 3));
        attn = ggml_reshape_2d(ctx, a, DIM, S);
    }
    if (stage_dbg) stage_dbg[9] = ggml_cont(ctx, attn);

    static float ZC = []{ const char * e = getenv("ZI_CLAMP"); return e ? (float) atof(e) : 0.0f; }();
    static float ZS = []{ const char * e = getenv("ZI_SCALE"); return e ? (float) atof(e) : 0.0f; }();
    if (ZC > 0.0f) attn = ggml_clamp(ctx, attn, -ZC, ZC);
    ggml_tensor * aop;
    if (ZS > 0.0f) {
        aop = ggml_scale(ctx, ggml_mul_mat(ctx, w.wo, ggml_scale(ctx, attn, 1.0f / ZS)), ZS);
    } else {
        aop = ggml_mul_mat(ctx, w.wo, attn);
    }
    if (stage_dbg) stage_dbg[10] = ggml_cont(ctx, aop);
    ggml_tensor * aon = do_fold ? ggml_mul(ctx, ggml_rms_norm(ctx, aop, EPS), fold(w.an2, gmsa))
                                : rms_norm_w(ctx, aop, w.an2);
    if (stage_dbg) stage_dbg[11] = ggml_cont(ctx, aon);
    x = (mod && !do_fold) ? ggml_add(ctx, x, ggml_mul(ctx, aon, gmsa)) : ggml_add(ctx, x, aon);
    if (stage_dbg) stage_dbg[0] = ggml_cont(ctx, x);

    ggml_tensor * hf = do_fold ? ggml_mul(ctx, ggml_rms_norm(ctx, x, EPS), fold(w.fn1, smlp))
                               : rms_norm_w(ctx, x, w.fn1);
    if (mod && !do_fold) hf = ggml_mul(ctx, hf, smlp);
    ggml_tensor * w1o = ggml_mul_mat(ctx, w.w1, hf);
    ggml_tensor * w3o = ggml_mul_mat(ctx, w.w3, hf);
    if (stage_dbg) { stage_dbg[1] = ggml_cont(ctx, w1o); stage_dbg[2] = ggml_cont(ctx, w3o); }
    // ZI_SWIGLU: silu(w1o)*w3o som EN GLU-op i stallet for SILU + MUL. Tva op:ar gor
    // 5 x 168 MB DDR-trafik per block (SILU las+skriv, MUL las 2+skriv), GLU gor 3.
    // Profil fore: SILU 13,3 ms + MUL 6,8 ms per block.
    // FORVAL: op-tid -0,67 s, och w1/w3 hamnar intill varandra i grafen sa ggml-hexagon
    // fuserar dem till MUL_MAT_NX (663 -> 403 MUL_MAT). Bilden = byte for byte ett utfall
    // som SILU+MUL-vagen ocksa ger. ⚠ Motorn har TVA utfall for identiska indata (cbf972 /
    // feac219 samma dag, samma binar) -- en SKILLNAD kraver N>=3, en exakt traff ar bevis.
    static const int ZI_SWIGLU = []{ const char * e = getenv("ZI_SWIGLU"); return e ? atoi(e) : 1; }();
    ggml_tensor * inter = ZI_SWIGLU ? ggml_swiglu_split(ctx, w1o, w3o)
                                    : ggml_mul(ctx, ggml_silu(ctx, w1o), w3o);
    // ZI_SWIGLU_F16 (forval 1): MEDVETET PRECISIONSVAL (2026-09-27). DSP-karnan raknar
    // sigmoiden i fp16 (64 banor/vektor) nar op_params[4] != 0; x0*s*x1 stannar f32. Se
    // hvx_swiglu_f16sig_f32_aa i htp/hvx-sigmoid.h. op_params[4] anvands inte av ggml for GLU
    // och kopieras ordagrant till DSP:n. ZI_SWIGLU_F16=0 = exakta f32-vagen, utan omstart av skel.
    static const int ZI_SWIGLU_F16 = []{ const char * e = getenv("ZI_SWIGLU_F16"); return e ? atoi(e) : 1; }();
    if (ZI_SWIGLU && ZI_SWIGLU_F16) inter->op_params[4] = 1;
    if (stage_dbg) stage_dbg[3] = ggml_cont(ctx, inter);
    if (ZC > 0.0f) inter = ggml_clamp(ctx, inter, -ZC, ZC);
    ggml_tensor * w2o;
    if (ZS > 0.0f) {
        w2o = ggml_scale(ctx, ggml_mul_mat(ctx, w.w2, ggml_scale(ctx, inter, 1.0f / ZS)), ZS);
    } else {
        w2o = ggml_mul_mat(ctx, w.w2, inter);
    }
    if (stage_dbg) stage_dbg[4] = ggml_cont(ctx, w2o);
    ggml_tensor * w2n = do_fold ? ggml_mul(ctx, ggml_rms_norm(ctx, w2o, EPS), fold(w.fn2, gmlp))
                                : rms_norm_w(ctx, w2o, w.fn2);
    if (stage_dbg) stage_dbg[5] = ggml_cont(ctx, w2n);
    x = (mod && !do_fold) ? ggml_add(ctx, x, ggml_mul(ctx, w2n, gmlp)) : ggml_add(ctx, x, w2n);

    return x;
}

// ---- streaming driver: one block per call, own ctx/backend buffer, freed on return ----

// Weights of one block, kept on the device between diffusion steps. Without this the DiT
// re-reads and re-packs every block's weights once per step, which costs about 7.6 s per
// step at 1024px - more than the whole flash-attention kernel.
struct BlockCache {
    ggml_context *        ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    BlockWeights          w{};
};

// Where the wall clock goes, summed over every run_block call. Printed at exit so we can
// see how much is weight loading and how much is shuttling activations through the host.
struct StreamTimes {
    int64_t weights = 0;   // gguf read + tensor_set of this block's weights
    int64_t io_up   = 0;   // x_in, ids, adaln up to the device
    int64_t compute = 0;   // graph compute
    int64_t io_down = 0;   // x_out back to the host
    int64_t total   = 0;   // whole run_block call, so graph build and the rest can be derived
    // Uppdelning av restposten 'graf+ovrigt' (8,3 s av 74,4 vid 1024 = 64 ms per block):
    // bygga grafen, allokera den, och allt annat i run_block.
    int64_t g_pre   = 0;   // run_block-start -> viktladdningen borjar
    int64_t w_wait  = 0;   // vantan pa prefetch-traden (ska vara ~0 nar den hinner)
    int64_t w_read  = 0;   // fread fran GGUF
    int64_t w_set   = 0;   // tensor_set = tegelpackningen
    int64_t w_bytes = 0;
    int64_t g_dbg   = 0;   // t_c1 -> t_gap0, diagnostikregionen
    int64_t g_gap   = 0;   // t_c1 -> t_d0, ska vara ~0
    int64_t g_post  = 0;   // compute klar -> run_block slut (ggml_free, kopior)
    int64_t g_free  = 0;   // ggml_backend_buffer_free + ggml_free, EFTER gamla avlasningen
    int64_t r_wait  = 0;   // ringen: vantan pa platsens fence (forra blocket som last den)
    int64_t r_fills = 0;   // ringen: antal platsfyllningar (strommade block)
    int64_t r_hits  = 0;   // ringen: blocket lag redan kvar i sin plats
    int      calls  = 0;
};
// Per-lager-diagnostiken (`[stream] layers.N ...`) kostar en 63 MB-KOPIA av aktiveringen
// plus TVA fulla svep over 15,8 M flyttal, 120 ganger per bild = 7,6 GB kopiering, allt
// for en utskriftsrad. Matt: 4,1 s av 61 (6,7%). NaN fangas anda av harnessen pa
// slutlatenten. Forval AV; ZI_STATS=1 slar pa den.
static const int ZI_STATS = []{ const char * e = getenv("ZI_STATS"); return e ? atoi(e) : 0; }();

static StreamTimes g_times;

// ZI_PREFETCH: las NASTA blocks vikter fran disk pa en egen trad medan DET HAR blocket
// raknar pa DSP:n. Matt utan prefetch: `vikter` 4,7 s = disk 2,05 s (3,46 GB @ 1688 MB/s,
// alltsa redan vid C:s hardvarutak) + repack 2,40 s. Bara overlappning kan ta disken.
//
// ⛔ ETT BLOCK I FORVAG, inte hela modellen: ett block ar ~105 MB, hela DiT:n 3,46 GB.
// Maskinen kor 16 GB med committed redan pa 14 GB - obegransad prefetch ar en minnesbomb.
static const int ZI_RESIDENT = []{ const char * e = getenv("ZI_RESIDENT"); return e ? atoi(e) : 1; }();
static const int ZI_PREFETCH = []{ const char * e = getenv("ZI_PREFETCH"); return e ? atoi(e) : 1; }();
// ZI_WEIGHT_BUDGET_MB: tak for blockvikterna pa NPU:n (residenta block + ringens platser).
// Osatt/negativ = obegransat (alla block residenta, som forut). Block blir residenta i den
// ordning de kors sa lange de ryms under taket MINUS ringen; ovriga strommas varje steg genom
// ZI_RING_SLOTS forallokerade viktplatser. Grunden for Flux/video: NPU-minnet ar pinnat och
// raknas som committed, sa taket styr hela processens fotavtryck.
static const long long ZI_WEIGHT_BUDGET_MB = []{ const char * e = getenv("ZI_WEIGHT_BUDGET_MB"); return e && *e ? atoll(e) : -1LL; }();
static const int ZI_RING_SLOTS = []{ const char * e = getenv("ZI_RING_SLOTS"); int n = e ? atoi(e) : 2; return n < 1 ? 1 : (n > 4 ? 4 : n); }();

// Namnen pa ett blocks vikttensorer. MASTE vara samma lista som `get(...)`-anropen i
// run_block, annars laser traden fel byteintervall och blocket faller tillbaka till fread.
static std::vector<std::string> block_tensor_names(const std::string & pfx, bool mod) {
    std::vector<std::string> n = {
        pfx + ".attention_norm1.weight", pfx + ".attention_norm2.weight",
        pfx + ".ffn_norm1.weight",       pfx + ".ffn_norm2.weight",
        pfx + ".attention.q_norm.weight", pfx + ".attention.k_norm.weight",
        pfx + ".attention.qkv.weight",    pfx + ".attention.out.weight",
        pfx + ".feed_forward.w1.weight",  pfx + ".feed_forward.w3.weight",
        pfx + ".feed_forward.w2.weight",
    };
    if (mod) {
        n.push_back(pfx + ".adaLN_modulation.0.weight");
        n.push_back(pfx + ".adaLN_modulation.0.bias");
    }
    return n;
}

struct Prefetcher {
    std::thread              th;
    std::mutex               mu;
    std::condition_variable  cv_req;   // huvudtraden -> arbetaren
    std::condition_variable  cv_done;  // arbetaren -> huvudtraden
    std::string              want;     // begart pfx, tomt = inget att gora
    bool                     want_mod = false;
    std::string              inflight; // pfx som arbetaren laser JUST NU
    std::string              have;     // pfx som ligger i `buf`
    bool                     busy  = false;
    bool                     quit  = false;
    std::map<int64_t, std::vector<uint8_t>> buf;   // gguf-tensor-id -> bytes
    FILE *                   f = nullptr;          // EGET handtag: egen filposition
};

static size_t g_mem_w = 0, g_mem_act = 0, g_mem_io = 0;   // NPU-buffertar per slag (minnesrapport)

struct StreamCtx {
    gguf_context * gctx;
    FILE * gguf_file;
    size_t data_base;
    ggml_backend_t backend;
    std::map<std::string, BlockCache> wcache;
    // One graph allocator for the whole run. Building a new one per block frees and
    // re-reserves the intermediate buffer 130 times per image, which shows up as the NPU
    // memory sawtooth and costs more than the weight residency saves.
    // Utdatabufferten aterbrukas mellan block. Forut allokerades en NY std::vector per
    // anrop: 3840 x 4109 floats = 63 MB som VARDEINITIERAS (nollas) och sidfelas in varje
    // gang. Matt: 1,5 s av 1,9 s 'graf+ovrigt' lag EFTER compute, alltsa NPU:n stillastaende
    // medan varden nollade 63 MB. resize() pa oforandrad storlek ar en no-op och nollar inte.
    std::vector<float> out_buf;
    // Stagingbuffert for f16-transport av aktiveringen (ZI_F16_IO, forval PA).
    // Matt 1024px/4 steg: in 2,55 -> 1,90 s OCH compute 48,6 -> 47,35 s. Compute-delen
    // ar storst: HMX ater bara fp16, sa DSP:n konverterade f32->f16 per matmul
    // (`transfer_activation_chunk_fp32_to_fp16`, matmul-ops.c:2024). Nu gors det en
    // gang pa varden i stallet. Totalt 60,3 -> 58,3 s = 3,3%.
    // ⚠ INTE gratis: x avrundas till fp16 vid varje blockgrans. Bildskillnad mot f32
    // ar cos 0,99975, medel 0,334 LSB, och den sitter pa HOGFREKVENTA KANTER (pixlar
    // >16 LSB har medelgradient 35,8 mot bildens 1,9). Mindre an ovriga andringar
    // gjorda samma dag. ZI_F16_IO=0 stanger av.
    std::vector<ggml_fp16_t> in_buf;
    Prefetcher pf;
    ggml_gallocr_t galloc = nullptr;
    // ZI_ACT_RESIDENT: residualstrommen pa NPU:n mellan blocken, tva platser a act_slot byte.
    // act_cur = platsen som haller senaste blockets utdata (-1 = varden ar auktoritativ).
    ggml_backend_buffer_t act_buf = nullptr;
    size_t  act_slot = 0;
    int     act_cur  = -1;
    int64_t act_S    = 0;
    // De sma indatatensorerna (ids, i_all, ff, adaln) - persistenta i stallet for en ny buffert per
    // block. Se patch_io_buf_persistent.py.
    ggml_backend_buffer_t io_buf = nullptr;
    size_t  io_cap   = 0;
    // Buffertringen (ZI_WEIGHT_BUDGET_MB). Varje plats: en WEIGHTS-buffert (repack i tensor_set),
    // mappad till DSP:n vid forsta bruk och sedan aldrig om; ett fence-event som spelas in efter
    // blockets graf; agaren (pfx) och dess tensorer, sa ett block som ligger kvar ateranvands.
    ggml_backend_buffer_t ring_buf[4]   = {};
    ggml_backend_event_t  ring_ev[4]    = {};
    bool                  ring_rec[4]   = {};
    ggml_context *        ring_ctx[4]   = {};
    BlockWeights          ring_w[4]     = {};
    std::string           ring_owner[4];
    size_t                ring_cap      = 0;
    int                   ring_next     = 0;
    int                   n_resident    = 0;
    std::map<std::string, bool> streamed;   // block som fatt nej av budgeten (beslutet ar stabilt)
    // ZI_PACKCACHE: blockvikterna i tegelpackad form (se pack_cache_open). on = anvands.
    struct {
        bool   on = false;
        FILE * f  = nullptr;
        std::map<int64_t, std::pair<uint64_t, uint64_t>> ent;   // gguf-id -> (offset, bytes) i cachefilen
        bool (*get_packed)(const ggml_tensor *, void *, size_t) = nullptr;
        bool (*set_packed)(ggml_tensor *, const void *, size_t) = nullptr;
    } pk;
};

// Var en vikts bytes ligger: i packcachen (fardigpackad) eller i GGUF:en (ra, packas i tensor_set).
static bool weight_src(const StreamCtx & sc, int64_t id, size_t & off, size_t & sz) {
    if (sc.pk.on) {
        auto it = sc.pk.ent.find(id);
        if (it != sc.pk.ent.end()) { off = (size_t) it->second.first; sz = (size_t) it->second.second; return true; }
    }
    off = sc.data_base + gguf_get_tensor_offset(sc.gctx, id);
    sz  = gguf_get_tensor_size(sc.gctx, id);
    return false;
}
static size_t g_mem_ring = 0;

// Bytes ett blocks vikter tar i en hexagon-WEIGHTS-buffert (tegelpackade = utfyllda till 32).
static size_t block_alloc_bytes(StreamCtx & sc, const std::string & pfx, bool mod) {
    ggml_backend_buffer_type_t bt = ggml_backend_get_default_buffer_type(sc.backend);
    const size_t al = ggml_backend_buft_get_alignment(bt);
    ggml_init_params p = { ggml_tensor_overhead() * 32, nullptr, true };
    ggml_context * c = ggml_init(p);
    size_t n = 0;
    for (const auto & nm : block_tensor_names(pfx, mod)) {
        int64_t id = gguf_find_tensor(sc.gctx, nm.c_str());
        if (id < 0) continue;
        ggml_tensor * t = ggml_new_tensor(c, gguf_get_tensor_type(sc.gctx, id), GGML_MAX_DIMS, gguf_get_tensor_ne(sc.gctx, id));
        n += GGML_PAD(ggml_backend_buft_get_alloc_size(bt, t), al);
    }
    ggml_free(c);
    return n;
}

// Far blocket bli residentt? Beslutet tas forsta gangen blocket kors och star sedan fast.
static bool ring_wants(StreamCtx & sc, const std::string & pfx, bool mod) {
    if (ZI_WEIGHT_BUDGET_MB < 0) return false;
    auto it = sc.streamed.find(pfx);
    if (it != sc.streamed.end()) return it->second;
    // Platsstorleken = storsta blocket (alla mod-block ar lika; context_refiner ar mindre).
    const size_t slot   = block_alloc_bytes(sc, "layers.0", true);
    const long long lim = ZI_WEIGHT_BUDGET_MB * 1048576LL - (long long) slot * ZI_RING_SLOTS;
    const bool st = (long long) (g_mem_w + block_alloc_bytes(sc, pfx, mod)) > lim;
    sc.streamed[pfx] = st;
    return st;
}

// Arbetaren: sover pa cv_req, laser ett blocks byteintervall till `buf`, signalerar.
// Den ror ALDRIG backenden eller ggml - bara sitt eget FILE* och sin egen map.
static void prefetch_worker(StreamCtx * sc) {
    Prefetcher & pf = sc->pf;
    for (;;) {
        std::string pfx;
        bool mod;
        {
            std::unique_lock<std::mutex> lk(pf.mu);
            pf.cv_req.wait(lk, [&]{ return pf.quit || !pf.want.empty(); });
            if (pf.quit) return;
            pfx = pf.want; mod = pf.want_mod;
            pf.want.clear();
            pf.busy = true;
            pf.inflight = pfx;
            pf.have.clear();
            pf.buf.clear();
        }

        std::map<int64_t, std::vector<uint8_t>> local;
        for (const auto & nm : block_tensor_names(pfx, mod)) {
            int64_t id = gguf_find_tensor(sc->gctx, nm.c_str());
            if (id < 0) { local.clear(); break; }   // tyst fallback till fread i run_block
            size_t off, sz;
            weight_src(*sc, id, off, sz);
            std::vector<uint8_t> b(sz);
            _fseeki64(pf.f, (long long) off, SEEK_SET);
            if (fread(b.data(), 1, sz, pf.f) != sz) { local.clear(); break; }
            local[id] = std::move(b);
        }

        {
            std::lock_guard<std::mutex> lk(pf.mu);
            if (!local.empty()) { pf.buf = std::move(local); pf.have = pfx; }
            pf.inflight.clear();
            pf.busy = false;
        }
        pf.cv_done.notify_all();
    }
}

// Begar nasta block. Anropas av loopen FORE run_block for det AKTUELLA blocket, sa
// lasningen overlappar med DSP-vantan. Hoppar over block vars vikter redan ar residenta.
static void prefetch_request(StreamCtx & sc, const std::string & pfx, bool mod) {
    if (!ZI_PREFETCH || !sc.pf.f) return;
    if (ZI_RESIDENT && sc.wcache.count(pfx)) return;   // redan pa DSP:n, disken behovs ej
    for (int k = 0; k < ZI_RING_SLOTS; k++) if (sc.ring_owner[k] == pfx) return;
    {
        std::lock_guard<std::mutex> lk(sc.pf.mu);
        if (sc.pf.have == pfx || sc.pf.want == pfx) return;
        sc.pf.want = pfx; sc.pf.want_mod = mod;
    }
    sc.pf.cv_req.notify_one();
}


// ---- packcachen (ZI_PACKCACHE) ---------------------------------------------------------------
// Fil: 64 B huvud {magic "PXRP0001", tagg[32], gguf-storlek u64, gguf-mtime u64, n u32, 4 B},
// sedan n indexposter a 152 B {namn[96], typ u32, 0 u32, ne[4] i64, offset u64, bytes u64}, sedan
// data (4096-justerad per tensor). En post = en blocktensor exakt som den ligger i en hexagon-
// WEIGHTS-buffert efter set_tensor (tegelpackad for Q4_0, ra for f32).
struct PackHdr { char magic[8]; char tag[32]; uint64_t gsize, gmtime; uint32_t n, pad; };
struct PackEnt { char name[96]; uint32_t type, pad; int64_t ne[4]; uint64_t off, size; };
static_assert(sizeof(PackHdr) == 64 && sizeof(PackEnt) == 152, "packcache layout");

static bool is_block_tensor(const char * nm) {
    return !strncmp(nm, "layers.", 7) || !strncmp(nm, "noise_refiner.", 14) || !strncmp(nm, "context_refiner.", 16);
}

static void pack_cache_open(StreamCtx & sc, const std::string & gguf_path, const std::string & path) {
    const int64_t t0 = ggml_time_us();
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(sc.backend));
    auto tag_fn = (const char * (*)(void)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_hexagon_pack_tag");
    sc.pk.get_packed = (bool (*)(const ggml_tensor *, void *, size_t)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_hexagon_get_tensor_packed");
    sc.pk.set_packed = (bool (*)(ggml_tensor *, const void *, size_t)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_hexagon_set_tensor_packed");
    if (!tag_fn || !sc.pk.get_packed || !sc.pk.set_packed) {
        fprintf(stderr, "[pack] backenden saknar packad-API - packcachen av\n");
        return;
    }
    struct _stat64 st;
    if (_stat64(gguf_path.c_str(), &st) != 0) return;
    PackHdr want{};
    memcpy(want.magic, "PXRP0001", 8);
    strncpy(want.tag, tag_fn(), sizeof(want.tag) - 1);
    want.gsize = (uint64_t) st.st_size;
    want.gmtime = (uint64_t) st.st_mtime;

    // alla blocktensorer, i GGUF-ordning, med sin packade storlek
    ggml_backend_buffer_type_t bt = ggml_backend_get_default_buffer_type(sc.backend);
    std::vector<int64_t> ids;
    std::vector<PackEnt> ents;
    {
        ggml_init_params p = { ggml_tensor_overhead() * 2, nullptr, true };
        const int64_t n = gguf_get_n_tensors(sc.gctx);
        for (int64_t id = 0; id < n; id++) {
            const char * nm = gguf_get_tensor_name(sc.gctx, id);
            if (!is_block_tensor(nm)) continue;
            if (strlen(nm) >= sizeof(PackEnt::name)) { fprintf(stderr, "[pack] for langt namn %s\n", nm); return; }
            ggml_context * c = ggml_init(p);
            ggml_tensor * t = ggml_new_tensor(c, gguf_get_tensor_type(sc.gctx, id), GGML_MAX_DIMS, gguf_get_tensor_ne(sc.gctx, id));
            PackEnt e{};
            strcpy(e.name, nm);
            e.type = (uint32_t) t->type;
            for (int k = 0; k < 4; k++) e.ne[k] = t->ne[k];
            e.size = ggml_backend_buft_get_alloc_size(bt, t);
            ggml_free(c);
            ids.push_back(id);
            ents.push_back(e);
        }
    }
    const uint64_t data0 = GGML_PAD(sizeof(PackHdr) + ents.size() * sizeof(PackEnt), 4096);
    {
        uint64_t off = data0;
        for (auto & e : ents) { e.off = off; off += GGML_PAD(e.size, 4096); }
    }

    // finns en giltig fil? (samma nyckel, samma poster)
    bool ok = false;
    if (FILE * f = fopen(path.c_str(), "rb")) {
        PackHdr h{};
        std::vector<PackEnt> got(ents.size());
        ok = fread(&h, sizeof h, 1, f) == 1 && !memcmp(&h, &want, offsetof(PackHdr, n)) && h.n == ents.size() &&
             fread(got.data(), sizeof(PackEnt), got.size(), f) == got.size() &&
             !memcmp(got.data(), ents.data(), got.size() * sizeof(PackEnt));
        fclose(f);
        if (!ok) fprintf(stderr, "[pack] %s ar inaktuell (annan modell, fil eller packlayout) - bygger om\n", path.c_str());
    }

    if (!ok) {
        // Bygg: varje tensor packas EN gang i en ateranvand WEIGHTS-buffert och sparas som den
        // blev. Skrivs till .tmp och doptes forst nar allt ar pa disk - en avbruten byggnad
        // lamnar aldrig en halv cache som ser giltig ut.
        const std::string tmp = path + ".tmp";
        FILE * f = fopen(tmp.c_str(), "wb");
        if (!f) { fprintf(stderr, "[pack] kan inte skriva %s - packcachen av\n", tmp.c_str()); return; }
        uint64_t maxsz = 0;
        for (auto & e : ents) maxsz = e.size > maxsz ? e.size : maxsz;
        ggml_backend_buffer_t b = ggml_backend_buft_alloc_buffer(bt, (size_t) maxsz);
        if (!b) { fclose(f); remove(tmp.c_str()); fprintf(stderr, "[pack] ingen byggbuffert\n"); return; }
        ggml_backend_buffer_set_usage(b, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        PackHdr h = want;
        h.n = (uint32_t) ents.size();
        fwrite(&h, sizeof h, 1, f);
        fwrite(ents.data(), sizeof(PackEnt), ents.size(), f);
        std::vector<uint8_t> raw, packed;
        bool good = true;
        uint64_t total = 0;
        for (size_t i = 0; i < ents.size() && good; i++) {
            ggml_init_params p = { ggml_tensor_overhead() * 2, nullptr, true };
            ggml_context * c = ggml_init(p);
            ggml_tensor * t = ggml_new_tensor(c, (ggml_type) ents[i].type, GGML_MAX_DIMS, ents[i].ne);
            ggml_tallocr ta = ggml_tallocr_new(b);
            good = ggml_tallocr_alloc(&ta, t) == GGML_STATUS_SUCCESS;
            const size_t rsz = gguf_get_tensor_size(sc.gctx, ids[i]);
            raw.resize(rsz);
            _fseeki64(sc.gguf_file, (long long) (sc.data_base + gguf_get_tensor_offset(sc.gctx, ids[i])), SEEK_SET);
            good = good && fread(raw.data(), 1, rsz, sc.gguf_file) == rsz;
            if (good) ggml_backend_tensor_set(t, raw.data(), 0, rsz);   // tegelpackningen
            packed.assign((size_t) GGML_PAD(ents[i].size, 4096), 0);
            good = good && sc.pk.get_packed(t, packed.data(), (size_t) ents[i].size);
            _fseeki64(f, (long long) ents[i].off, SEEK_SET);
            good = good && fwrite(packed.data(), 1, packed.size(), f) == packed.size();
            total += ents[i].size;
            ggml_free(c);
        }
        ggml_backend_buffer_free(b);
        good = fclose(f) == 0 && good;
        if (good) remove(path.c_str());
        if (!good || rename(tmp.c_str(), path.c_str()) != 0) {
            remove(tmp.c_str());
            fprintf(stderr, "[pack] bygget misslyckades - packcachen av\n");
            return;
        }
        fprintf(stderr, "[pack] byggd: %zu tensorer, %.0f MB, %.1f s -> %s\n", ents.size(), total / 1048576.0,
                (ggml_time_us() - t0) * 1e-6, path.c_str());
    }

    sc.pk.f = fopen(path.c_str(), "rb");
    if (!sc.pk.f) return;
    for (size_t i = 0; i < ents.size(); i++) sc.pk.ent[ids[i]] = { ents[i].off, ents[i].size };
    sc.pk.on = true;
    fprintf(stderr, "[pack] packcache pa: %zu tensorer (%.2f s)\n", ents.size(), (ggml_time_us() - t0) * 1e-6);
}

// runs one block: loads its weights from disk, computes, returns the new x (host-side).
// x_in/adaln/ids are host arrays; S = x_in.size()/DIM.
static std::vector<float> & run_block(StreamCtx & sc, const std::string & pfx, bool mod,
                                     const std::vector<float> & x_in, int64_t S,
                                     const std::vector<int32_t> & ids0, const std::vector<int32_t> & ids1, const std::vector<int32_t> & ids2,
                                     const std::vector<float> & adaln, bool want_stage_dbg_in = false,
                                     const std::string & next_pfx = std::string(), bool next_mod = false,
                                     bool in_dev = false, bool out_host = true) {
    const bool want_stage_dbg = want_stage_dbg_in && ZI_STATS;
    const int64_t t_blk0 = ggml_time_us();

    // Keep this block's weights on the device across steps. Measured on a 1024px 4 step
    // Q4_0 image: weight loading drops from 49.1 s to 3.0 s, compute rises 4 s, so the
    // image goes from 140 s to 100 s. Set ZI_RESIDENT=0 to reload per call instead.
    static const int ZI_F16_IO = []{ const char * e = getenv("ZI_F16_IO"); return e ? atoi(e) : 1; }();
    // Residualstrommen stannar pa NPU:n mellan blocken, se StreamCtx::act_buf. Da ar ZI_F16_IO
    // meningslos (ingen transport kvar) och x ar f32 hela vagen. ZI_ACT_RESIDENT=0 = gamla vagen.
    static const int ZI_ACT_RESIDENT = []{ const char * e = getenv("ZI_ACT_RESIDENT"); return e ? atoi(e) : 1; }();
    const bool res = ZI_ACT_RESIDENT != 0;
    if (!res) { in_dev = false; out_host = true; }
    if (in_dev && (sc.act_cur < 0 || sc.act_S != S)) {
        fprintf(stderr, "%s: in_dev utan giltig plats (cur %d, S %lld mot %lld)\n", pfx.c_str(), sc.act_cur, (long long) sc.act_S, (long long) S);
        exit(1);
    }
    const bool f16_io = ZI_F16_IO && !res;

    size_t n_tensors = 200;
    ggml_init_params cparams = { ggml_tensor_overhead() * n_tensors + ggml_graph_overhead_custom(256, false), nullptr, true };
    ggml_context * ctx = ggml_init(cparams);

    BlockWeights w{};
    BlockCache * cached = nullptr;
    if (ZI_RESIDENT) {
        auto it = sc.wcache.find(pfx);
        if (it != sc.wcache.end()) { w = it->second.w; cached = &it->second; }
    }

    // Buffertringen: blocket fick nej av budgeten -> en av ZI_RING_SLOTS platser. Ligger det
    // redan kvar i en plats (fa strommade block) ateranvands den utan uppladdning.
    int  ring_slot = -1;
    bool ring_hit  = false;
    if (!cached && ZI_RESIDENT && ring_wants(sc, pfx, mod)) {
        for (int k = 0; k < ZI_RING_SLOTS; k++) {
            if (sc.ring_owner[k] == pfx) { ring_slot = k; ring_hit = true; break; }
        }
        if (ring_hit) {
            w = sc.ring_w[ring_slot];
            g_times.r_hits++;
        } else {
            ring_slot = sc.ring_next;
            sc.ring_next = (sc.ring_next + 1) % ZI_RING_SLOTS;
            // Platsen far inte skrivas forran blocket som senast last den ar klart pa DSP:n.
            const int64_t t_rw = ggml_time_us();
            if (sc.ring_rec[ring_slot]) ggml_backend_event_synchronize(sc.ring_ev[ring_slot]);
            g_times.r_wait += ggml_time_us() - t_rw;
            if (sc.ring_ctx[ring_slot]) { ggml_free(sc.ring_ctx[ring_slot]); sc.ring_ctx[ring_slot] = nullptr; }
            sc.ring_owner[ring_slot].clear();
            g_times.r_fills++;
        }
    }
    const bool ring = ring_slot >= 0;

    // Weights go in their own context when resident, so the per-call context can be freed
    // without dropping them. Non-resident they share the per-call context as before.
    ggml_context * ctx_w = nullptr;
    if (!cached && !ring_hit) {
        if (ZI_RESIDENT) {
            ggml_init_params wparams = { ggml_tensor_overhead() * 32, nullptr, true };
            ctx_w = ggml_init(wparams);
        } else {
            ctx_w = ctx;
        }
    }

    std::vector<ggml_tensor *> to_upload;
    std::vector<int64_t> to_upload_id;
    auto get = [&](const std::string & name) -> ggml_tensor * {
        int64_t id = gguf_find_tensor(sc.gctx, name.c_str());
        if (id < 0) { fprintf(stderr, "missing tensor %s\n", name.c_str()); exit(1); }
        const int64_t * ne = gguf_get_tensor_ne(sc.gctx, id);
        ggml_type type = gguf_get_tensor_type(sc.gctx, id);
        ggml_tensor * t = ggml_new_tensor(ctx_w, type, GGML_MAX_DIMS, ne);
        ggml_set_name(t, name.c_str());
        to_upload.push_back(t);
        to_upload_id.push_back(id);
        return t;
    };

    if (!cached && !ring_hit) {
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
    }   // !cached

    ggml_tensor * x_t = ggml_new_tensor_2d(ctx, f16_io ? GGML_TYPE_F16 : GGML_TYPE_F32, DIM, S);
    ggml_tensor * i0 = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, S);
    ggml_tensor * i1 = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, S);
    ggml_tensor * i2 = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, S);
    // MSECT/mrope kraver EN positions-tensor med fyra staplade arrayer (4*S).
    ggml_tensor * i_all = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4 * S);
    // freq_factors: mrope har EN global theta_scale = theta^(-2/n_dims), men de tre
    // separata ropesen har var sin (theta^(-2/AXES[a])). ff dividerar theta per sektor
    // och kompenserar exakt: ff[s] = theta^(2*(s-s0)*(1/d_a - 1/HD)).
    ggml_tensor * ff = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, HD / 2);
    ggml_tensor * adaln_t = mod ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 256) : nullptr;
    ggml_set_input(x_t); ggml_set_input(i0); ggml_set_input(i1); ggml_set_input(i2); ggml_set_input(i_all); ggml_set_input(ff);
    if (mod) ggml_set_input(adaln_t);

    // Resident weights get their own buffer, allocated once. WEIGHTS usage triggers the
    // ggml-hexagon tiled repack in set-tensor, so this way we repack once, not per step.
    ggml_backend_buffer_t buf_w = nullptr;
    if (ring && !ring_hit) {
        // Platsen allokeras EN gang (storsta blocket) och mappas vid forsta bruk; sedan placeras
        // blockets tensorer linjart i den. Ingen ny DSP-mappning per block.
        const size_t need = block_alloc_bytes(sc, "layers.0", true);
        if (!sc.ring_buf[ring_slot] || sc.ring_cap < need) {
            if (sc.ring_buf[ring_slot]) { g_mem_ring -= ggml_backend_buffer_get_size(sc.ring_buf[ring_slot]); ggml_backend_buffer_free(sc.ring_buf[ring_slot]); }
            sc.ring_buf[ring_slot] = ggml_backend_buft_alloc_buffer(ggml_backend_get_default_buffer_type(sc.backend), need);
            if (!sc.ring_buf[ring_slot]) { fprintf(stderr, "failed to allocate ring slot %d (%zu B)\n", ring_slot, need); exit(1); }
            ggml_backend_buffer_set_usage(sc.ring_buf[ring_slot], GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            g_mem_ring += ggml_backend_buffer_get_size(sc.ring_buf[ring_slot]);
            sc.ring_cap = need;
            if (!sc.ring_ev[ring_slot]) sc.ring_ev[ring_slot] = ggml_backend_event_new(ggml_backend_get_device(sc.backend));
            if (!sc.ring_ev[ring_slot]) { fprintf(stderr, "ring: backenden saknar event\n"); exit(1); }
        }
        ggml_tallocr ta = ggml_tallocr_new(sc.ring_buf[ring_slot]);
        for (ggml_tensor * t : to_upload) {
            if (ggml_tallocr_alloc(&ta, t) != GGML_STATUS_SUCCESS) { fprintf(stderr, "ring: %s ryms inte i platsen\n", t->name); exit(1); }
        }
    } else if (!cached && ZI_RESIDENT) {
        buf_w = ggml_backend_alloc_ctx_tensors(ctx_w, sc.backend);
        if (!buf_w) { fprintf(stderr, "failed to allocate weight buffer for %s\n", pfx.c_str()); exit(1); }
        ggml_backend_buffer_set_usage(buf_w, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        g_mem_w += ggml_backend_buffer_get_size(buf_w);
    }

    if (res) {
        const size_t need = GGML_PAD((size_t) DIM * S * sizeof(float), 4096);
        if (!in_dev && need > sc.act_slot) {          // bara nar varden laddar upp: inget att bevara
            if (sc.act_buf) ggml_backend_buffer_free(sc.act_buf);
            sc.act_buf = ggml_backend_buft_alloc_buffer(ggml_backend_get_default_buffer_type(sc.backend), 2 * need);
            if (!sc.act_buf) { fprintf(stderr, "failed to allocate activation buffer (%zu B)\n", 2 * need); exit(1); }
            g_mem_act = ggml_backend_buffer_get_size(sc.act_buf);
            sc.act_slot = need;
        }
        if (!in_dev) sc.act_cur = 0;                  // varden skriver plats 0 nedan
        char * base = (char *) ggml_backend_buffer_get_base(sc.act_buf);
        if (ggml_backend_tensor_alloc(sc.act_buf, x_t, base + (size_t) sc.act_cur * sc.act_slot) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "%s: kunde inte binda x_t till aktiveringsplatsen\n", pfx.c_str()); exit(1);
        }
    }
    // Med residenta aktiveringar OCH vikter binds aven de sma indatatensorerna till en persistent
    // buffert: ingen ny fd/DSP-mappning per block (64 platser, 130 block per bild).
    const bool io_persist = res && ZI_RESIDENT;
    if (io_persist) {
        ggml_tensor * small[6] = { i0, i1, i2, i_all, ff, adaln_t };
        ggml_backend_buffer_type_t bt = ggml_backend_get_default_buffer_type(sc.backend);
        const size_t al = ggml_backend_buft_get_alignment(bt);
        size_t off[6];
        size_t need = 0;
        for (int k = 0; k < 6; k++) {
            off[k] = need;
            if (small[k]) need += GGML_PAD(ggml_nbytes(small[k]), al);
        }
        if (need > sc.io_cap) {
            if (sc.io_buf) ggml_backend_buffer_free(sc.io_buf);
            sc.io_buf = ggml_backend_buft_alloc_buffer(bt, need);
            if (!sc.io_buf) { fprintf(stderr, "failed to allocate io buffer (%zu B)\n", need); exit(1); }
            g_mem_io = ggml_backend_buffer_get_size(sc.io_buf);
            sc.io_cap = need;
        }
        char * base = (char *) ggml_backend_buffer_get_base(sc.io_buf);
        for (int k = 0; k < 6; k++) {
            if (small[k] && ggml_backend_tensor_alloc(sc.io_buf, small[k], base + off[k]) != GGML_STATUS_SUCCESS) {
                fprintf(stderr, "%s: kunde inte binda indatatensor %d\n", pfx.c_str(), k); exit(1);
            }
        }
    }
    ggml_backend_buffer_t buf = nullptr;
    if (!io_persist) {
        buf = ggml_backend_alloc_ctx_tensors(ctx, sc.backend);
        if (!buf) { fprintf(stderr, "failed to allocate block buffer for %s\n", pfx.c_str()); exit(1); }
        if (!ZI_RESIDENT) {
            ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        }
    }

    g_times.g_pre += ggml_time_us() - t_blk0;
    int64_t t_w0 = ggml_time_us();
    {
        // Vantar in prefetch-traden om den hunnit bli ombedd att lasa just DET HAR blocket.
        // Ar den klar kostar vantan noll -- och da ar `disk` redan betald bakom forra
        // blockets compute. Har den inte ens borjat faller vi tillbaka pa fread nedan.
        std::map<int64_t, std::vector<uint8_t>> * pfbuf = nullptr;
        if (ZI_PREFETCH && sc.pf.f && !to_upload.empty()) {
            const int64_t t_p0 = ggml_time_us();
            std::unique_lock<std::mutex> lk(sc.pf.mu);
            // ⛔ FORSTA VERSIONEN VANTADE PA FEL BLOCK. Villkoret `sc.pf.busy` ar sant
            // ocksa nar traden laser N+1, sa block N blockerade pa N+1:s lasning innan
            // det ens hade laddat sig sjalvt: `vanta 2,39 s` mot `disk 2,05` utan prefetch.
            // Vanta BARA om det ar just DET HAR blocket som ar i flykten eller i kon.
            if (sc.pf.have == pfx) {
                pfbuf = &sc.pf.buf;
            } else if (sc.pf.inflight == pfx || sc.pf.want == pfx) {
                sc.pf.cv_done.wait(lk, [&]{ return sc.pf.have == pfx || (!sc.pf.busy && sc.pf.want != pfx); });
                if (sc.pf.have == pfx) pfbuf = &sc.pf.buf;
            }
            lk.unlock();
            g_times.w_wait += ggml_time_us() - t_p0;
        }

        std::vector<uint8_t> stage;
        for (size_t i = 0; i < to_upload.size(); i++) {
            size_t off, sz;
            const bool packed = weight_src(sc, to_upload_id[i], off, sz);
            FILE * src_f = packed ? sc.pk.f : sc.gguf_file;
            const int64_t t_r0 = ggml_time_us();
            if (pfbuf) {
                auto it = pfbuf->find(to_upload_id[i]);
                if (it != pfbuf->end() && it->second.size() == sz) {
                    stage.swap(it->second);   // ingen kopia, tradens buffert overtas
                } else {
                    pfbuf = nullptr;          // ofullstandig: fall tillbaka for resten
                }
            }
            if (!pfbuf) {
                stage.resize(sz);
                _fseeki64(src_f, (long long) off, SEEK_SET);
                size_t got = fread(stage.data(), 1, sz, src_f);
                if (got != sz) { fprintf(stderr, "short read for %s\n", to_upload[i]->name); exit(1); }
            }
            const int64_t t_r1 = ggml_time_us();
            // tensor_set pa en WEIGHTS-buffert triggar ggml-hexagons tegelpackning pa VARDEN.
            // Dela hinken sa vi vet om `vikter` ar disk eller repack innan vi bygger prefetch.
            if (packed) {
                // fardigpackad: bara en kopia in i NPU-bufferten (+ samma flaggor som set_tensor satter)
                if (!sc.pk.set_packed(to_upload[i], stage.data(), sz)) {
                    fprintf(stderr, "packcache: %s passar inte bufferten (%zu B)\n", to_upload[i]->name, sz); exit(1);
                }
            } else {
                ggml_backend_tensor_set(to_upload[i], stage.data(), 0, sz);
            }
            g_times.w_read += t_r1 - t_r0;
            g_times.w_set  += ggml_time_us() - t_r1;
            g_times.w_bytes += sz;
        }
    }
    if (ring && !ring_hit) {
        sc.ring_ctx[ring_slot]   = ctx_w;
        sc.ring_w[ring_slot]     = w;
        sc.ring_owner[ring_slot] = pfx;
    } else if (!cached && ZI_RESIDENT) {
        sc.n_resident++;
        BlockCache bc;
        bc.ctx = ctx_w;
        bc.buf = buf_w;
        bc.w   = w;
        sc.wcache[pfx] = bc;
    }
    int64_t t_w1 = ggml_time_us();
    g_times.weights += t_w1 - t_w0;

    // Nu ar vara egna vikter konsumerade -- forst HAR far traden borja pa nasta block.
    // Begar man tidigare (overst i anroparens varv) nollar traden bufferten som holl
    // DET HAR blocket, och varje block missar sin egen prefetch. Matt: 100% miss.
    if (!next_pfx.empty()) prefetch_request(sc, next_pfx, next_mod);

    if (in_dev) {
        // indata ligger redan i platsen sc.act_cur - forra blockets utdata
    } else if (f16_io) {
        sc.in_buf.resize(x_in.size());
        ggml_fp32_to_fp16_row(x_in.data(), sc.in_buf.data(), x_in.size());
        ggml_backend_tensor_set(x_t, sc.in_buf.data(), 0, sc.in_buf.size() * sizeof(ggml_fp16_t));
    } else {
        ggml_backend_tensor_set(x_t, x_in.data(), 0, x_in.size() * sizeof(float));
    }
    ggml_backend_tensor_set(i0, ids0.data(), 0, ids0.size() * sizeof(int32_t));
    ggml_backend_tensor_set(i1, ids1.data(), 0, ids1.size() * sizeof(int32_t));
    ggml_backend_tensor_set(i2, ids2.data(), 0, ids2.size() * sizeof(int32_t));
    {   // [ids0 | ids1 | ids2 | 0]
        std::vector<int32_t> all((size_t) 4 * S, 0);
        std::copy(ids0.begin(), ids0.end(), all.begin());
        std::copy(ids1.begin(), ids1.end(), all.begin() + S);
        std::copy(ids2.begin(), ids2.end(), all.begin() + 2 * S);
        ggml_backend_tensor_set(i_all, all.data(), 0, all.size() * sizeof(int32_t));
    }
    {   // ff[s] per sektor (par-index)
        std::vector<float> ffv((size_t) HD / 2, 1.0f);
        int s0 = 0;
        for (int a = 0; a < 3; a++) {
            const int npair = AXES[a] / 2;
            const double e   = 2.0 * (1.0 / (double) AXES[a] - 1.0 / (double) HD);
            for (int j = 0; j < npair; j++) ffv[s0 + j] = (float) pow((double) THETA, e * (double) j);
            s0 += npair;
        }
        ggml_backend_tensor_set(ff, ffv.data(), 0, ffv.size() * sizeof(float));
    }
    if (mod) ggml_backend_tensor_set(adaln_t, adaln.data(), 0, adaln.size() * sizeof(float));
    g_times.io_up += ggml_time_us() - t_w1;

    ggml_tensor * m_dbg = nullptr;
    ggml_tensor * stage_dbg[14] = {nullptr};
    ggml_tensor * x_g = f16_io ? ggml_cast(ctx, x_t, GGML_TYPE_F32) : x_t;
    // m_dbg var OVILLKORLIG: varje block byggde en extra ggml_cont-utnod, satte den som
    // output och las tillbaka den SYNKRONT fore x_out. 130 ggr/bild. Nu bakom ZI_STATS.
    std::vector<ggml_tensor *> pre_nodes;
    ggml_tensor * x_out = build_block(ctx, x_g, i0, i1, i2, i_all, ff, adaln_t, w, mod,
                                     ZI_STATS ? &m_dbg : nullptr, want_stage_dbg ? stage_dbg : nullptr, &pre_nodes);
    ggml_set_output(x_out);
    if (res) {
        if (x_out->view_src || !ggml_is_contiguous(x_out) || x_out->type != GGML_TYPE_F32 ||
            x_out->ne[0] != DIM || x_out->ne[1] != S) {
            fprintf(stderr, "%s: x_out kan inte bindas till aktiveringsplatsen\n", pfx.c_str()); exit(1);
        }
        char * base = (char *) ggml_backend_buffer_get_base(sc.act_buf);
        if (ggml_backend_tensor_alloc(sc.act_buf, x_out, base + (size_t) (1 - sc.act_cur) * sc.act_slot) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "%s: kunde inte binda x_out till aktiveringsplatsen\n", pfx.c_str()); exit(1);
        }
    }
    if (m_dbg) ggml_set_output(m_dbg);
    for (auto * t : stage_dbg) if (t) ggml_set_output(t);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 256, false);
    if (m_dbg) ggml_build_forward_expand(gf, m_dbg);
    for (auto * t : stage_dbg) if (t) ggml_build_forward_expand(gf, t);
    for (auto * t : pre_nodes) ggml_build_forward_expand(gf, t);   // FORE x_out, se ZI_ADALN_FOLD
    ggml_build_forward_expand(gf, x_out);
    if (!sc.galloc) {
        sc.galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(sc.backend));
    }
    if (!ggml_gallocr_alloc_graph(sc.galloc, gf)) { fprintf(stderr, "gallocr failed for %s\n", pfx.c_str()); exit(1); }

    int64_t t_c0 = ggml_time_us();
    ggml_status st = ggml_backend_graph_compute(sc.backend, gf);
    // Fence efter blockets ops: nasta fyllning av samma plats vantar bara pa DETTA block.
    if (ring) { ggml_backend_event_record(sc.ring_ev[ring_slot], sc.backend); sc.ring_rec[ring_slot] = true; }
    int64_t t_c1 = ggml_time_us();
    if (st != GGML_STATUS_SUCCESS) { fprintf(stderr, "compute failed for %s: %d\n", pfx.c_str(), st); exit(1); }
    // ⛔ DENNA RAD VAR "efter compute 4,1 s". 130 OBUFFRADE stderr-skrivningar per bild
    // genom PS-omslagets pipe = ~31 ms styck. Instrumentering som kostar mer an det den
    // mater. Bakom ZI_STATS nu.
    if (ZI_STATS) fprintf(stderr, "[time] %s compute %.2f ms\n", pfx.c_str(), (t_c1 - t_c0) / 1000.0);
    g_times.compute += t_c1 - t_c0;
    const int64_t t_dbg0 = ggml_time_us();

    // ⛔ HAR SATT 3,8 s. Anroparna skickar `l == 0` / `r == 0` som want_stage_dbg -- kvarglomd
    // ablationsstallning. For de blocken lastes 14 mellantensorer tillbaka till varden, varav
    // `inter` ar S x HID = 4113 x 10240 floats = 168 MB. Tre block rackte till 3,8 s.
    // Och ggml_set_output pa dem tvingar dessutom fram noder som annars hade fallit bort.
    if (want_stage_dbg) {
        const char * names[14] = {"x_after_attn", "w1o", "w3o", "inter", "w2o", "w2n",
                                   "qkv", "q_rope", "scores", "attn", "aop", "aon",
                                   "h_postnorm", "h_postgate"};
        for (int i = 0; i < 14; i++) {
            if (!stage_dbg[i]) continue;
            size_t n = ggml_nelements(stage_dbg[i]);
            std::vector<float> d(n);
            ggml_backend_tensor_get(stage_dbg[i], d.data(), 0, n * sizeof(float));
            int nnan = 0, ninf = 0; float mx = 0;
            for (float v : d) { if (v != v) nnan++; else if (!std::isfinite(v)) ninf++; else if (fabsf(v) > mx) mx = fabsf(v); }
            printf("[stage] %-14s %-12s n=%zu nan=%d inf=%d max_finite=%.4g\n", pfx.c_str(), names[i], n, nnan, ninf, mx);
            if (const char * sd = getenv("ZI_STAGE_DUMP")) {   // env = katalog att skriva till
                std::string fn = std::string(sd) + "\\stg_" + names[i] + ".f32";
                FILE * sf = fopen(fn.c_str(), "wb");
                if (sf) { fwrite(d.data(), sizeof(float), d.size(), sf); fclose(sf); }
            }
        }
    }

    if (m_dbg) {
        std::vector<float> m(4 * DIM);
        ggml_backend_tensor_get(m_dbg, m.data(), 0, m.size() * sizeof(float));
        auto stat = [&](int off, const char * nm) {
            double mn = 1e30, mx = -1e30, mean = 0;
            for (int i = 0; i < DIM; i++) { float v = m[off + i]; mean += v; if (v < mn) mn = v; if (v > mx) mx = v; }
            mean /= DIM;
            printf("[gate] %-16s %-6s min=%.4f max=%.4f mean=%.4f\n", pfx.c_str(), nm, mn, mx, mean);
        };
        stat(DIM, "gmsa_pre");  // pre-tanh, m_gmsa slice
        stat(3 * DIM, "gmlp_pre");
    }

    const int64_t t_gap0 = ggml_time_us();
    g_times.g_dbg += t_gap0 - t_dbg0;
    sc.out_buf.resize((size_t) DIM * S);   // no-op efter forsta blocket
    int64_t t_d0 = ggml_time_us();
    g_times.g_gap += t_d0 - t_gap0;
    if (res) { sc.act_cur = 1 - sc.act_cur; sc.act_S = S; }
    if (out_host) ggml_backend_tensor_get(x_out, sc.out_buf.data(), 0, sc.out_buf.size() * sizeof(float));
    g_times.io_down += ggml_time_us() - t_d0;
    g_times.calls++;
    g_times.g_post += ggml_time_us() - t_c1;

    // Upprensningen last tidigare INTE av: g_times.total togs FORE den har, sa
    // buffertfrigoring och kontextrivning lag utanfor varje hink. Nu mats de, och
    // totalen las av SIST sa den tacker hela anropet.
    const int64_t t_f0 = ggml_time_us();
    if (buf) ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    g_times.g_free += ggml_time_us() - t_f0;
    g_times.total  += ggml_time_us() - t_blk0;
    return sc.out_buf;
}

// Fasklockor for main() utanfor blockloopen. lap() bokfor tiden sedan forra lap (disjunkta
// fonster over hela main), sub() ar en varav-post inne i ett fonster.
struct PhaseClock {
    // ggml_time_us() delar med en frekvens som satts forst i ggml_time_init() (INIT_ONCE):
    // en statisk initierare hade delat med noll. start() ar forsta satsen i main.
    int64_t t0 = 0, t = 0;
    void start() { ggml_time_init(); t0 = t = ggml_time_us(); }
    std::vector<std::pair<std::string, int64_t>> acc, subs;
    static void put(std::vector<std::pair<std::string, int64_t>> & v, const char * n, int64_t us) {
        for (auto & p : v) if (p.first == n) { p.second += us; return; }
        v.emplace_back(n, us);
    }
    void lap(const char * n) { const int64_t now = ggml_time_us(); put(acc, n, now - t); t = now; }
    void sub(const char * n, int64_t us) { put(subs, n, us); }
};
static PhaseClock g_ph;

int main(int argc, char ** argv) {
    g_ph.start();
    // Flash-attention tiling. The backend cost model picks Br=256/Bc=1280 for our shape,
    // but Br=512/Bc=704 measures 4.3% faster (3672 vs 3837 ms/step at 1024px). A refit of
    // that model did not generalise - it kept choosing untested tilings that were slower -
    // so we pin the measured value here instead, where we know the shape. Set either env
    // var yourself to override, or ZI_FA_TILING=0 to leave it to the cost model.
    // 09-27: with resident K/V (no mask DMA cache) there is room for bigger Bc. Br=256/Bc=1408
    // (3 kv blocks, one softmax row vector per thread) measures 80.1 vs 84.7 ms per call
    // (-5.4%); flash err vs CPU on this shape 8.97e-5 vs 9.16e-5 (N=3). Br must be a
    // multiple of 256: 384/576/640 leave the 4 softmax threads unbalanced (+20-25%).
    {
        const char * off = getenv("ZI_FA_TILING");
        if (!(off && atoi(off) == 0)) {
            if (!getenv("GGML_HEXAGON_FA_BR")) _putenv_s("GGML_HEXAGON_FA_BR", "256");
            if (!getenv("GGML_HEXAGON_FA_BC")) _putenv_s("GGML_HEXAGON_FA_BC", "1408");
        }
    }

    std::string gguf_path = argc > 1 ? argv[1] : "D:\\prov\\2026-09-22_zimage-gguf\\z-image-turbo.gguf";
    std::string dir = argc > 2 ? argv[2] : "D:\\prov\\2026-09-22_zimage-gguf\\full_test";
    std::string dev_name = argc > 3 ? argv[3] : "CPU";
    float uni_scale = argc > 4 ? (float) atof(argv[4]) : 1.0f;
    std::string final_dev_name = argc > 5 ? argv[5] : dev_name;

    ggml_backend_load_all();
    g_ph.lap("start: backend-laddning");
    ggml_backend_dev_t dev = ggml_backend_dev_by_name(dev_name.c_str());
    if (!dev) {
        fprintf(stderr, "device '%s' not found. Available devices:\n", dev_name.c_str());
        for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
            ggml_backend_dev_t d = ggml_backend_dev_get(i);
            fprintf(stderr, "  [%zu] name=%s  desc=%s\n", i, ggml_backend_dev_name(d), ggml_backend_dev_description(d));
        }
        return 1;
    }
    StreamCtx sc{};
    sc.backend = ggml_backend_dev_init(dev, nullptr);
    printf("[stream] backend: %s\n", ggml_backend_dev_description(dev));

    ggml_backend_dev_t final_dev = ggml_backend_dev_by_name(final_dev_name.c_str());
    if (!final_dev) { fprintf(stderr, "final_layer device '%s' not found\n", final_dev_name.c_str()); return 1; }
    ggml_backend_t final_backend = ggml_backend_dev_init(final_dev, nullptr);
    printf("[stream] final_layer backend: %s\n", ggml_backend_dev_description(final_dev));
    g_ph.lap("start: 2 dev_init");

    gguf_init_params gp = { true, nullptr };
    sc.gctx = gguf_init_from_file(gguf_path.c_str(), gp);
    if (!sc.gctx) { fprintf(stderr, "failed to load %s\n", gguf_path.c_str()); return 1; }
    sc.gguf_file = fopen(gguf_path.c_str(), "rb");
    sc.data_base = gguf_get_data_offset(sc.gctx);

    if (const char * pc = getenv("ZI_PACKCACHE")) {
        if (*pc) pack_cache_open(sc, gguf_path, pc);
    }
    g_ph.lap("start: packcache");

    // Prefetch-traden far ETT EGET FILE*: delad filposition mellan tva tradar ar en
    // tystnande datatavling (_fseeki64 + fread ar inte atomiska tillsammans).
    if (ZI_PREFETCH) {
        sc.pf.f = fopen(sc.pk.on ? getenv("ZI_PACKCACHE") : gguf_path.c_str(), "rb");
        if (sc.pf.f) sc.pf.th = std::thread(prefetch_worker, &sc);
    }
    g_ph.lap("start: gguf-index + prefetch-trad");

    // Serve mode: everything above is paid once. After each image we wait on stdin for the
    // next input directory, so the backend, the GGUF index and the resident weights all
    // stay warm. First image pays the setup, the rest do not.
    static const int ZI_SERVE = []{ const char * e = getenv("ZI_SERVE"); return e ? atoi(e) : 0; }();
    bool first_image = true;

    for (;;) {
    if (!first_image) {
        fprintf(stderr, "[serve] klar. ange indata-katalog (tom rad = samma som forra) eller 'quit':\n");
        fflush(stderr);
        std::string line;
        if (!std::getline(std::cin, line)) break;
        while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
        if (line == "quit" || line == "q") break;
        if (!line.empty()) dir = line;
        g_times = StreamTimes{};
    }
    first_image = false;

    int Ht, Wt, Nimg, Scap, Su;
    { FILE * f = fopen((dir + "\\meta.txt").c_str(), "r");
      if (!f) { fprintf(stderr, "cannot open meta.txt in %s\n", dir.c_str()); if (!ZI_SERVE) return 1; continue; }
      fscanf(f, "%d %d %d %d %d", &Ht, &Wt, &Nimg, &Scap, &Su); fclose(f); }
    printf("[stream] Ht=%d Wt=%d Nimg=%d Scap=%d Su=%d dev=%s\n", Ht, Wt, Nimg, Scap, Su, dev_name.c_str());

    // x_embedder / cap_embedder are plain linears -- do them with a throwaway tiny ctx too.
    auto embed = [&](const std::string & w_name, const std::string & b_name, const std::vector<float> & in, int64_t in_dim, int64_t out_dim, int64_t S) -> std::vector<float> {
        ggml_init_params cparams = { ggml_tensor_overhead() * 32 + ggml_graph_overhead_custom(64, false), nullptr, true };
        ggml_context * ctx = ggml_init(cparams);
        int64_t wid = gguf_find_tensor(sc.gctx, w_name.c_str());
        int64_t bid = gguf_find_tensor(sc.gctx, b_name.c_str());
        ggml_tensor * w = ggml_new_tensor(ctx, gguf_get_tensor_type(sc.gctx, wid), GGML_MAX_DIMS, gguf_get_tensor_ne(sc.gctx, wid));
        ggml_tensor * b = ggml_new_tensor(ctx, gguf_get_tensor_type(sc.gctx, bid), GGML_MAX_DIMS, gguf_get_tensor_ne(sc.gctx, bid));
        ggml_tensor * x_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, in_dim, S);
        ggml_set_input(x_t);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, sc.backend);
        ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        auto up = [&](ggml_tensor * t, int64_t id) {
            size_t off = sc.data_base + gguf_get_tensor_offset(sc.gctx, id);
            size_t sz = gguf_get_tensor_size(sc.gctx, id);
            std::vector<uint8_t> stage(sz);
            _fseeki64(sc.gguf_file, (long long) off, SEEK_SET);
            fread(stage.data(), 1, sz, sc.gguf_file);
            ggml_backend_tensor_set(t, stage.data(), 0, sz);
        };
        up(w, wid); up(b, bid);
        ggml_backend_tensor_set(x_t, in.data(), 0, in.size() * sizeof(float));
        ggml_tensor * mm = ggml_mul_mat(ctx, w, x_t);
        ggml_tensor * y = ggml_add(ctx, mm, b);
        ggml_set_output(y);
        ggml_set_output(mm);
        ggml_cgraph * gf = ggml_new_graph_custom(ctx, 32, false);
        ggml_build_forward_expand(gf, mm);
        ggml_build_forward_expand(gf, y);
        ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(sc.backend));
        ggml_gallocr_alloc_graph(ga, gf);
        ggml_backend_graph_compute(sc.backend, gf);
        std::vector<float> out((size_t)(out_dim * S));
        ggml_backend_tensor_get(y, out.data(), 0, out.size() * sizeof(float));
        // Golden-dumpen fran 09-22 kostade 13,8 s per 1024-bild nar den var ovillkorlig: en extra
        // readback pa 63 MB + skrivning till SD-kortet D: vid VARJE steg. Nu bara med ZI_DUMP=1.
        static const bool dump_mm = []{ const char * e = getenv("ZI_DUMP"); return e && atoi(e) != 0; }();
        const int64_t t_dd = ggml_time_us();
        if (dump_mm && w_name.rfind("x_embedder", 0) == 0) {
            std::vector<float> mmv((size_t)(out_dim * S));
            ggml_backend_tensor_get(mm, mmv.data(), 0, mmv.size() * sizeof(float));
            FILE * f = fopen("D:\\prov\\2026-09-22_zimage-gguf\\golden_test\\got_x_mulmat_only.f32", "wb");
            fwrite(mmv.data(), sizeof(float), mmv.size(), f);
            fclose(f);
        }
        g_ph.sub("x_embedder: debugdump (readback + D:)", ggml_time_us() - t_dd);
        ggml_gallocr_free(ga);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return out;
    };

    // cap_embedder.0 (RMSNorm) + .1 (linear) -- do the RMSNorm inline on host (cheap, Scap*CAPD floats)
    auto host_rmsnorm = [&](std::vector<float> & x, int64_t rows, int64_t D, const std::vector<float> & w) {
        for (int64_t r = 0; r < rows; r++) {
            float * row = x.data() + r * D;
            double ss = 0; for (int64_t d = 0; d < D; d++) ss += (double) row[d] * row[d];
            float inv = (float) (1.0 / sqrt(ss / D + EPS));
            for (int64_t d = 0; d < D; d++) row[d] = row[d] * inv * w[d];
        }
    };

    std::vector<float> cap_raw = read_bin<float>(dir + "\\cap_raw.bin");
    std::vector<float> img_raw = read_bin<float>(dir + "\\img_raw.bin");
    // t_embedder (ggml-graf): t -> sinusoidal(256) -> mlp.0(256->1024) -> SiLU -> mlp.2(1024->256) -> adaln[256]
    // Lambda sa att residens-laget (ZI_STEPS) kan rakna om den per steg utan att
    // ladda om modellen. Enheten ar sjalvstandig och kan flyttas in i biblioteket.
    auto make_adaln = [&](float tv) -> std::vector<float> {
        // t_scale ur transformer/config.json ("t_scale": 1000.0). Referensen
        // (diffusers ZImageTransformer2DModel) gor t_embedder(t * t_scale) -- utan
        // den blir alla sinusoid-argument <= 1 => embeddingen nastan konstant och
        // modellen far ingen brusnivainformation.
        const float T_SCALE = 1000.0f;
        const float tvs = tv * T_SCALE;
        std::vector<float> tf(256);
        const int half = 128;
        for (int i = 0; i < half; i++) {
            float fr = expf(-logf(10000.0f) * (float) i / (float) half);
            float a = tvs * fr;
            tf[i] = cosf(a);
            tf[half + i] = sinf(a);
        }
        std::vector<float> hmid = embed("t_embedder.mlp.0.weight", "t_embedder.mlp.0.bias", tf, 256, 1024, 1);
        for (auto & v : hmid) v = v / (1.0f + expf(-v));  // SiLU
        std::vector<float> a256 = embed("t_embedder.mlp.2.weight", "t_embedder.mlp.2.bias", hmid, 1024, 256, 1);
        printf("[stream] t_embedder t=%.4f -> adaln[256] (ggml)\n", tv);
        return a256;
    };
    std::vector<int32_t> cap_ids0 = read_bin<int32_t>(dir + "\\cap_ids0.bin");
    std::vector<int32_t> cap_ids1 = read_bin<int32_t>(dir + "\\cap_ids1.bin");
    std::vector<int32_t> cap_ids2 = read_bin<int32_t>(dir + "\\cap_ids2.bin");
    std::vector<int32_t> img_ids0 = read_bin<int32_t>(dir + "\\img_ids0.bin");
    std::vector<int32_t> img_ids1 = read_bin<int32_t>(dir + "\\img_ids1.bin");
    std::vector<int32_t> img_ids2 = read_bin<int32_t>(dir + "\\img_ids2.bin");

    {
        std::vector<float> cap0_w_data;
        int64_t id = gguf_find_tensor(sc.gctx, "cap_embedder.0.weight");
        size_t off = sc.data_base + gguf_get_tensor_offset(sc.gctx, id);
        size_t sz = gguf_get_tensor_size(sc.gctx, id);
        cap0_w_data.resize(sz / sizeof(float));
        _fseeki64(sc.gguf_file, (long long) off, SEEK_SET);
        fread(cap0_w_data.data(), 1, sz, sc.gguf_file);
        host_rmsnorm(cap_raw, Scap, CAPD, cap0_w_data);
    }
    // ZI_DUMP=1 slar pa mellansteg-dumparna. AV som forval: vid 1024px ar varje
    // lager-dump Su*DIM*4 = 63 MB och det ar 30 lager => ~1,9 GB skrivet PER STEG
    // (~15 GB per 8-stegs-korning) rakt ner pa D:-SD-kortet. Det ar en stor del av
    // disklasten; utdatan (got_final_out.f32) skrivs alltid.
    static const int ZI_DUMP = []{ const char * e = getenv("ZI_DUMP"); return e ? atoi(e) : 0; }();
    auto dump = [&](const std::string & fname, const std::vector<float> & v) {
        if (!ZI_DUMP) return;
        FILE * f = fopen((dir + "\\" + fname).c_str(), "wb");
        fwrite(v.data(), sizeof(float), v.size(), f);
        fclose(f);
    };

    g_ph.lap("bild: indata + schema");
    std::vector<float> cap = embed("cap_embedder.1.weight", "cap_embedder.1.bias", cap_raw, CAPD, DIM, Scap);
    dump("got_cap_after_embed.f32", cap);
    g_ph.lap("bild: cap_embedder");
    for (int r = 0; r < NR; r++) {
        cap.swap(run_block(sc, "context_refiner." + std::to_string(r), false, cap, Scap, cap_ids0, cap_ids1, cap_ids2, {},
                           false, r + 1 < NR ? "context_refiner." + std::to_string(r + 1) : std::string(), false,
                           r > 0, r + 1 == NR || ZI_STATS || ZI_DUMP));
    }
    printf("[stream] cap done\n");
    g_ph.lap("bild: context_refiner (block)");
    dump("got_cap_after_cr.f32", cap);

    // ---------------- RESIDENS-LAGE ----------------
    // ZI_STEPS=N kor HELA scheduler-loopen internt i EN process: modellen/sessionen
    // oppnas en gang och sidcachen forblir varm mellan stegen (mot tidigare: en ny
    // process per steg som stromade om 6,1 GB fran D:-SD-kortet varje gang).
    // Cap-vagen ovan ar redan hissad ur loopen -- context_refiner kors med mod=false
    // och ar darmed t-oberoende. ZI_STEPS=0 (forval) = oforandrat legacy-beteende.
    // Z-Image-Turbo is distilled for about 4 steps. 8 steps cost twice the compute for no
    // visible gain: a 1024px 4 step image is just as sharp, only a different sample.
    // ZI_STEPS=0 still selects the legacy single forward that writes the stage files;
    // callers that want it (run_scheduler.py, velocity_truth_test.py) now ask for it.
    static const int ZI_STEPS = []{ const char * e = getenv("ZI_STEPS"); return e ? atoi(e) : 4; }();
    const int n_steps = ZI_STEPS > 0 ? ZI_STEPS : 1;
    const int LH = Ht * PATCH;
    const int LW = Wt * PATCH;                 // 09-28: icke-kvadratiskt (t.ex. 848x480)

    std::vector<float> sig;                    // sigma-schema, N+1 varden
    std::vector<float> xlat;                   // rå latent [INCH,LH,LW] mellan stegen
    if (ZI_STEPS > 0) {
        // ZI_SIGMA=1 (FORVAL sedan 2026-09-27): gamla motorns schema, t = 1 - i/N.
        //   Ger [1 .9 .75 .5 0]: sista steget gor RIKTIGT arbete.
        // ZI_SIGMA=0: linspace(1, 1/NTRAIN, N) + shift, enligt HF/Qualcomm-referensen.
        //   Ger [1 .857 .601 .003 0]: sista steget ar en NO-OP, sa modellen kor i praktiken
        //   3 steg. Matt: DUBBELSVANS pa 2 av 4 seeds (forval, 1234), borta med ZI_SIGMA=1
        //   pa samma seeds med bevarad komposition. Den gamla kommentaren nedan "vid 1024
        //   vinner forvalet" stammer INTE for anatomi.
        // ⚠ cascade2.py raknar samma schema i Python och MASTE folja samma ZI_SIGMA.
        //   Skillnaden ar stor i sista steget: forvalet ger sigma [1 .857 .601 .003 0] dar
        //   sista steget bara ar en polering, gamla ger [1 .9 .75 .5 0] med ett riktigt
        //   sista steg. Vid 1024 vinner forvalet; vid 256 ar det oprovat.
        static const int ZI_SIGMA = []{ const char * e = getenv("ZI_SIGMA"); return e ? atoi(e) : 1; }();
        // ZI_SHIFT=auto: skala shift med sekvenslangden, som SD3/Flux. Ett FAST shift ar
        // intrimmat for EN upplosning - vid 1024 (Su=4109) stammer 3,0, men vid 256 (Su=269)
        // underavbrusar modellen: latentens std blev 0,679 / 0,809 / 1,074 for 256/512/1024
        // dar ~1,05 ar ratt. mu interpoleras linjart i sekvenslangd, shift = exp(mu).
        const double NTRAIN = 1000.0;
        double SHIFT = 3.0;
        {
            const char * e = getenv("ZI_SHIFT");
            if (e && strcmp(e, "auto") == 0) {
                const double BASE_SEQ = 256.0, MAX_SEQ = 4096.0, BASE_MU = 0.5, MAX_MU = 1.15;
                double mu = BASE_MU + (MAX_MU - BASE_MU) * ((double) Su - BASE_SEQ) / (MAX_SEQ - BASE_SEQ);
                SHIFT = exp(mu);
                printf("[shift] Su=%d -> mu=%.3f -> shift=%.3f (fast var 3,000)\n", Su, mu, SHIFT);
            } else if (e) {
                SHIFT = atof(e);
                printf("[shift] fast shift=%.3f\n", SHIFT);
            }
        }
        for (int i = 0; i < n_steps; i++) {
            double raw = (ZI_SIGMA == 1)
                       ? 1.0 - (double) i / (double) n_steps
                       : ((n_steps == 1) ? 1.0 : 1.0 - (double) i * (1.0 - 1.0 / NTRAIN) / (double) (n_steps - 1));
            sig.push_back((float) (SHIFT * raw / (1.0 + (SHIFT - 1.0) * raw)));
        }
        sig.push_back(0.0f);
        if (ZI_SIGMA == 1) {
            printf("[sigma] gamla motorns schema:");
            for (size_t i = 0; i < sig.size(); i++) printf(" %.3f", sig[i]);
            printf("\n");
        }
        xlat = read_bin<float>(dir + "\\lat_init.f32");
        if (xlat.size() != (size_t) INCH * LH * LW) {
            fprintf(stderr, "lat_init.f32 saknas/fel storlek (%zu, vantade %d)\n", xlat.size(), INCH * LH * LW);
            return 1;
        }
        printf("[resident] %d steg, latent %dx%d, sigma[0]=%.3f sigma[N-1]=%.4f\n", n_steps, LH, LW, sig[0], sig[n_steps - 1]);
    }

    // patchify: [INCH,LH,LW] -> [Nimg, PATCH*PATCH*INCH], inom-token (ph,pw,c)
    // (samma ordning som referensens permute(1,3,5,2,4,6,0) => (pF,pH,pW,C))
    auto patchify = [&](const std::vector<float> & lat) {
        std::vector<float> out((size_t) Nimg * PATCH_IN);
        for (int hi = 0; hi < Ht; hi++)
            for (int wi = 0; wi < Wt; wi++)
                for (int phi = 0; phi < PATCH; phi++)
                    for (int pwi = 0; pwi < PATCH; pwi++)
                        for (int c = 0; c < INCH; c++)
                            out[(size_t)(hi * Wt + wi) * PATCH_IN + (phi * PATCH + pwi) * INCH + c] =
                                lat[(size_t) c * LH * LW + (hi * PATCH + phi) * LW + (wi * PATCH + pwi)];
        return out;
    };

    std::vector<float> latent_out;             // fylls av final_layer varje steg
    // Cascade support: run only part of the schedule. The sigma table is always built
    // for the full n_steps, so a run with STEP_FROM=3 continues exactly where a run with
    // STEP_TO=3 stopped - as long as it is handed that run's latent as lat_init.
    static const int ZI_STEP_FROM = []{ const char * e = getenv("ZI_STEP_FROM"); return e ? atoi(e) : 0; }();
    static const int ZI_STEP_TO   = []{ const char * e = getenv("ZI_STEP_TO");   return e ? atoi(e) : -1; }();
    const int step_from = ZI_STEP_FROM;
    const int step_to   = (ZI_STEP_TO >= 0 && ZI_STEP_TO <= n_steps) ? ZI_STEP_TO : n_steps;
    if (ZI_STEPS > 0 && (step_from != 0 || step_to != n_steps)) {
        printf("[kaskad] kor steg %d..%d av %d (sigma %.4f -> %.4f)\n",
               step_from, step_to - 1, n_steps, sig[step_from], sig[step_to]);
    }

    for (int step = step_from; step < step_to; step++) {
    std::vector<float> adaln;
    if (ZI_STEPS > 0) {
        adaln  = make_adaln(1.0f - sig[step]);
        img_raw = patchify(xlat);
    } else {
        std::vector<float> t_in = read_bin<float>(dir + "\\t.bin");
        adaln = make_adaln(t_in.empty() ? 0.5f : t_in[0]);
    }
    if (ZI_DUMP) { FILE * fa = fopen((dir + "\\got_adaln.f32").c_str(), "wb"); fwrite(adaln.data(), sizeof(float), adaln.size(), fa); fclose(fa); }
    g_ph.lap("steg: adaln + patchify");

    std::vector<float> x = embed("x_embedder.weight", "x_embedder.bias", img_raw, PATCH_IN, DIM, Nimg);
    dump("got_x_after_embed.f32", x);
    g_ph.lap("steg: x_embedder");
    for (int r = 0; r < NR; r++) {
        x.swap(run_block(sc, "noise_refiner." + std::to_string(r), true, x, Nimg, img_ids0, img_ids1, img_ids2, adaln, r == 0,
                         r + 1 < NR ? "noise_refiner." + std::to_string(r + 1) : std::string("layers.0"), true,
                         r > 0, r + 1 == NR || ZI_STATS || ZI_DUMP));
        dump("got_x_after_nr" + std::to_string(r) + ".f32", x);
    }
    printf("[stream] x done\n");
    g_ph.lap("steg: noise_refiner (block)");
    dump("got_x_after_nr.f32", x);

    std::vector<float> uni = x;
    uni.insert(uni.end(), cap.begin(), cap.end());
    if (uni_scale != 1.0f) {
        for (auto & v : uni) v *= uni_scale;
        printf("[stream] scaled uni by %g before layers\n", uni_scale);
    }
    std::vector<int32_t> uni_ids0 = img_ids0, uni_ids1 = img_ids1, uni_ids2 = img_ids2;
    uni_ids0.insert(uni_ids0.end(), cap_ids0.begin(), cap_ids0.end());
    uni_ids1.insert(uni_ids1.end(), cap_ids1.begin(), cap_ids1.end());
    uni_ids2.insert(uni_ids2.end(), cap_ids2.begin(), cap_ids2.end());
    dump("got_uni_pre.f32", uni);

    g_ph.lap("steg: uni-sammanfogning");
    std::vector<float> uni_prev;
    for (int l = 0; l < NL; l++) {
        if (ZI_STATS) { uni_prev = uni; }   // maste tas FORE blocket
        uni.swap(run_block(sc, "layers." + std::to_string(l), true, uni, Su, uni_ids0, uni_ids1, uni_ids2, adaln, l == 0,
                           l + 1 < NL ? "layers." + std::to_string(l + 1) : std::string(), true,
                           l > 0, l + 1 == NL || ZI_STATS || ZI_DUMP));
        if (ZI_STATS) {
            double mean = 0, mx = 0; int nnan = 0;
            for (float v : uni) { if (v != v) { nnan++; continue; } mean += v; if (fabsf(v) > mx) mx = fabsf(v); }
            mean /= uni.size();
            double se = 0, sy = 0;
            for (size_t i = 0; i < uni.size(); i++) { double d = uni[i] - uni_prev[i]; se += d * d; sy += (double) uni_prev[i] * uni_prev[i]; }
            double delta_relL2 = sqrt(se / (sy > 0 ? sy : 1));
            printf("[stream] layers.%-2d  mean=%.5f max_abs=%.4f nan=%d  delta_vs_prev_relL2=%.6f\n", l, mean, mx, nnan, delta_relL2);
        }
        if (ZI_DUMP) {
            char fn[256]; snprintf(fn, sizeof(fn), "%s\\layer_%02d.bin", dir.c_str(), l);
            FILE * f = fopen(fn, "wb");
            fwrite(uni.data(), sizeof(float), uni.size(), f);
            fclose(f);
        }
    }

    if (ZI_DUMP) {
        FILE * fo = fopen((dir + "\\ggml_out.bin").c_str(), "wb");
        fwrite(uni.data(), sizeof(float), uni.size(), fo);
        fclose(fo);
        printf("[stream] wrote ggml_out.bin\n");
    }

    g_ph.lap("steg: layers (block)");
    // ---- final_layer: adaLN scale (LayerNorm, not RMSNorm) + linear -> unpatchify ----
    {
        std::vector<float> sadaln(256);
        for (int i = 0; i < 256; i++) sadaln[i] = adaln[i] / (1.0f + expf(-adaln[i])); // silu

        size_t n_tensors = 64;
        ggml_init_params cparams = { ggml_tensor_overhead() * n_tensors + ggml_graph_overhead_custom(64, false), nullptr, true };
        ggml_context * ctx = ggml_init(cparams);

        auto get_fl = [&](const std::string & name) -> ggml_tensor * {
            int64_t id = gguf_find_tensor(sc.gctx, name.c_str());
            if (id < 0) { fprintf(stderr, "missing tensor %s\n", name.c_str()); exit(1); }
            ggml_tensor * t = ggml_new_tensor(ctx, gguf_get_tensor_type(sc.gctx, id), GGML_MAX_DIMS, gguf_get_tensor_ne(sc.gctx, id));
            ggml_set_name(t, name.c_str());
            return t;
        };
        ggml_tensor * ada_w = get_fl("final_layer.adaLN_modulation.1.weight");
        ggml_tensor * ada_b = get_fl("final_layer.adaLN_modulation.1.bias");
        ggml_tensor * lin_w = get_fl("final_layer.linear.weight");
        ggml_tensor * lin_b = get_fl("final_layer.linear.bias");

        ggml_tensor * sadaln_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 256);
        ggml_tensor * uni_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, DIM, Su);
        ggml_set_input(sadaln_t); ggml_set_input(uni_t);

        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, final_backend);
        ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        g_ph.lap("final: ctx + buffert");
        auto up_fl = [&](ggml_tensor * t, const std::string & name) {
            int64_t id = gguf_find_tensor(sc.gctx, name.c_str());
            size_t off = sc.data_base + gguf_get_tensor_offset(sc.gctx, id);
            size_t sz = gguf_get_tensor_size(sc.gctx, id);
            std::vector<uint8_t> stage(sz);
            _fseeki64(sc.gguf_file, (long long) off, SEEK_SET);
            fread(stage.data(), 1, sz, sc.gguf_file);
            ggml_backend_tensor_set(t, stage.data(), 0, sz);
        };
        up_fl(ada_w, "final_layer.adaLN_modulation.1.weight");
        up_fl(ada_b, "final_layer.adaLN_modulation.1.bias");
        up_fl(lin_w, "final_layer.linear.weight");
        up_fl(lin_b, "final_layer.linear.bias");
        g_ph.lap("final: vikter fran GGUF");
        ggml_backend_tensor_set(sadaln_t, sadaln.data(), 0, sadaln.size() * sizeof(float));
        ggml_backend_tensor_set(uni_t, uni.data(), 0, uni.size() * sizeof(float));
        g_ph.lap("final: indata upp");

        ggml_tensor * scale = ggml_add(ctx, ggml_mul_mat(ctx, ada_w, sadaln_t), ada_b);
        scale = ggml_scale_bias(ctx, scale, 1.0f, 1.0f);
        ggml_tensor * xf = ggml_mul(ctx, ggml_norm(ctx, uni_t, 1e-6f), scale);
        ggml_tensor * outtok = ggml_add(ctx, ggml_mul_mat(ctx, lin_w, xf), lin_b); // [64, Su]
        ggml_set_output(outtok);

        ggml_cgraph * gf = ggml_new_graph_custom(ctx, 64, false);
        ggml_build_forward_expand(gf, outtok);
        ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(final_backend));
        ggml_gallocr_alloc_graph(ga, gf);
        ggml_backend_graph_compute(final_backend, gf);
        g_ph.lap("final: gallocr + compute");

        std::vector<float> outtok_h((size_t) 64 * Su);
        ggml_backend_tensor_get(outtok, outtok_h.data(), 0, outtok_h.size() * sizeof(float));
        ggml_gallocr_free(ga);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);

        g_ph.lap("final: ut + fri");
        // unpatchify: outs[c, hi*PATCH+phi, wi*PATCH+pwi] = outtok[(hi*Wt+wi), (phi*PATCH+pwi)*INCH+c]
        int Himg = Ht * PATCH, Wimg = Wt * PATCH;
        latent_out.assign((size_t) INCH * Himg * Wimg, 0.0f);
        for (int hi = 0; hi < Ht; hi++)
            for (int wi = 0; wi < Wt; wi++)
                for (int phi = 0; phi < PATCH; phi++)
                    for (int pwi = 0; pwi < PATCH; pwi++)
                        for (int c = 0; c < INCH; c++) {
                            size_t tok = (size_t) (hi * Wt + wi);
                            float v = outtok_h[tok * 64 + (phi * PATCH + pwi) * INCH + c];
                            latent_out[(size_t) c * Himg * Wimg + (hi * PATCH + phi) * Wimg + (wi * PATCH + pwi)] = v;
                        }

        if (ZI_STEPS == 0) {                   // legacy: en forward -> fil
            FILE * ffl = fopen((dir + "\\got_final_out.f32").c_str(), "wb");
            fwrite(latent_out.data(), sizeof(float), latent_out.size(), ffl);
            fclose(ffl);
        }
        double mean = 0, mx = 0; int nnan = 0;
        for (float v : latent_out) { if (v != v) { nnan++; continue; } mean += v; if (fabsf(v) > mx) mx = fabsf(v); }
        mean /= latent_out.size();
        printf("[stream] final_layer+unpatchify: n=%zu mean=%.5f max_abs=%.4f nan=%d\n",
               latent_out.size(), mean, mx, nnan);
    }

    g_ph.lap("steg: unpatchify + statistik");
    // FlowMatchEuler-steg: x += (sigma[i+1]-sigma[i]) * (-dit).  Samma matte som
    // Qualcomms referens (x += dt*model_output); var model_output = -dit.
    if (ZI_STEPS > 0) {
        const float ds = sig[step + 1] - sig[step];
        double s2 = 0;
        for (size_t i = 0; i < xlat.size(); i++) { xlat[i] += ds * (-latent_out[i]); s2 += (double) xlat[i] * xlat[i]; }
        printf("[resident] steg %d/%d  t_norm=%.3f  d_sigma=%+.4f  |x|=%.1f std=%.3f\n",
               step + 1, n_steps, 1.0f - sig[step], ds, sqrt(s2), sqrt(s2 / xlat.size()));
    }
    g_ph.lap("steg: schemauppdatering");
    }  // ---- slut pa residens-loopen ----

    if (ZI_STEPS > 0) {
        FILE * fl = fopen((dir + "\\lat_out.f32").c_str(), "wb");
        fwrite(xlat.data(), sizeof(float), xlat.size(), fl);
        fclose(fl);
        printf("[resident] klar -> lat_out.f32\n");
        // zimage.py vantar pa just den har raden och kor VAE:n medan vi river ned (~1 s
        // residenta vikter + DSP-sessionen). stdout ar ett ror = fullbuffrat: utan flush
        // kommer raden forst vid exit och overlappet blir noll.
        fflush(stdout);
    }

    g_ph.lap("skriv lat_out");
    if (ZI_WEIGHT_BUDGET_MB >= 0) {
        fprintf(stderr, "[budget] ring: tak %lld MB | %d block residenta (%.0f MB) | %d platser (%.0f MB) | %lld fyllningar, %lld traffar | fence-vanta %.2f s\n",
                ZI_WEIGHT_BUDGET_MB, sc.n_resident, g_mem_w / 1048576.0, ZI_RING_SLOTS, g_mem_ring / 1048576.0,
                (long long) g_times.r_fills, (long long) g_times.r_hits, g_times.r_wait * 1e-6);
    }
    fprintf(stderr, "[budget] minne: vikter %.0f MB | graf (gallocr) %.0f MB | aktivering %.0f MB | io %.0f MB\n",
            (g_mem_w + g_mem_ring) / 1048576.0, sc.galloc ? ggml_gallocr_get_buffer_size(sc.galloc, 0) / 1048576.0 : 0.0,
            g_mem_act / 1048576.0, g_mem_io / 1048576.0);
    {
        const double k = 1e-6;
        const int64_t sum = g_times.weights + g_times.io_up + g_times.compute + g_times.io_down;
        const int64_t other = g_times.total - sum;
        fprintf(stderr, "[budget] %d block | vikter %.1f | in %.1f | compute %.1f | ut %.1f | graf+ovrigt %.1f | block totalt %.1f s\n",
                g_times.calls, g_times.weights * k, g_times.io_up * k,
                g_times.compute * k, g_times.io_down * k, other * k, g_times.total * k);

        fprintf(stderr, "[budget] vikter: vanta %.2f s | disk %.2f s (%.2f GB, %.0f MB/s) | repack %.2f s\n",
                g_times.w_wait * k,
                g_times.w_read * k, g_times.w_bytes / 1e9,
                g_times.w_read ? (g_times.w_bytes / 1e6) / (g_times.w_read * k) : 0.0,
                g_times.w_set * k);
        fprintf(stderr, "[budget] varav: fore %.1f | efter compute %.1f (dbg %.1f) | upprensning %.1f | oforklarat %.1f s\n",
                g_times.g_pre * k, g_times.g_post * k, g_times.g_dbg * k, g_times.g_free * k,
                (other - g_times.g_pre - g_times.g_post - g_times.g_free) * k);
    }

    g_ph.lap("budgetutskrift");
    if (!ZI_SERVE) break;
    // Serverlaget: slapp allt som inte ar residenta vikter medan vi vantar pa nasta bild - dar laddas
    // text-encodern pa NPU:n, och den ska inte ligga ovanpa oss. Allt allokeras latt igen.
    static const int ZI_SERVE_RELEASE = []{ const char * e = getenv("ZI_SERVE_RELEASE"); return e ? atoi(e) : 1; }();
    if (ZI_SERVE_RELEASE) {
        ggml_backend_synchronize(sc.backend);
        for (int k = 0; k < 4; k++) {
            if (sc.ring_buf[k]) { ggml_backend_buffer_free(sc.ring_buf[k]); sc.ring_buf[k] = nullptr; }
            if (sc.ring_ctx[k]) { ggml_free(sc.ring_ctx[k]); sc.ring_ctx[k] = nullptr; }
            sc.ring_owner[k].clear();
            sc.ring_rec[k] = false;
        }
        sc.ring_cap = 0; sc.ring_next = 0; g_mem_ring = 0;
        if (sc.galloc) { ggml_gallocr_free(sc.galloc); sc.galloc = nullptr; }
        if (sc.act_buf) { ggml_backend_buffer_free(sc.act_buf); sc.act_buf = nullptr; }
        sc.act_slot = 0; sc.act_cur = -1; sc.act_S = 0; g_mem_act = 0;
        if (sc.io_buf) { ggml_backend_buffer_free(sc.io_buf); sc.io_buf = nullptr; }
        sc.io_cap = 0; g_mem_io = 0;
        std::vector<float>().swap(sc.out_buf);
        std::vector<ggml_fp16_t>().swap(sc.in_buf);
        {
            std::lock_guard<std::mutex> lk(sc.pf.mu);
            if (sc.pf.inflight.empty()) { sc.pf.buf.clear(); sc.pf.have.clear(); }
        }
        g_ph.lap("serve: slapp buffertar");
    }
    }  // ---- slut pa serve-loopen ----

    // The resident weight buffers must go before the backend they live on.
    for (auto & kv : sc.wcache) {
        if (kv.second.buf) ggml_backend_buffer_free(kv.second.buf);
        if (kv.second.ctx) ggml_free(kv.second.ctx);
    }
    ggml_backend_synchronize(sc.backend);
    for (int k = 0; k < 4; k++) {
        if (sc.ring_ev[k]) ggml_backend_event_free(sc.ring_ev[k]);
        if (sc.ring_buf[k]) ggml_backend_buffer_free(sc.ring_buf[k]);
        if (sc.ring_ctx[k]) ggml_free(sc.ring_ctx[k]);
    }
    if (sc.galloc) ggml_gallocr_free(sc.galloc);
    if (sc.act_buf) ggml_backend_buffer_free(sc.act_buf);
    if (sc.io_buf) ggml_backend_buffer_free(sc.io_buf);
    g_ph.lap("slut: fri residenta vikter");

    // Traden maste joinas FORE fclose och backend_free: annars kan den ligga i ett fread
    // pa ett handtag vi just stangt, och processen avslutas med en levande trad.
    if (sc.pf.th.joinable()) {
        { std::lock_guard<std::mutex> lk(sc.pf.mu); sc.pf.quit = true; }
        sc.pf.cv_req.notify_all();
        sc.pf.th.join();
    }
    if (sc.pf.f) { fclose(sc.pf.f); sc.pf.f = nullptr; }
    g_ph.lap("slut: prefetch-join");

    if (sc.pk.f) fclose(sc.pk.f);
    fclose(sc.gguf_file);
    gguf_free(sc.gctx);
    ggml_backend_free(sc.backend);
    ggml_backend_free(final_backend);
    g_ph.lap("slut: gguf + backend_free");
    {
        const double k = 1e-6;
        int64_t sum = 0, blk = 0;
        for (auto & p : g_ph.acc) {
            fprintf(stderr, "[budget] fas: %-40s %6.2f s\n", p.first.c_str(), p.second * k);
            sum += p.second;
            if (p.first.find("(block)") != std::string::npos) blk += p.second;
        }
        for (auto & p : g_ph.subs) fprintf(stderr, "[budget] fas:   varav %-34s %6.2f s\n", p.first.c_str(), p.second * k);
        fprintf(stderr, "[budget] fas: main totalt %.2f s | block (run_block) %.2f | mellan blocken %.2f | utanfor blocken %.2f s\n",
                sum * k, g_times.total * k, (blk - g_times.total) * k, (sum - blk) * k);
    }
    return 0;
}
