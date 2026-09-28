// zimage-encode: the Qwen3-4B text encoder for Z-Image on the NPU, as a SHORT-LIVED process.
//
// Why a separate, short-lived process (2026-09-28): everything the NPU can see is pinned and
// counts as committed memory. Kept alive, the encoder holds ~2.8 GB for the whole session on top
// of the image model. This tool is started when the user begins typing, loads the model (from the
// file cache, ~2.5 s, hidden behind the typing), encodes ONE prompt, frees the model and exits
// normally - so the DSP session is closed cleanly (a hard kill can leave the NPU wedged).
//
// Protocol on stdin/stdout, one line each:
//   -> "READY"                       model loaded
//   <- "<text file>\t<output file>"  the text is used verbatim (chat template included)
//   -> "DONE <n_tokens>"             cap_feats [n_tokens x n_embd] float32 written to the output
//   -> "ERROR <message>"             on failure; EOF on stdin = quit without encoding
//
// cap_feats = hidden_states[-2]: the INPUT to the last layer (layer 35 of 36), no final norm - the
// same tap as the llama-server patch (QWEN_EMBD_LAYER). llama_set/get_embeddings_layer_inp come
// from the PulseX llama patch (src/llama-ext.h).
//
//   zimage-encode <model.gguf> [device=HTP0] [layer=35]

#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

LLAMA_API void    llama_set_embeddings_layer_inp(struct llama_context * ctx, uint32_t lid, bool value);
LLAMA_API float * llama_get_embeddings_layer_inp(struct llama_context * ctx, uint32_t lid);

static void quiet_log(enum ggml_log_level level, const char * text, void *) {
    static const bool all = getenv("ZI_ENC_LOG") != nullptr;   // ZI_ENC_LOG=1: llama/ggml-loggen (buffertstorlekar)
    if (all || level >= GGML_LOG_LEVEL_ERROR) {
        fputs(text, stderr);
    }
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: zimage-encode <model.gguf> [device=HTP0] [layer=35]\n");
        return 2;
    }
    const std::string model_path = argv[1];
    const std::string device     = argc > 2 ? argv[2] : "HTP0";
    const int         layer      = argc > 3 ? atoi(argv[3]) : 35;

    llama_log_set(quiet_log, nullptr);
    ggml_backend_load_all();
    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.load_mode          = LLAMA_LOAD_MODE_MMAP;   // read straight from the file cache (== --load-mode mmap)
    ggml_backend_dev_t devs[2] = { nullptr, nullptr };
    if (device != "none" && device != "cpu") {
        devs[0] = ggml_backend_dev_by_name(device.c_str());
        if (!devs[0]) {
            printf("ERROR no device %s\n", device.c_str());
            fflush(stdout);
            return 1;
        }
        mp.devices      = devs;
        mp.n_gpu_layers = 999;
        // Utdatalagret (token_embd som lm_head) behovs aldrig - vi laser indata till lager 35, inga
        // logits. Pa NPU:n var det en kopia pa ~320 MB; pa CPU delar det filvyn med token_embd.
        static ggml_backend_buffer_type_t cpu_bt =
            ggml_backend_dev_buffer_type(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU));
        static const llama_model_tensor_buft_override ovr[] = {
            { "^(output|token_embd)\\.weight$", cpu_bt },
            { nullptr, nullptr },
        };
        if (!getenv("ZI_ENC_OUTPUT_NPU")) {
            mp.tensor_buft_overrides = ovr;
            mp.use_extra_bufts       = false;   // annars packas kopian om till CPU_REPACK: 304 MB privat = commit
        }
    } else {
        mp.n_gpu_layers    = 0;
        mp.use_extra_bufts = false;   // no repack: the weights stay in the file cache (== --no-repack)
    }

    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) {
        printf("ERROR cannot load %s\n", model_path.c_str());
        fflush(stdout);
        return 1;
    }

    // Kontexten (graf + KV pa NPU:n) i promptens storlek: 512 token kostade 302 + 72 MB, en prompt
    // ar ~20-150. Den forsta skapas medan anvandaren skriver (128, ZI_ENC_CTX); ar prompten langre
    // byggs den om i ratt storlek efter Enter.
    auto make_ctx = [&](int n) -> llama_context * {
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx        = (uint32_t) n;
        cp.n_batch      = (uint32_t) n;
        cp.n_ubatch     = (uint32_t) n;
        cp.embeddings   = true;
        cp.pooling_type = LLAMA_POOLING_TYPE_NONE;
        llama_context * c = llama_init_from_model(model, cp);
        if (c) llama_set_embeddings_layer_inp(c, (uint32_t) layer, true);
        return c;
    };
    const char * ectx = getenv("ZI_ENC_CTX");
    int ctx_n = ectx ? atoi(ectx) : 128;
    if (ctx_n < 32) ctx_n = 32;
    llama_context * ctx = make_ctx(ctx_n);
    if (!ctx) {
        printf("ERROR cannot create context\n");
        fflush(stdout);
        llama_model_free(model);
        return 1;
    }
    const int n_embd = llama_model_n_embd(model);

    printf("READY\n");
    fflush(stdout);

    int         rc = 0;
    std::string line;
    if (std::getline(std::cin, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        const size_t tab = line.find('\t');
        if (tab == std::string::npos) {
            printf("ERROR expected '<text file>\\t<output file>'\n");
            rc = 1;
        } else {
            const std::string in_path = line.substr(0, tab), out_path = line.substr(tab + 1);
            std::ifstream     f(in_path, std::ios::binary);
            std::stringstream ss;
            ss << f.rdbuf();
            const std::string text = ss.str();          // verbatim: the template ends with a newline

            const llama_vocab * vocab = llama_model_get_vocab(model);
            std::vector<llama_token> tok(text.size() + 16);
            int n = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), tok.data(), (int32_t) tok.size(),
                                   /*add_special*/ true, /*parse_special*/ true);
            if (n > ctx_n && n <= 2048) {          // langre prompt: kontexten i ratt storlek
                llama_free(ctx);
                ctx_n = (n + 63) / 64 * 64;
                ctx   = make_ctx(ctx_n);
                if (!ctx) { printf("ERROR cannot create context (%d)\n", ctx_n); fflush(stdout); llama_model_free(model); return 1; }
            }
            if (n < 0 || n > ctx_n) {
                printf("ERROR tokenize gave %d tokens\n", n);
                rc = 1;
            } else {
                tok.resize(n);
                llama_batch batch = llama_batch_init(n, 0, 1);
                for (int i = 0; i < n; i++) {
                    batch.token[i]     = tok[i];
                    batch.pos[i]       = i;
                    batch.n_seq_id[i]  = 1;
                    batch.seq_id[i][0] = 0;
                    batch.logits[i]    = i == n - 1;   // lager-35-kranen tar alla token; logits bara en
                }
                batch.n_tokens = n;
                if (llama_decode(ctx, batch) < 0) {
                    printf("ERROR decode failed\n");
                    rc = 1;
                } else {
                    const float * e = llama_get_embeddings_layer_inp(ctx, (uint32_t) layer);
                    FILE *        o = e ? fopen(out_path.c_str(), "wb") : nullptr;
                    if (!o) {
                        printf("ERROR no output (%s)\n", e ? "cannot open file" : "layer tap missing");
                        rc = 1;
                    } else {
                        fwrite(e, sizeof(float), (size_t) n * n_embd, o);
                        fclose(o);
                        printf("DONE %d\n", n);
                    }
                }
                llama_batch_free(batch);
            }
        }
        fflush(stdout);
    }

    // free everything before exit: closes the NPU session cleanly
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return rc;
}
