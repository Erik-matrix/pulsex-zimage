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
    if (level >= GGML_LOG_LEVEL_ERROR) {
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

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx        = 512;
    cp.n_batch      = 512;
    cp.n_ubatch     = 512;
    cp.embeddings   = true;
    cp.pooling_type = LLAMA_POOLING_TYPE_NONE;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        printf("ERROR cannot create context\n");
        fflush(stdout);
        llama_model_free(model);
        return 1;
    }
    llama_set_embeddings_layer_inp(ctx, (uint32_t) layer, true);
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
            if (n < 0 || n > (int) cp.n_ctx) {
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
                    batch.logits[i]    = true;
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
