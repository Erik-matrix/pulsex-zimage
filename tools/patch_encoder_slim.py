# 09-28 ("Krymp encoderns spik sa den inte ligger ovanpa DiT:n"): zimage-encode pa NPU:n holl
# 2,8 GB: vikter 2274 MB (dar ingick en KOPIA av token_embd som utdatalager, ~320 MB q6_K - vi
# behover aldrig logits), graf 302 MB och KV 72 MB for 512 token (en prompt ar ~20-150).
#  1) utdatalagret pa CPU (mmap: delar filvyn med token_embd, ingen commit)
#  2) logits bara for sista token (kranen i lager 35 laser ALLA token fore utgallringen)
#  3) kontexten 128 token (ZI_ENC_CTX), byggs om i ratt storlek om prompten ar langre
#   python tools/patch_encoder_slim.py
import io

P = r"C:\PulseCore\PulseX\zimage_dit\zimage_encode.cpp"
s = io.open(P, encoding="utf-8", newline="").read()


def rep(old, new):
    global s
    if s.count(old) != 1:
        raise SystemExit("ankare %d ggr: %r" % (s.count(old), old[:70]))
    s = s.replace(old, new)


rep('''        mp.devices      = devs;
        mp.n_gpu_layers = 999;
''', '''        mp.devices      = devs;
        mp.n_gpu_layers = 999;
        // Utdatalagret (token_embd som lm_head) behovs aldrig - vi laser indata till lager 35, inga
        // logits. Pa NPU:n var det en kopia pa ~320 MB; pa CPU delar det filvyn med token_embd.
        static ggml_backend_buffer_type_t cpu_bt =
            ggml_backend_dev_buffer_type(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU));
        static const llama_model_tensor_buft_override ovr[] = {
            { "^(output|token_embd)\\\\.weight$", cpu_bt },
            { nullptr, nullptr },
        };
        if (!getenv("ZI_ENC_OUTPUT_NPU")) mp.tensor_buft_overrides = ovr;
''')

rep('''    llama_context_params cp = llama_context_default_params();
    cp.n_ctx        = 512;
    cp.n_batch      = 512;
    cp.n_ubatch     = 512;
    cp.embeddings   = true;
    cp.pooling_type = LLAMA_POOLING_TYPE_NONE;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        printf("ERROR cannot create context\\n");
        fflush(stdout);
        llama_model_free(model);
        return 1;
    }
    llama_set_embeddings_layer_inp(ctx, (uint32_t) layer, true);
''', '''    // Kontexten (graf + KV pa NPU:n) i promptens storlek: 512 token kostade 302 + 72 MB, en prompt
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
        printf("ERROR cannot create context\\n");
        fflush(stdout);
        llama_model_free(model);
        return 1;
    }
''')

rep('''            if (n < 0 || n > (int) cp.n_ctx) {
                printf("ERROR tokenize gave %d tokens\\n", n);
                rc = 1;
            } else {''', '''            if (n > ctx_n && n <= 2048) {          // langre prompt: kontexten i ratt storlek
                llama_free(ctx);
                ctx_n = (n + 63) / 64 * 64;
                ctx   = make_ctx(ctx_n);
                if (!ctx) { printf("ERROR cannot create context (%d)\\n", ctx_n); fflush(stdout); llama_model_free(model); return 1; }
            }
            if (n < 0 || n > ctx_n) {
                printf("ERROR tokenize gave %d tokens\\n", n);
                rc = 1;
            } else {''')

rep('''                    batch.logits[i]    = true;''', '''                    batch.logits[i]    = i == n - 1;   // lager-35-kranen tar alla token; logits bara en''')
io.open(P, "w", encoding="utf-8", newline="").write(s)
print("OK")
