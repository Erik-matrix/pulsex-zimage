# 09-28 ("Bygg buffertringen och mat kurvan pa Z-Image"): viktbudget + buffertring.
# ZI_WEIGHT_BUDGET_MB = tak for blockvikterna pa NPU:n (resident + ring). Block blir residenta
# i anropsordning sa lange de ryms under taket minus ringen; resten strommas genom ZI_RING_SLOTS
# forallokerade WEIGHTS-buffertar (mappas EN gang). En slot skrivs forst nar fence-eventet fran
# dess forra block har passerat - inte en full synchronize.   python tools/patch_weight_ring.py
import io

P = r"C:\PulseCore\PulseX\zimage_dit\zimage_stream.cpp"
s = io.open(P, encoding="utf-8", newline="").read()


def rep(old, new, cnt=1):
    global s
    if s.count(old) != cnt:
        raise SystemExit("ankare %d ggr: %r" % (s.count(old), old[:80]))
    s = s.replace(old, new)


# 1) inställningar + tidsposter
rep('''static const int ZI_PREFETCH = []{ const char * e = getenv("ZI_PREFETCH"); return e ? atoi(e) : 1; }();
''', '''static const int ZI_PREFETCH = []{ const char * e = getenv("ZI_PREFETCH"); return e ? atoi(e) : 1; }();
// ZI_WEIGHT_BUDGET_MB: tak for blockvikterna pa NPU:n (residenta block + ringens platser).
// Osatt/negativ = obegransat (alla block residenta, som forut). Block blir residenta i den
// ordning de kors sa lange de ryms under taket MINUS ringen; ovriga strommas varje steg genom
// ZI_RING_SLOTS forallokerade viktplatser. Grunden for Flux/video: NPU-minnet ar pinnat och
// raknas som committed, sa taket styr hela processens fotavtryck.
static const long long ZI_WEIGHT_BUDGET_MB = []{ const char * e = getenv("ZI_WEIGHT_BUDGET_MB"); return e && *e ? atoll(e) : -1LL; }();
static const int ZI_RING_SLOTS = []{ const char * e = getenv("ZI_RING_SLOTS"); int n = e ? atoi(e) : 2; return n < 1 ? 1 : (n > 4 ? 4 : n); }();
''')
rep('''    int64_t g_free  = 0;   // ggml_backend_buffer_free + ggml_free, EFTER gamla avlasningen
''', '''    int64_t g_free  = 0;   // ggml_backend_buffer_free + ggml_free, EFTER gamla avlasningen
    int64_t r_wait  = 0;   // ringen: vantan pa platsens fence (forra blocket som last den)
    int64_t r_fills = 0;   // ringen: antal platsfyllningar (strommade block)
    int64_t r_hits  = 0;   // ringen: blocket lag redan kvar i sin plats
''')

# 2) StreamCtx: ringen
rep('''    ggml_backend_buffer_t io_buf = nullptr;
    size_t  io_cap   = 0;
};
''', '''    ggml_backend_buffer_t io_buf = nullptr;
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
};
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
''')

# 3) prefetch: hoppa aven over block som ligger kvar i en ringplats
rep('''    if (ZI_RESIDENT && sc.wcache.count(pfx)) return;   // redan pa DSP:n, disken behovs ej
''', '''    if (ZI_RESIDENT && sc.wcache.count(pfx)) return;   // redan pa DSP:n, disken behovs ej
    for (int k = 0; k < ZI_RING_SLOTS; k++) if (sc.ring_owner[k] == pfx) return;
''')

# 4) run_block: ringvagen
rep('''    // Weights go in their own context when resident, so the per-call context can be freed
    // without dropping them. Non-resident they share the per-call context as before.
    ggml_context * ctx_w = nullptr;
    if (!cached) {
        if (ZI_RESIDENT) {
''', '''    // Buffertringen: blocket fick nej av budgeten -> en av ZI_RING_SLOTS platser. Ligger det
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
''')
rep('''    if (!cached) {
    w.an1 = get(pfx + ".attention_norm1.weight");''', '''    if (!cached && !ring_hit) {
    w.an1 = get(pfx + ".attention_norm1.weight");''')
rep('''    ggml_backend_buffer_t buf_w = nullptr;
    if (!cached && ZI_RESIDENT) {
        buf_w = ggml_backend_alloc_ctx_tensors(ctx_w, sc.backend);''', '''    ggml_backend_buffer_t buf_w = nullptr;
    if (ring && !ring_hit) {
        // Platsen allokeras EN gang (storsta blocket) och mappas vid forsta bruk; sedan placeras
        // blockets tensorer linjart i den. Ingen ny DSP-mappning per block.
        const size_t need = block_alloc_bytes(sc, "layers.0", true);
        if (!sc.ring_buf[ring_slot] || sc.ring_cap < need) {
            if (sc.ring_buf[ring_slot]) { g_mem_ring -= ggml_backend_buffer_get_size(sc.ring_buf[ring_slot]); ggml_backend_buffer_free(sc.ring_buf[ring_slot]); }
            sc.ring_buf[ring_slot] = ggml_backend_buft_alloc_buffer(ggml_backend_get_default_buffer_type(sc.backend), need);
            if (!sc.ring_buf[ring_slot]) { fprintf(stderr, "failed to allocate ring slot %d (%zu B)\\n", ring_slot, need); exit(1); }
            ggml_backend_buffer_set_usage(sc.ring_buf[ring_slot], GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            g_mem_ring += ggml_backend_buffer_get_size(sc.ring_buf[ring_slot]);
            sc.ring_cap = need;
            if (!sc.ring_ev[ring_slot]) sc.ring_ev[ring_slot] = ggml_backend_event_new(ggml_backend_get_device(sc.backend));
            if (!sc.ring_ev[ring_slot]) { fprintf(stderr, "ring: backenden saknar event\\n"); exit(1); }
        }
        ggml_tallocr ta = ggml_tallocr_new(sc.ring_buf[ring_slot]);
        for (ggml_tensor * t : to_upload) {
            if (ggml_tallocr_alloc(&ta, t) != GGML_STATUS_SUCCESS) { fprintf(stderr, "ring: %s ryms inte i platsen\\n", t->name); exit(1); }
        }
    } else if (!cached && ZI_RESIDENT) {
        buf_w = ggml_backend_alloc_ctx_tensors(ctx_w, sc.backend);''')
rep('''    if (!cached && ZI_RESIDENT) {
        BlockCache bc;''', '''    if (ring && !ring_hit) {
        sc.ring_ctx[ring_slot]   = ctx_w;
        sc.ring_w[ring_slot]     = w;
        sc.ring_owner[ring_slot] = pfx;
    } else if (!cached && ZI_RESIDENT) {
        sc.n_resident++;
        BlockCache bc;''')
rep('''    ggml_status st = ggml_backend_graph_compute(sc.backend, gf);
    int64_t t_c1 = ggml_time_us();
    if (st != GGML_STATUS_SUCCESS) { fprintf(stderr, "compute failed for %s: %d\\n", pfx.c_str(), st); exit(1); }
''', '''    ggml_status st = ggml_backend_graph_compute(sc.backend, gf);
    // Fence efter blockets ops: nasta fyllning av samma plats vantar bara pa DETTA block.
    if (ring) { ggml_backend_event_record(sc.ring_ev[ring_slot], sc.backend); sc.ring_rec[ring_slot] = true; }
    int64_t t_c1 = ggml_time_us();
    if (st != GGML_STATUS_SUCCESS) { fprintf(stderr, "compute failed for %s: %d\\n", pfx.c_str(), st); exit(1); }
''')

# 5) rapport + upprensning
rep('''    fprintf(stderr, "[budget] minne: vikter %.0f MB | graf (gallocr) %.0f MB | aktivering %.0f MB | io %.0f MB\\n",
            g_mem_w / 1048576.0,''', '''    if (ZI_WEIGHT_BUDGET_MB >= 0) {
        fprintf(stderr, "[budget] ring: tak %lld MB | %d block residenta (%.0f MB) | %d platser (%.0f MB) | %lld fyllningar, %lld traffar | fence-vanta %.2f s\\n",
                ZI_WEIGHT_BUDGET_MB, sc.n_resident, g_mem_w / 1048576.0, ZI_RING_SLOTS, g_mem_ring / 1048576.0,
                (long long) g_times.r_fills, (long long) g_times.r_hits, g_times.r_wait * 1e-6);
    }
    fprintf(stderr, "[budget] minne: vikter %.0f MB | graf (gallocr) %.0f MB | aktivering %.0f MB | io %.0f MB\\n",
            (g_mem_w + g_mem_ring) / 1048576.0,''')
rep('''    if (sc.galloc) ggml_gallocr_free(sc.galloc);
    if (sc.act_buf) ggml_backend_buffer_free(sc.act_buf);''', '''    ggml_backend_synchronize(sc.backend);
    for (int k = 0; k < 4; k++) {
        if (sc.ring_ev[k]) ggml_backend_event_free(sc.ring_ev[k]);
        if (sc.ring_buf[k]) ggml_backend_buffer_free(sc.ring_buf[k]);
        if (sc.ring_ctx[k]) ggml_free(sc.ring_ctx[k]);
    }
    if (sc.galloc) ggml_gallocr_free(sc.galloc);
    if (sc.act_buf) ggml_backend_buffer_free(sc.act_buf);''')

io.open(P, "w", encoding="utf-8", newline="").write(s)
print("OK")
