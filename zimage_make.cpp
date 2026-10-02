// zimage-make: a prompt in, a picture out - the whole Z-Image-Turbo chain WITHOUT Python.
//
// 2026-10-02 (Z-Image should get the same window as FLUX). zimage.py is the reference; this is the same chain in one
// C++ program, so a window (pulsex-zimage.exe, flux_gui.cpp built for Z-Image) has nothing to install beside it:
//   1. the prompt (the LoRA's trigger word in front, zimage.json defaults.prompt_prefix) in the Qwen3 chat template
//                                                                        -> zimage-encode.exe      (NPU, own process)
//   2. the start noise and the position tables the engine reads          -> zimage-dit-stream.exe  (NPU, own process)
//   3. the latent to pixels                                              -> taef1_decode.exe       (CPU)
//   4. enlarging (QuickSRNet as a QNN context, in this process), the text on the picture, JPEG or PNG
// Settings come from zimage.json - the file zimage.py and zimage.ps1 read ({models} expands, relative paths start at
// the file's folder). Everything shared with flux-make is in ../flux_dit/pulsex_make_util.hpp.
//
// ONE thing differs from zimage.py: the start noise. zimage.py takes it from numpy's generator (PCG64 + ziggurat),
// which this program does not rebuild; it uses torch.randn's (as flux-make). So a seed gives ANOTHER picture here
// than in zimage.py - both are pictures of the prompt. --noise-file feeds a given noise (the check that this chain
// and zimage.py make the same picture from the same noise: tools/make_vs_py_1002.py).
//
//   zimage-make "prompt" [--size 512|WxH] [--seed N] [--steps 4] [--up 1024|2048] [--enrich nouns|full]
//               [--title text] [--sub text] [--top] [--out file.jpg|png] [--quality 95] [--config zimage.json] [--keep]
//   zimage-make --prompt-file prompt.txt --progress ...     (what the window runs)
//   zimage-make --add-text picture --title ... | --enlarge picture [--up N]      (no model: as flux-make)
// stdout: the picture's path and the timings. With --progress also "PROGRESS <stage>" lines.

#include "pulsex_make_util.hpp"
#include "flux_text.hpp"
#include "flux_enrich.hpp"

// the text of the object that follows "key": ... in a JSON text ("" when absent) - zimage.json has "paths" and
// "defaults", and some names ("encoder") are in both
static std::string json_section(const std::string & js, const std::string & key) {
    size_t k = js.find("\"" + key + "\"");
    if (k == std::string::npos) return "";
    k = js.find('{', k);
    if (k == std::string::npos) return "";
    int depth = 0;
    bool in_str = false;
    for (size_t i = k; i < js.size(); i++) {
        const char c = js[i];
        if (in_str) { if (c == '\\') i++; else if (c == '"') in_str = false; continue; }
        if (c == '"') in_str = true;
        else if (c == '{') depth++;
        else if (c == '}' && --depth == 0) return js.substr(k, i - k + 1);
    }
    return "";
}
// a value that is not a string (a number, true / false) as text; "" when absent
static std::string json_raw(const std::string & js, const std::string & key) {
    size_t k = js.find("\"" + key + "\"");
    if (k == std::string::npos) return "";
    k = js.find(':', k);
    size_t a = k + 1;
    while (a < js.size() && (js[a] == ' ' || js[a] == '\t')) a++;
    size_t b = a;
    while (b < js.size() && js[b] != ',' && js[b] != '\n' && js[b] != '\r' && js[b] != '}') b++;
    std::string v = js.substr(a, b - a);
    while (!v.empty() && (v.back() == ' ' || v.back() == '\t')) v.pop_back();
    if (v.size() >= 2 && v.front() == '"' && v.back() == '"') v = v.substr(1, v.size() - 2);
    return v;
}

// zimage-encode.exe speaks on stdin / stdout: "READY", then one line "<text file>\t<output file>", then "DONE <n>"
static int encode(const std::string & exe, const std::string & model, const std::string & cwd, const std::string & text_file, const std::string & out_file,
                  std::string & err) {
    SECURITY_ATTRIBUTES sa{ sizeof sa, nullptr, TRUE };
    HANDLE ord = nullptr, owr = nullptr, ird = nullptr, iwr = nullptr;
    if (!CreatePipe(&ord, &owr, &sa, 0) || !CreatePipe(&ird, &iwr, &sa, 0)) throw std::runtime_error("CreatePipe failed");
    SetHandleInformation(ord, HANDLE_FLAG_INHERIT, 0);
    SetHandleInformation(iwr, HANDLE_FLAG_INHERIT, 0);
    HANDLE nul = CreateFileW(L"NUL", GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, &sa, OPEN_EXISTING, 0, nullptr);
    std::wstring cmd = L"\"" + wide(exe) + L"\" \"" + wide(model) + L"\" HTP0 35";
    STARTUPINFOW si{};
    si.cb = sizeof si;
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdOutput = owr; si.hStdError = nul; si.hStdInput = ird;
    PROCESS_INFORMATION pi{};
    const BOOL ok = CreateProcessW(nullptr, &cmd[0], nullptr, nullptr, TRUE, CREATE_NO_WINDOW, nullptr, wide(cwd).c_str(), &si, &pi);
    CloseHandle(owr); CloseHandle(ird);
    if (nul != INVALID_HANDLE_VALUE) CloseHandle(nul);
    if (!ok) { CloseHandle(ord); CloseHandle(iwr); throw std::runtime_error("cannot start " + exe); }
    auto read_line = [&]() {
        std::string line;
        char c;
        DWORD n = 0;
        while (ReadFile(ord, &c, 1, &n, nullptr) && n == 1) {
            if (c == '\n') break;
            if (c != '\r') line += c;
        }
        return line;
    };
    int n_tok = -1;
    const std::string ready = read_line();
    if (ready == "READY") {
        const std::string req = text_file + "\t" + out_file + "\n";
        DWORD wrote = 0;
        WriteFile(iwr, req.data(), (DWORD) req.size(), &wrote, nullptr);
        const std::string ans = read_line();
        if (ans.compare(0, 4, "DONE") == 0) n_tok = atoi(ans.c_str() + 4);
        else err = ans.empty() ? "no answer" : ans;
    } else err = ready.empty() ? "the encoder process ended" : ready;
    CloseHandle(iwr);                        // EOF: the process frees the model, closes the NPU and exits by itself
    WaitForSingleObject(pi.hProcess, INFINITE);
    CloseHandle(ord);
    CloseHandle(pi.hProcess); CloseHandle(pi.hThread);
    return n_tok;
}

template <typename T> static void write_bin(const std::string & p, const std::vector<T> & v) {
    FILE * f = fopen(p.c_str(), "wb");
    if (!f) throw std::runtime_error("cannot write " + p);
    fwrite(v.data(), sizeof(T), v.size(), f);
    fclose(f);
}

static int run(int argc, char ** argv) {
    std::string prompt, size = "512", out, cfg_path = exe_dir() + "\\zimage.json", title, subtitle, add_text, enlarge_src, enrich_mode = "off", noise_file;
    bool text_top = false, enrich_only = false, keep = false;
    int up = 0, quality = 95, steps = 0;
    uint32_t seed = 0;
    if (!exists(cfg_path) && exists(exe_dir() + "\\..\\zimage.json")) cfg_path = exe_dir() + "\\..\\zimage.json";      // bin\ under the folder with zimage.json
    if (const char * e = getenv("ZIMAGE_CONFIG")) if (*e) cfg_path = e;
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        auto val = [&]() -> std::string { if (i + 1 >= argc) throw std::runtime_error(a + " needs a value"); return argv[++i]; };
        if (a == "--size") size = val();
        else if (a == "--seed") seed = (uint32_t) strtoul(val().c_str(), nullptr, 10);
        else if (a == "--steps") steps = atoi(val().c_str());
        else if (a == "--out") out = val();
        else if (a == "--config") cfg_path = val();
        else if (a == "--keep") keep = true;
        else if (a == "--up") up = atoi(val().c_str());
        else if (a == "--progress") g_progress = true;
        else if (a == "--title") title = val();
        else if (a == "--sub" || a == "--subtitle") subtitle = val();
        else if (a == "--top") text_top = true;
        else if (a == "--quality") quality = atoi(val().c_str());
        else if (a == "--add-text") add_text = val();
        else if (a == "--enlarge") enlarge_src = val();
        else if (a == "--enrich") enrich_mode = val();
        else if (a == "--enrich-only") enrich_only = true;
        else if (a == "--noise-file") noise_file = val();              // [16, h/8, w/8] float32: the start noise, instead of --seed's
        else if (a == "--prompt-file") {
            const std::vector<uint8_t> v = fluxqnn::read_file(val());
            prompt.assign(v.begin(), v.end());
            if (prompt.compare(0, 3, "\xEF\xBB\xBF") == 0) prompt.erase(0, 3);
            while (!prompt.empty() && (prompt.back() == '\n' || prompt.back() == '\r' || prompt.back() == ' ')) prompt.pop_back();
        }
        else if (prompt.empty()) prompt = a;
        else throw std::runtime_error("what is " + a + "?");
    }
    const bool has_text = !title.empty() || !subtitle.empty();
    if (!add_text.empty()) {               // the text alone, on a picture that exists - no NPU
        if (!has_text) throw std::runtime_error("--add-text needs --title and / or --sub");
        fluximg::Rgb im = fluximg::load(add_text);
        fluxtext::add_text(im, title, subtitle, text_top);
        if (out.empty()) { const size_t d = add_text.find_last_of('.'); out = (d == std::string::npos ? add_text : add_text.substr(0, d)) + "_text.png"; }
        fluximg::save(out, im, quality, true);
        printf("%s  %dx%d, text added\n", out.c_str(), im.w, im.h);
        fflush(stdout);
        return 0;
    }
    if (prompt.empty() && enlarge_src.empty()) {
        fprintf(stderr, "usage: zimage-make \"prompt\" [--size 512|WxH] [--seed N] [--steps 4] [--up 1024|2048] [--enrich nouns|full] [--title text] [--sub text] [--top] [--out file.jpg]\n");
        return 2;
    }
    int w = atoi(size.c_str()), h = w;
    { const size_t x = size.find_first_of("xX"); if (x != std::string::npos) { w = atoi(size.substr(0, x).c_str()); h = atoi(size.substr(x + 1).c_str()); } }
    if (w < 256 || h < 256 || w > 2048 || h > 2048 || w % 16 || h % 16) throw std::runtime_error("--size: 256..2048 per side and divisible by 16 (for example 848x480)");

    // zimage.json: "paths" (with {models}) and "defaults"
    const std::vector<uint8_t> cj = fluxqnn::read_file(cfg_path);
    const std::string js(cj.begin(), cj.end()), base = dir_of(full_path(cfg_path));
    const std::string paths = json_section(js, "paths"), defs = json_section(js, "defaults");
    if (paths.empty()) throw std::runtime_error(cfg_path + " has no \"paths\"");
    const std::string models = cfg(paths, base, "models");
    auto path = [&](const std::string & key) {
        std::string p = json_str(paths, key);
        const size_t m = p.find("{models}");
        if (m != std::string::npos) p = models + p.substr(m + 8);
        if (p.empty()) return p;
        p = winpath(p);
        const bool absolute = (p.size() > 1 && p[1] == ':') || p[0] == '\\';
        return full_path(absolute ? p : base + "\\" + p);
    };
    const std::string bin = json_str(paths, "bin").empty() ? exe_dir() : path("bin"), dit = path("dit"), enc = path("encoder"), taef1 = path("taef1"),
                      rt = path("qnn_runtime");
    std::string sr = path("quicksrnet_x4");
    if (sr.size() > 5 && sr.substr(sr.size() - 5) == ".onnx") {          // the compiled context zimage.py left beside the model
        const std::string stem = sr.substr(0, sr.size() - 5);
        sr = stem + "_ctx_qnn.bin";
        if (!exists(sr) && exists(stem + "_512x512_ctx_qnn.bin")) sr = stem + "_512x512_ctx_qnn.bin";      // a model of another input size: its 512 x 512 copy
    }
    if (steps <= 0) steps = (std::max)(1, atoi(json_raw(defs, "steps").c_str()));
    char shift[32];
    snprintf(shift, sizeof shift, "%g", json_raw(defs, "shift").empty() ? 1.0 : atof(json_raw(defs, "shift").c_str()));
    const std::string prefix = json_str(defs, "prompt_prefix");
    const bool pack_cache = json_raw(defs, "pack_cache") != "false";
    const int budget = atoi(json_raw(defs, "npu_weight_budget_mb").c_str());

    // the Norrland cue table (the tsv flux-make reads: tools/make_cues_tsv_1002.py); Z-Image's order: appended
    std::string sent = prompt;
    if (enrich_mode == "nouns" || enrich_mode == "full") {
        std::string cues = path("cues");
        if (!cues.empty() && !exists(cues)) cues.clear();           // named but not there: look where it usually is
        for (const std::string & c : { base + "\\cues.tsv", exe_dir() + "\\cues.tsv", exe_dir() + "\\..\\cues.tsv", exe_dir() + "\\flux_cues.tsv" })
            if (cues.empty() && exists(c)) cues = c;                // beside the program, beside zimage.json, or in the source tree
        const fluxenrich::Table tab = fluxenrich::load(cues);
        if (!tab.ok) fprintf(stderr, "(--enrich: no cue table at %s - the prompt is sent as typed)\n", cues.c_str());
        else sent = fluxenrich::enrich(tab, prompt, enrich_mode == "full", nullptr, false);
    } else if (enrich_mode != "off") throw std::runtime_error("--enrich takes nouns, full or off");
    if (enrich_only) { printf("%s\n", sent.c_str()); return 0; }

    if (up && ((std::min)(w, h) < 512 || up <= (std::min)(w, h) || up > 4 * (std::min)(w, h) || !exists(sr) || !exists(rt)) && enlarge_src.empty()) {
        fprintf(stderr, "(--up %d: the short side must be at least 512 and grow at most four times, and the QuickSRNet context %s must exist - keeping %dx%d)\n", up, sr.c_str(), w, h);
        up = 0;
    }
    if (!enlarge_src.empty()) {            // an EXISTING picture through the upscaler (and the text, if any)
        fluximg::Rgb im = fluximg::load(enlarge_src);
        const int m = (std::min)(im.w, im.h);
        if (!up) up = m < 1024 ? 2 * m : 2048;
        if (m < 512 || (std::max)(im.w, im.h) > 1536 || up <= m || up > 4 * m || !exists(sr) || !exists(rt))
            throw std::runtime_error("--enlarge: the picture's short side must be 512 or more, its long side 1536 or less, --up between them and four times the short side, and the QuickSRNet context must exist (" + sr + ")");
        const double t0 = now_s();
        SetDllDirectoryW(wide(rt).c_str());
        {
            pcore_npu::QnnHtpSession S;
            if (!S.open()) throw std::runtime_error("QNN session: " + S.error());
            progress("upscale");
            enlarge(S, sr, im.px, im.w, im.h, up);
        }
        fluxtext::add_text(im, title, subtitle, text_top);
        if (out.empty()) { const size_t d = enlarge_src.find_last_of('.'); out = (d == std::string::npos ? enlarge_src : enlarge_src.substr(0, d)) + "_" + std::to_string(up) + ".png"; }
        fluximg::save(out, im, quality, has_text);
        printf("%s  %dx%d, enlarged in %.1f s\n", out.c_str(), im.w, im.h, now_s() - t0);
        fflush(stdout);
        return 0;
    }
    for (const std::string * f : { &bin, &dit, &enc, &taef1 })
        if (f->empty() || !exists(*f)) throw std::runtime_error("zimage.json: missing: " + *f);
    for (const unsigned char c : bin)
        if (c >= 0x80) throw std::runtime_error("the NPU cannot load its files from " + bin + " (letters outside A-Z) - move the program to a folder such as C:\\PulseX");
    const std::string taef1_w = models + "\\taef1\\diffusion_pytorch_model.safetensors";
    if (!exists(taef1_w)) throw std::runtime_error("the picture decoder's weights are missing: " + taef1_w);

    const double T0 = now_s();
    char tmp[MAX_PATH];
    GetTempPathA(MAX_PATH, tmp);
    const std::string wd = std::string(tmp) + "pulsex_zimage_" + std::to_string(GetCurrentProcessId()) + "_" + std::to_string(GetTickCount64() & 0xFFFFFF);
    CreateDirectoryW(wide(wd).c_str(), nullptr);
    // what the engines read from the environment (zimage.py sets the same)
    // Where the NPU looks for its DSP-side modules: the engines' folder (the ggml module) and, for the upscaler, the QNN
    // runtime's (Qualcomm's libQnnHtpV73Skel.so and its catalog beside QnnHtp.dll - without them the upscaler ends in
    // 0x80000406 on a PC that has no such path set by hand; this one has, which hid it until 10-02). In the process
    // block (the engines inherit it) and in the CRT's copy (what a library in this process reads with getenv).
    {
        std::wstring old(32768, L'\0');
        const DWORD n = GetEnvironmentVariableW(L"ADSP_LIBRARY_PATH", &old[0], (DWORD) old.size());
        old.resize(n < old.size() ? n : 0);
        const std::wstring adsp = wide(bin) + (rt.empty() ? L"" : L";" + wide(rt)) + (old.empty() ? L"" : L";" + old);
        SetEnvironmentVariableW(L"ADSP_LIBRARY_PATH", adsp.c_str());
        _wputenv_s(L"ADSP_LIBRARY_PATH", adsp.c_str());
    }
    SetEnvironmentVariableA("ZI_FLASH", "1");
    SetEnvironmentVariableA("ZI_DUMP", "0");
    SetEnvironmentVariableA("ZI_SERVE", nullptr);
    SetEnvironmentVariableA("ZI_STEPS", std::to_string(steps).c_str());
    SetEnvironmentVariableA("ZI_SHIFT", shift);
    SetEnvironmentVariableA("ZI_WEIGHT_BUDGET_MB", budget >= 0 ? std::to_string(budget).c_str() : nullptr);
    if (!getenv("GGML_HEXAGON_PD_DUMP")) SetEnvironmentVariableA("GGML_HEXAGON_PD_DUMP", "1");
    if (pack_cache) SetEnvironmentVariableW(L"ZI_PACKCACHE", wide(dit.substr(0, dit.rfind('.')) + ".hexpack").c_str());
    else SetEnvironmentVariableA("ZI_PACKCACHE", nullptr);

    // 1. text: the LoRA's trigger word first (unless it is there), the chat template, the short-lived NPU encoder
    std::string full = sent;
    if (!prefix.empty()) {
        std::string head = prefix;
        while (!head.empty() && (head.back() == ' ' || head.back() == ',')) head.pop_back();
        std::string lead = full;
        lead.erase(0, lead.find_first_not_of(" \t"));
        const std::string lo = fluxenrich::lower(lead), hl = fluxenrich::lower(head);
        if (lo.compare(0, hl.size(), hl) != 0) full = prefix + full;
    }
    if (sent != prompt) { printf("sent: %s\n", sent.c_str()); fflush(stdout); }      // what the cue table made of it (the trigger word is not shown)
    write_text(wd + "\\encode_in.txt", "<|im_start|>user\n" + full + "<|im_end|>\n<|im_start|>assistant\n");
    double t = now_s();
    progress("text");
    std::string err;
    if (encode(bin + "\\zimage-encode.exe", enc, bin, wd + "\\encode_in.txt", wd + "\\cap_raw.bin", err) < 0)
        throw std::runtime_error("the text encoder failed: " + err);
    const size_t cap_bytes = fluxqnn::read_file(wd + "\\cap_raw.bin").size();          // [scap, 2560] float32
    const int scap = (int) (cap_bytes / (2560 * 4));
    if (scap < 1 || cap_bytes % (2560 * 4)) throw std::runtime_error("the text encoder wrote " + std::to_string(cap_bytes) + " bytes - not rows of 2560 numbers");
    const double t_enc = now_s() - t;

    // 2. what the engine reads: positions, the start noise, t
    const int lh = h / 8, lw = w / 8, ht = lh / 2, wt = lw / 2, nimg = ht * wt;
    {
        std::vector<int32_t> ids((size_t) scap);
        for (int i = 0; i < scap; i++) ids[(size_t) i] = i + 1;
        write_bin(wd + "\\cap_ids0.bin", ids);
        std::fill(ids.begin(), ids.end(), 0);
        write_bin(wd + "\\cap_ids1.bin", ids);
        write_bin(wd + "\\cap_ids2.bin", ids);
        std::vector<int32_t> i0((size_t) nimg, scap + 1), i1((size_t) nimg), i2((size_t) nimg);
        for (int y = 0; y < ht; y++)
            for (int x = 0; x < wt; x++) { i1[(size_t) y * wt + x] = y; i2[(size_t) y * wt + x] = x; }
        write_bin(wd + "\\img_ids0.bin", i0); write_bin(wd + "\\img_ids1.bin", i1); write_bin(wd + "\\img_ids2.bin", i2);
        write_text(wd + "\\meta.txt", std::to_string(ht) + " " + std::to_string(wt) + " " + std::to_string(nimg) + " " + std::to_string(scap) + " " + std::to_string(nimg + scap) + "\n");
        std::vector<float> noise;
        if (!noise_file.empty()) {
            const std::vector<uint8_t> raw = fluxqnn::read_file(noise_file);
            if (raw.size() != (size_t) 16 * lh * lw * 4) throw std::runtime_error("--noise-file has the wrong size for " + std::to_string(w) + "x" + std::to_string(h));
            noise.resize((size_t) 16 * lh * lw);
            memcpy(noise.data(), raw.data(), raw.size());
        } else noise = torch_randn((size_t) 16 * lh * lw, seed);
        write_bin(wd + "\\lat_init.f32", noise);
        write_bin(wd + "\\img_raw.bin", std::vector<float>((size_t) nimg * 16 * 4, 0.0f));
        write_bin(wd + "\\t.bin", std::vector<float>(1, 1.0f));
    }
    t = now_s();
    progress("step 0 " + std::to_string(steps));
    std::string so;
    std::function<void(const std::string &)> on_step;
    if (g_progress)
        on_step = [steps](const std::string & line) {          // "[resident] steg 2/4 ..." = the second step is done
            if (line.compare(0, 16, "[resident] steg ") == 0) progress("step " + std::to_string(atoi(line.c_str() + 16)) + " " + std::to_string(steps));
        };
    if (spawn({ bin + "\\zimage-dit-stream.exe", dit, wd, "HTP0" }, bin, wd + "\\stream.log", so, nullptr, on_step) != 0)
        throw std::runtime_error("the picture model failed: " + so.substr(so.size() > 600 ? so.size() - 600 : 0) + tail_of(wd + "\\stream.log"));
    const double t_dit = now_s() - t;
    {
        const std::vector<uint8_t> lraw = fluxqnn::read_file(wd + "\\lat_out.f32");
        if (lraw.size() != (size_t) 16 * lh * lw * 4) throw std::runtime_error("lat_out.f32 has the wrong size");
        const float * lat = (const float *) lraw.data();
        for (size_t i = 0; i < (size_t) 16 * lh * lw; i++) if (!std::isfinite(lat[i])) throw std::runtime_error("the latent contains NaN");
    }

    // 3. pixels: taef1 takes the raw diffusion latent [16, lh, lw]
    t = now_s();
    progress("picture");
    SetEnvironmentVariableW(L"PULSE_TAEF1_WEIGHTS", wide(taef1_w).c_str());
    if (spawn({ taef1, wd + "\\lat_out.f32", wd + "\\taef1_out.rgb", "--hw", std::to_string(lh) + "x" + std::to_string(lw) }, bin, wd + "\\taef1.log", so) != 0 ||
        !exists(wd + "\\taef1_out.rgb"))
        throw std::runtime_error("the picture decoder failed: " + so + tail_of(wd + "\\taef1.log"));
    std::vector<uint8_t> rgb = fluxqnn::read_file(wd + "\\taef1_out.rgb");
    if (rgb.size() != (size_t) w * h * 3) throw std::runtime_error("the picture decoder wrote another size than " + std::to_string(w) + "x" + std::to_string(h));
    const double t_vae = now_s() - t;
    double t_up = 0;
    if (up) {
        t = now_s();
        progress("upscale");
        SetDllDirectoryW(wide(rt).c_str());
        // The picture exists; an upscaler that does not run here (its graph is compiled for one chip, its runtime is
        // Qualcomm's) must not cost it: the picture is then saved at its own size, and the reason is said.
        try {
            pcore_npu::QnnHtpSession S;                    // the engines have closed the NPU; this opens and closes it for the upscaler
            if (!S.open()) throw std::runtime_error("QNN session: " + S.error());
            enlarge(S, sr, rgb, w, h, up);
            t_up = now_s() - t;
        } catch (const std::exception & e) {
            fprintf(stderr, "(the picture could not be enlarged - saved as %dx%d: %s)\n", w, h, e.what());
            printf("note: not enlarged\n");
            fflush(stdout);
        }
    }
    if (out.empty()) {
        SYSTEMTIME st;
        GetLocalTime(&st);
        char b[80];
        snprintf(b, sizeof b, "zimage_%04d%02d%02d_%02d%02d%02d_s%u.jpg", st.wYear, st.wMonth, st.wDay, st.wHour, st.wMinute, st.wSecond, seed);
        out = b;
    }
    {
        fluximg::Rgb im;
        im.w = w; im.h = h; im.px.swap(rgb);
        fluxtext::add_text(im, title, subtitle, text_top);
        fluximg::save(out, im, quality, has_text);
    }
    printf("%s  %dx%d, %d steps, seed %u, %d prompt tokens\n", out.c_str(), w, h, steps, seed, scap);
    if (t_up > 0) printf("text %.1f s + picture model %.1f s + decoder %.1f s + upscale %.1f s = %.1f s\n", t_enc, t_dit, t_vae, t_up, now_s() - T0);
    else printf("text %.1f s + picture model %.1f s + decoder %.1f s = %.1f s\n", t_enc, t_dit, t_vae, now_s() - T0);
    if (keep) printf("work folder kept: %s\n", wd.c_str());
    else {
        WIN32_FIND_DATAW fd;                                  // this run's own folder: whatever is in it
        const HANDLE hf = FindFirstFileW(wide(wd + "\\*").c_str(), &fd);
        if (hf != INVALID_HANDLE_VALUE) {
            do {
                if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) DeleteFileW((wide(wd) + L"\\" + fd.cFileName).c_str());
            } while (FindNextFileW(hf, &fd));
            FindClose(hf);
        }
        RemoveDirectoryW(wide(wd).c_str());
    }
    fflush(stdout);
    return 0;
}

int main(int argc, char ** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception & e) {
        fprintf(stderr, "zimage-make: %s\n", e.what());
        return 1;
    }
}
