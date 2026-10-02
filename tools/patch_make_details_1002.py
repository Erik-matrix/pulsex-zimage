# 2026-10-02: zimage_make.cpp against zimage.py, detail by detail (read side by side):
#   * the number of prompt tokens is the SIZE of the encoder's file / (2560 * 4), as zimage.py's reshape(-1, 2560)
#   * ZI_SHIFT is written "%g" (zimage.py: "%g" % float(shift)); ZI_WEIGHT_BUDGET_MB only when >= 0
#   * a program folder with letters outside ASCII: the NPU driver does not load its files from there - say so
#   * the work folder is emptied whatever the engines left in it
# and zimage_stream.cpp flushes after every "[resident] steg" line, so a window can show the steps as they come
# (stdout is a pipe = fully buffered; until now the lines arrived together with "klar").
# and CMakeLists.txt gets the zimage-make target.
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent


def patch(name, pairs):
    p = HERE / name
    s = p.read_text(encoding="utf-8")
    for old, new in pairs:
        assert s.count(old) == 1, (name, old[:70], s.count(old))
        s = s.replace(old, new)
    p.write_text(s, encoding="utf-8", newline="\n")
    print("patched", name)


patch("zimage_make.cpp", [
    ("    const std::string shift = json_raw(defs, \"shift\").empty() ? \"1\" : json_raw(defs, \"shift\");\n",
     "    char shift[32];\n"
     "    snprintf(shift, sizeof shift, \"%g\", json_raw(defs, \"shift\").empty() ? 1.0 : atof(json_raw(defs, \"shift\").c_str()));\n"),
    ("    const std::string budget = json_raw(defs, \"npu_weight_budget_mb\").empty() ? \"0\" : json_raw(defs, \"npu_weight_budget_mb\");\n",
     "    const int budget = atoi(json_raw(defs, \"npu_weight_budget_mb\").c_str());\n"),
    ("    SetEnvironmentVariableA(\"ZI_SHIFT\", shift.c_str());\n    SetEnvironmentVariableA(\"ZI_WEIGHT_BUDGET_MB\", budget.c_str());\n",
     "    SetEnvironmentVariableA(\"ZI_SHIFT\", shift);\n"
     "    SetEnvironmentVariableA(\"ZI_WEIGHT_BUDGET_MB\", budget >= 0 ? std::to_string(budget).c_str() : nullptr);\n"),
    ("    const std::string taef1_w = models + ",
     "    for (const unsigned char c : bin)\n"
     "        if (c >= 0x80) throw std::runtime_error(\"the NPU cannot load its files from \" + bin + \" (letters outside A-Z) - move the program to a folder such as C:\\\\PulseX\");\n"
     "    const std::string taef1_w = models + "),
    ("    const int scap = encode(bin + \"\\\\zimage-encode.exe\", enc, bin, wd + \"\\\\encode_in.txt\", wd + \"\\\\cap_raw.bin\", err);\n"
     "    if (scap < 1) throw std::runtime_error(\"the text encoder failed: \" + err);\n",
     "    if (encode(bin + \"\\\\zimage-encode.exe\", enc, bin, wd + \"\\\\encode_in.txt\", wd + \"\\\\cap_raw.bin\", err) < 0)\n"
     "        throw std::runtime_error(\"the text encoder failed: \" + err);\n"
     "    const size_t cap_bytes = fluxqnn::read_file(wd + \"\\\\cap_raw.bin\").size();          // [scap, 2560] float32\n"
     "    const int scap = (int) (cap_bytes / (2560 * 4));\n"
     "    if (scap < 1 || cap_bytes % (2560 * 4)) throw std::runtime_error(\"the text encoder wrote \" + std::to_string(cap_bytes) + \" bytes - not rows of 2560 numbers\");\n"),
    ("        for (const char * f : { \"encode_in.txt\", \"cap_raw.bin\", \"cap_ids0.bin\", \"cap_ids1.bin\", \"cap_ids2.bin\", \"img_ids0.bin\", \"img_ids1.bin\", \"img_ids2.bin\", \"meta.txt\",\n"
     "                                \"lat_init.f32\", \"img_raw.bin\", \"t.bin\", \"lat_out.f32\", \"stream.log\", \"taef1_out.rgb\", \"taef1.log\", \"ggml_out.bin\" })\n"
     "            DeleteFileW(wide(wd + \"\\\\\" + f).c_str());\n",
     "        WIN32_FIND_DATAW fd;                                  // this run's own folder: whatever is in it\n"
     "        const HANDLE hf = FindFirstFileW(wide(wd + \"\\\\*\").c_str(), &fd);\n"
     "        if (hf != INVALID_HANDLE_VALUE) {\n"
     "            do {\n"
     "                if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) DeleteFileW((wide(wd) + L\"\\\\\" + fd.cFileName).c_str());\n"
     "            } while (FindNextFileW(hf, &fd));\n"
     "            FindClose(hf);\n"
     "        }\n"),
])

patch("zimage_stream.cpp", [
    ("               step + 1, n_steps, 1.0f - sig[step], ds, sqrt(s2), sqrt(s2 / xlat.size()));\n",
     "               step + 1, n_steps, 1.0f - sig[step], ds, sqrt(s2), sqrt(s2 / xlat.size()));\n"
     "        fflush(stdout);       // 10-02: a window shows the steps as they come (stdout is a pipe = fully buffered)\n"),
])

p = HERE / "CMakeLists.txt"
s = p.read_text(encoding="utf-8")
if "zimage-make" not in s:
    s = s.rstrip("\n") + """

# 2026-10-02: the whole chain without Python (see zimage_make.cpp) - what the window (pulsex-zimage) runs.
# Shares pulsex_make_util.hpp, flux_text.hpp, flux_enrich.hpp and the QNN wrapper with flux-make (../flux_dit).
set(FLUX_QNN_INCLUDE "" CACHE PATH "QNN headers: <QAIRT SDK>/include/QNN (needed for flux-make and flux-qnn)")
if(WIN32 AND EXISTS "${FLUX_QNN_INCLUDE}/QnnInterface.h")
    add_executable(zimage-make zimage_make.cpp)
    target_include_directories(zimage-make PRIVATE ${FLUX_QNN_INCLUDE} "${CMAKE_CURRENT_SOURCE_DIR}/../flux_dit")
    target_compile_features(zimage-make PRIVATE cxx_std_17)
    target_link_libraries(zimage-make PRIVATE windowscodecs ole32 gdiplus)
    target_link_options(zimage-make PRIVATE "LINKER:/MANIFESTINPUT:${CMAKE_CURRENT_SOURCE_DIR}/utf8.manifest")
endif()
"""
    p.write_text(s, encoding="utf-8", newline="\n")
    print("patched CMakeLists.txt")
