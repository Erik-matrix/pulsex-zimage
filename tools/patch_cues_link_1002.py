# 2026-10-02 ("lagg aven lankarna till ordlistorna i ZImage"): Z-Image gets the word list linked the way FLUX has it -
# paths.cues in the settings file - and its own copy of the list, so it does not reach into flux_dit:
#   * flux_dit/tools/make_cues_tsv_1002.py writes the SAME bytes to flux_dit/flux_cues.tsv and zimage_dit/cues.tsv
#   * zimage.json (this machine) and zimage.example.json (published) name it: paths.cues
#   * zimage-make: a named list that is not there falls back to the one beside the program / the settings file
#     (before: the picture was then made without the word list, with one line on stderr)
#   * the QuickSRNet context is looked for under both names zimage.py leaves it (<name>_ctx_qnn.bin and, when the
#     model was not 512 x 512, <name>_512x512_ctx_qnn.bin) - in zimage-make and in the window
#   * the window's target (pulsex-zimage) moves from flux_dit/CMakeLists.txt to zimage_dit/CMakeLists.txt, with the
#     shared sources in ../flux_dit or, in the published repository, in window/
#   * flux_dit/tools/export_github.py: flux_make.cpp now needs pulsex_make_util.hpp
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
Z, F = ROOT / "zimage_dit", ROOT / "flux_dit"


def patch(p, pairs):
    s = p.read_text(encoding="utf-8")
    for old, new in pairs:
        assert s.count(old) == 1, (p.name, old[:70], s.count(old))
        s = s.replace(old, new)
    p.write_text(s, encoding="utf-8", newline="\n")
    print("patched", p.relative_to(ROOT))


patch(F / "tools" / "make_cues_tsv_1002.py", [
    ("# -> flux_dit\\flux_cues.tsv, a file flux-make.exe reads without a JSON library.",
     "# -> flux_dit\\flux_cues.tsv and (the same bytes) zimage_dit\\cues.tsv, a file flux-make.exe and zimage-make.exe read\n# without a JSON library."),
    ("DST.write_text(\"\\n\".join(L) + \"\\n\", encoding=\"utf-8\", newline=\"\\n\")\n",
     "DST.write_text(\"\\n\".join(L) + \"\\n\", encoding=\"utf-8\", newline=\"\\n\")\n"
     "(SRC.parent / \"cues.tsv\").write_bytes(DST.read_bytes())          # Z-Image's own copy (zimage.json paths.cues)\n"),
])

patch(Z / "zimage_make.cpp", [
    ("        std::string cues = path(\"cues\");\n",
     "        std::string cues = path(\"cues\");\n"
     "        if (!cues.empty() && !exists(cues)) cues.clear();           // named but not there: look where it usually is\n"),
    ("        for (const std::string & c : { exe_dir() + \"\\\\cues.tsv\", exe_dir() + \"\\\\flux_cues.tsv\", base + \"\\\\flux_cues.tsv\", base + \"\\\\..\\\\flux_dit\\\\flux_cues.tsv\" })",
     "        for (const std::string & c : { base + \"\\\\cues.tsv\", exe_dir() + \"\\\\cues.tsv\", exe_dir() + \"\\\\..\\\\cues.tsv\", exe_dir() + \"\\\\flux_cues.tsv\" })"),
    ("    if (sr.size() > 5 && sr.substr(sr.size() - 5) == \".onnx\") sr = sr.substr(0, sr.size() - 5) + \"_ctx_qnn.bin\";\n",
     "    if (sr.size() > 5 && sr.substr(sr.size() - 5) == \".onnx\") {          // the compiled context zimage.py left beside the model\n"
     "        const std::string stem = sr.substr(0, sr.size() - 5);\n"
     "        sr = stem + \"_ctx_qnn.bin\";\n"
     "        if (!exists(sr) && exists(stem + \"_512x512_ctx_qnn.bin\")) sr = stem + \"_512x512_ctx_qnn.bin\";      // a model of another input size: its 512 x 512 copy\n"
     "    }\n"),
])

patch(F / "flux_gui.cpp", [
    ("            std::string sr = zat(\"quicksrnet_x4\");\n"
     "            if (sr.size() > 5 && sr.substr(sr.size() - 5) == \".onnx\") sr = sr.substr(0, sr.size() - 5) + \"_ctx_qnn.bin\";\n"
     "            has_up = file_exists(sr);\n",
     "            std::string sr = zat(\"quicksrnet_x4\");                // the compiled context, under either name zimage.py leaves it (as zimage-make)\n"
     "            if (sr.size() > 5 && sr.substr(sr.size() - 5) == \".onnx\") sr = sr.substr(0, sr.size() - 5);\n"
     "            has_up = file_exists(sr + \"_ctx_qnn.bin\") || file_exists(sr + \"_512x512_ctx_qnn.bin\") || file_exists(sr);\n"),
])

patch(F / "tools" / "export_github.py", [
    ("\"flux_qnn_graph.hpp\", \"pcore_qnn_min.hpp\",", "\"flux_qnn_graph.hpp\", \"pcore_qnn_min.hpp\", \"pulsex_make_util.hpp\","),
])

# CMake: the window's target moves to zimage_dit
p = F / "CMakeLists.txt"
s = p.read_text(encoding="utf-8")
a = s.index("\n    # 2026-10-02: the same window for Z-Image-Turbo (PULSEX_ZIMAGE)")
b = s.index("endif()", a)
s = s[:a] + "\n" + s[b:]
p.write_text(s, encoding="utf-8", newline="\n")
print("patched flux_dit/CMakeLists.txt (pulsex-zimage removed)")

p = Z / "CMakeLists.txt"
s = p.read_text(encoding="utf-8")
a = s.index("\n# 2026-10-02: the whole chain without Python (see zimage_make.cpp)")
s = s[:a] + """
# 2026-10-02: the whole chain without Python (see zimage_make.cpp), and the window that runs it (PulseX Z-Image).
# Both share sources with PulseX Image: pulsex_make_util.hpp, flux_text.hpp, flux_enrich.hpp, flux_image.hpp, the QNN
# wrapper and the window itself (flux_gui.cpp, built here with PULSEX_ZIMAGE). They are in ../flux_dit in the
# development tree and in window/ in the published repository.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../flux_dit/flux_gui.cpp")
    set(PULSEX_SHARED_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../flux_dit")
else()
    set(PULSEX_SHARED_DIR "${CMAKE_CURRENT_SOURCE_DIR}/window")
endif()
set(FLUX_QNN_INCLUDE "" CACHE PATH "QNN headers: <QAIRT SDK>/include/QNN (needed for zimage-make)")
if(WIN32 AND EXISTS "${FLUX_QNN_INCLUDE}/QnnInterface.h")
    add_executable(zimage-make zimage_make.cpp)
    target_include_directories(zimage-make PRIVATE ${FLUX_QNN_INCLUDE} ${PULSEX_SHARED_DIR})
    target_compile_features(zimage-make PRIVATE cxx_std_17)
    target_link_libraries(zimage-make PRIVATE windowscodecs ole32 gdiplus)
    target_link_options(zimage-make PRIVATE "LINKER:/MANIFESTINPUT:${CMAKE_CURRENT_SOURCE_DIR}/utf8.manifest")
endif()
set(FLUX_IMGUI_DIR "" CACHE PATH "Dear ImGui root (needed for the window, pulsex-zimage)")
if(WIN32 AND EXISTS "${FLUX_IMGUI_DIR}/imgui.cpp")
    add_executable(pulsex-zimage
        ${PULSEX_SHARED_DIR}/flux_gui.cpp
        ${FLUX_IMGUI_DIR}/imgui.cpp
        ${FLUX_IMGUI_DIR}/imgui_draw.cpp
        ${FLUX_IMGUI_DIR}/imgui_tables.cpp
        ${FLUX_IMGUI_DIR}/imgui_widgets.cpp
        ${FLUX_IMGUI_DIR}/backends/imgui_impl_win32.cpp
        ${FLUX_IMGUI_DIR}/backends/imgui_impl_dx11.cpp
    )
    target_include_directories(pulsex-zimage PRIVATE ${FLUX_IMGUI_DIR} ${FLUX_IMGUI_DIR}/backends ${PULSEX_SHARED_DIR})
    target_compile_features(pulsex-zimage PRIVATE cxx_std_17)
    target_compile_definitions(pulsex-zimage PRIVATE NOMINMAX PULSEX_ZIMAGE)
    target_link_libraries(pulsex-zimage PRIVATE d3d11 dxgi d3dcompiler comdlg32 windowscodecs ole32 shell32 dwmapi)
    target_link_options(pulsex-zimage PRIVATE "LINKER:/SUBSYSTEM:WINDOWS" "LINKER:/ENTRY:mainCRTStartup"
                        "LINKER:/MANIFESTINPUT:${PULSEX_SHARED_DIR}/gui.manifest")
endif()
"""
p.write_text(s, encoding="utf-8", newline="\n")
print("patched zimage_dit/CMakeLists.txt (zimage-make + pulsex-zimage)")

# the settings files: paths.cues
for name, value in (("zimage.json", "C:/PulseCore/PulseX/zimage_dit/cues.tsv"), ("zimage.example.json", "cues.tsv")):
    p = Z / name
    j = json.loads(p.read_text(encoding="utf-8"))
    j["paths"]["cues"] = value
    if "_help" in j:
        j["_help"]["paths.cues"] = ("the word list for Swedish words and material words as a table (cues.tsv, made from cues.json) - read by "
                                    "zimage-make.exe and the window (PulseX Z-Image); zimage.py reads cues.json itself")
    p.write_text(json.dumps(j, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print("patched", name)
