set(TARGET zimage-dit-block)
add_executable(${TARGET} zimage_block.cpp)
target_link_libraries(${TARGET} PRIVATE ggml ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET2 zimage-dit-full)
add_executable(${TARGET2} zimage_full.cpp)
target_link_libraries(${TARGET2} PRIVATE ggml ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET2} PRIVATE cxx_std_17)

set(TARGET3 zimage-dit-stream)
add_executable(${TARGET3} zimage_stream.cpp)
if(WIN32)
    # 10-01: ANSI code page = UTF-8 (paths with å/ä/ö, Ł, ...) - merged into the linker's own manifest
    target_link_options(${TARGET3} PRIVATE "LINKER:/MANIFESTINPUT:${CMAKE_CURRENT_SOURCE_DIR}/utf8.manifest")
endif()
target_link_libraries(${TARGET3} PRIVATE ggml ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET3} PRIVATE cxx_std_17)

set(TARGET4 zimage-dit-vae)
add_executable(${TARGET4} zimage_vae.cpp)
target_link_libraries(${TARGET4} PRIVATE ggml ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET4} PRIVATE cxx_std_17)

set(TARGET5 zimage-dit-vae-stream)
add_executable(${TARGET5} zimage_vae_stream.cpp)
target_link_libraries(${TARGET5} PRIVATE ggml)
target_compile_features(${TARGET5} PRIVATE cxx_std_17)

# 09-28: the text encoder as a short-lived NPU process (see zimage_encode.cpp)
set(TARGET6 zimage-encode)
add_executable(${TARGET6} zimage_encode.cpp)
if(WIN32)
    # 10-01: the prompt file path arrives as UTF-8 on stdin - merged into the linker's own manifest
    target_link_options(${TARGET6} PRIVATE "LINKER:/MANIFESTINPUT:${CMAKE_CURRENT_SOURCE_DIR}/utf8.manifest")
endif()
target_link_libraries(${TARGET6} PRIVATE llama ggml ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET6} PRIVATE cxx_std_17)

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
