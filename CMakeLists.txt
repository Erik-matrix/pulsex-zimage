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
