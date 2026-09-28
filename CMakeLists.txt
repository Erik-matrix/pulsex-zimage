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
