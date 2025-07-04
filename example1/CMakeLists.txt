add_executable(vector_operations main.cpp cuda.cu simd.cpp)

# target_link_libraries(vector_operations PRIVATE CUDA::cudart)

target_include_directories(vector_operations PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_compile_options(vector_operations PRIVATE -march=native)
