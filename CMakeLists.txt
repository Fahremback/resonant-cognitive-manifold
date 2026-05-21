cmake_minimum_required(VERSION 3.20)

# Passa a flag para permitir compiladores não homologados antes do project() para que o TryCompile do CMake funcione
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")

project(rcm LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Código-fonte
set(SOURCES
    src/main.cpp
    src/ssd_storage.cpp
    src/vram_cache.cu
    src/resonance_engine.cu
    src/projector.cpp
    src/code_tokenizer.cpp
)

# Executável
add_executable(rcm ${SOURCES})

# Adiciona o diretório include
target_include_directories(rcm PRIVATE include)

# Flags específicas para cada compilador usando Generator Expressions para evitar contaminação
if(MSVC)
    target_compile_options(rcm PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/O2>
        $<$<COMPILE_LANGUAGE:CXX>:/W4>
        
        $<$<COMPILE_LANGUAGE:CUDA>:-allow-unsupported-compiler>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/O2>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/EHsc>
    )
else()
    target_compile_options(rcm PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O3>
        $<$<COMPILE_LANGUAGE:CXX>:-Wall>
        $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
        
        $<$<COMPILE_LANGUAGE:CUDA>:-allow-unsupported-compiler>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wextra>
    )
endif()

# Configuração de CUDA target architectures
set_target_properties(rcm PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
