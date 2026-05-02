cmake_minimum_required(VERSION 3.16)

project(cipher_memdef_core LANGUAGES C CXX)

find_package(LLVM 14 REQUIRED CONFIG)

add_library(cipher_memdef_store_ir MODULE passes/MemoryWritePatchIR.cpp)
target_compile_features(cipher_memdef_store_ir PRIVATE cxx_std_17)
target_include_directories(cipher_memdef_store_ir PRIVATE ${LLVM_INCLUDE_DIRS})
target_compile_definitions(cipher_memdef_store_ir PRIVATE ${LLVM_DEFINITIONS})
if(NOT LLVM_ENABLE_RTTI)
  target_compile_options(cipher_memdef_store_ir PRIVATE -fno-rtti)
endif()
set_target_properties(cipher_memdef_store_ir PROPERTIES
  PREFIX ""
  OUTPUT_NAME "cipher_memdef_store_ir"
)

add_library(cipher_memdef_runtime STATIC runtime/cipher_memdef_runtime.c)
target_compile_features(cipher_memdef_runtime PRIVATE c_std_11)
