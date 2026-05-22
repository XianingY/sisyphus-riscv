cmake_minimum_required(VERSION 3.16)
project(sisyphus_compiler CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(DEFAULT_TARGET "riscv" CACHE STRING "Default backend target when no --target is specified (riscv|arm)")
set_property(CACHE DEFAULT_TARGET PROPERTY STRINGS riscv arm)

if(NOT DEFAULT_TARGET STREQUAL "riscv" AND NOT DEFAULT_TARGET STREQUAL "arm")
  message(FATAL_ERROR "DEFAULT_TARGET must be 'riscv' or 'arm', got: ${DEFAULT_TARGET}")
endif()

file(GLOB_RECURSE SISYPHUS_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

add_executable(compiler ${SISYPHUS_SOURCES})
target_include_directories(compiler PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")

if(DEFAULT_TARGET STREQUAL "arm")
  target_compile_definitions(compiler PRIVATE DEFAULT_TARGET_ARM=1)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(compiler PRIVATE
    -O2
    -g
    -Wall
    -Wextra
    -Werror=return-type
    -Wno-sign-compare
    -Wno-parentheses
  )
endif()

include(CTest)
if(BUILD_TESTING)
  add_test(
    NAME smoke_compile_dual_target
    COMMAND bash -lc "cd ${CMAKE_CURRENT_SOURCE_DIR} && scripts/run_smoke.sh"
  )
  add_test(
    NAME rvv_vectorize_smoke
    COMMAND bash -lc "cd ${CMAKE_CURRENT_SOURCE_DIR} && scripts/run_vectorize_tests.sh"
  )
  add_test(
    NAME hir_polyhedral_smoke
    COMMAND bash -lc "cd ${CMAKE_CURRENT_SOURCE_DIR} && scripts/run_polyhedral_tests.sh"
  )
endif()
