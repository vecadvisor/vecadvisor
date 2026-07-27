cmake_minimum_required(VERSION 3.20)

project(vecadvisor_native LANGUAGES CXX)

option(VECADVISOR_NATIVE_BUILD_TESTS "Build native VecAdvisor tests" ON)
option(VECADVISOR_NATIVE_ENABLE_AVX2 "Build AVX2 distance kernels when supported" ON)

add_library(vecadvisor_distance STATIC src/distance.cpp)
target_include_directories(
  vecadvisor_distance
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_features(vecadvisor_distance PUBLIC cxx_std_17)

if(MSVC)
  target_compile_options(vecadvisor_distance PRIVATE /W4 /permissive-)
else()
  target_compile_options(vecadvisor_distance PRIVATE -Wall -Wextra -Wpedantic)
endif()

set(VECADVISOR_AVX2_ENABLED OFF)
string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" VECADVISOR_SYSTEM_PROCESSOR)
set(VECADVISOR_X86_LIKE OFF)
if(VECADVISOR_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|i[3-6]86)$")
  set(VECADVISOR_X86_LIKE ON)
endif()

if(VECADVISOR_NATIVE_ENABLE_AVX2 AND VECADVISOR_X86_LIKE)
  if(MSVC)
    target_sources(vecadvisor_distance PRIVATE src/distance_avx2.cpp)
    set_source_files_properties(src/distance_avx2.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
    set(VECADVISOR_AVX2_ENABLED ON)
  else()
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-mavx2" VECADVISOR_HAS_MAVX2)
    if(VECADVISOR_HAS_MAVX2)
      target_sources(vecadvisor_distance PRIVATE src/distance_avx2.cpp)
      set_source_files_properties(src/distance_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2")
      set(VECADVISOR_AVX2_ENABLED ON)
    endif()
  endif()
endif()

if(VECADVISOR_AVX2_ENABLED)
  target_compile_definitions(vecadvisor_distance PRIVATE VECADVISOR_BUILD_AVX2=1)
else()
  target_compile_definitions(vecadvisor_distance PRIVATE VECADVISOR_BUILD_AVX2=0)
endif()

if(VECADVISOR_NATIVE_BUILD_TESTS)
  enable_testing()

  add_executable(vecadvisor_distance_test tests/distance_test.cpp)
  target_link_libraries(vecadvisor_distance_test PRIVATE vecadvisor_distance)
  target_compile_features(vecadvisor_distance_test PRIVATE cxx_std_17)
  if(MSVC)
    target_compile_options(vecadvisor_distance_test PRIVATE /W4 /permissive-)
  else()
    target_compile_options(vecadvisor_distance_test PRIVATE -Wall -Wextra -Wpedantic)
  endif()

  add_test(NAME vecadvisor_distance_test COMMAND vecadvisor_distance_test)
endif()
