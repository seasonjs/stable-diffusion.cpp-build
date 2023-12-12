cmake_minimum_required(VERSION 3.12)
project("stable-diffusion-build")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
#set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O2")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")


if (NOT XCODE AND NOT MSVC AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif ()

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED true)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake;${CMAKE_MODULE_PATH}")

# Instruction set specific
option(SD_AVX "sd: enable AVX" ON)
option(SD_AVX2 "sd: enable AVX2" ON)
option(SD_AVX512 "sd: enable AVX512" OFF)
option(SD_FMA "sd: enable FMA" ON)

if (SD_AVX512)
    set(GGML_AVX512 ON)
elseif (SD_AVX2)
    set(GGML_AVX2 ON)
elseif (SD_AVX)
    set(GGML_AVX ON)
endif ()
if (SD_FMA)
    set(GGML_FMA ON)
endif ()

include(sd)
include_directories(${sd_SOURCE_DIR})

set(SD_ABI sd-abi)

add_library(${SD_ABI} SHARED
        stable-diffusion-abi.cpp
        stable-diffusion-abi.h)

target_include_directories(${SD_ABI} PUBLIC .)

set_target_properties(zip PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(ggml PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(stable-diffusion PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(${SD_ABI} PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_definitions(${SD_ABI} PRIVATE STABLE_DIFFUSION_SHARED STABLE_DIFFUSION_BUILD)
target_link_libraries(${SD_ABI} PRIVATE stable-diffusion ${CMAKE_THREAD_LIBS_INIT})
