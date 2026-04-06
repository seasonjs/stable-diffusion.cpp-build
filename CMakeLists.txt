cmake_minimum_required(VERSION 3.12)

project("stable-diffusion.cpp-build")

include(FetchContent)
include(./cmake/fetch_prebuilt_ggml.cmake)

set(
    SD_SOURCE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/build/_deps/stable-diffusion-src"
    CACHE PATH
    "Optional local stable-diffusion source tree. When present it is preferred over downloading stable-diffusion."
)

set(GGML_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/vendor/ggml" CACHE PATH "Path to an unpacked ggml SDK")
set(GGML_RELEASE_BASE_URL "https://github.com/seasonjs/ggml.cpp-build/releases/download" CACHE STRING "Base release URL, for example https://github.com/seasonjs/ggml.cpp-build/releases/download")
set(GGML_RELEASE_TAG "v0.9.8" CACHE STRING "Release tag containing the prebuilt ggml SDK archives")

# add dependencies
get_filename_component(_sd_local_source_dir "${SD_SOURCE_DIR}" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Ensure non-empty build number for stable-diffusion.cpp versioning

set(STABLE_DIFFUSION_UPSTREAM_TAG "b6d6dfbf" CACHE STRING "Default stable-diffusion build number" FORCE)


# Strategy: ggml dynamic + stable-diffusion static
# Note: We'll use BUILD_SHARED_LIBS=ON to build both as shared initially,
# but stable-diffusion.cpp's CMake allows us to use ggml as dynamic

# Important: Build shared libraries initially (ggml + stable-diffusion)
set(BUILD_SHARED_LIBS ON CACHE BOOL "Build shared libs" FORCE)
set(STABLE_DIFFUSION_USE_SYSTEM_GGML ON CACHE BOOL "Use system ggml" FORCE)

message("check STABLE_DIFFUSION_UPSTREAM_TAG ${STABLE_DIFFUSION_UPSTREAM_TAG}")

add_compile_definitions(GGML_MAX_NAME=128)

if (EXISTS "${GGML_ROOT_DIR}")
    ggml_import_prebuilt(
        ROOT_DIR "${GGML_ROOT_DIR}"
        OUT_ROOT_DIR GGML_PACKAGE_ROOT
    )
elseif (GGML_RELEASE_BASE_URL)
    ggml_import_prebuilt(
        RELEASE_BASE_URL "${GGML_RELEASE_BASE_URL}"
        RELEASE_TAG "${GGML_RELEASE_TAG}"
        OUT_ROOT_DIR GGML_PACKAGE_ROOT
    )
else()
    message(FATAL_ERROR
        "Set GGML_ROOT_DIR to an unpacked ggml SDK or provide GGML_RELEASE_BASE_URL so CMake can download one.")
endif()

if (EXISTS "${_sd_local_source_dir}/CMakeLists.txt")
    message(STATUS "Using local stable-diffusion source tree: ${_sd_local_source_dir}")
    add_subdirectory("${_sd_local_source_dir}" "${CMAKE_BINARY_DIR}/stable-diffusion")
else()
    message(STATUS "Fetching stable-diffusion.cpp ${STABLE_DIFFUSION_UPSTREAM_TAG} from GitHub")
    FetchContent_Declare(
        stable_diffusion
        GIT_REPOSITORY https://github.com/Cyberhan123/stable-diffusion.cpp.git
        GIT_TAG        ${STABLE_DIFFUSION_UPSTREAM_TAG}
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(stable_diffusion)
endif()

# We don't need to copy ggml's runtime binaries to the output directory
# ggml_copy_runtime_binaries(main_app
#     ROOT_DIR "${GGML_PACKAGE_ROOT}"
# )
