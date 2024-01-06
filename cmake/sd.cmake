include(FetchContent)
# FetchContent_MakeAvailable was not added until CMake 3.14
if(${CMAKE_VERSION} VERSION_LESS 3.14)
    include(add_FetchContent_MakeAvailable.cmake)
endif()

set(SD_GIT_TAG  2b6ec97fe244d03c40aa8d70131d40bb086099b0)
set(SD_GIT_URL  https://github.com/leejet/stable-diffusion.cpp)
#set(BUILD_SHARED_LIBS OFF)

FetchContent_Declare(
  sd
  GIT_REPOSITORY    ${SD_GIT_URL}
  GIT_TAG           ${SD_GIT_TAG}
)
FetchContent_MakeAvailable(sd)

set(GGML_AVX512 OFF)
set(GGML_AVX2 OFF)
set(GGML_AVX OFF)
