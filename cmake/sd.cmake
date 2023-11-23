include(FetchContent)
# FetchContent_MakeAvailable was not added until CMake 3.14
if(${CMAKE_VERSION} VERSION_LESS 3.14)
    include(add_FetchContent_MakeAvailable.cmake)
endif()

set(SD_GIT_TAG  46eacf7fa133e98df6f51b769e4bb68a48ff3fa3)
set(SD_GIT_URL  https://github.com/Cyberhan123/stable-diffusion.cpp)

FetchContent_Declare(
  sd
  GIT_REPOSITORY    ${SD_GIT_URL}
  GIT_TAG           ${SD_GIT_TAG}
)

FetchContent_MakeAvailable(sd)