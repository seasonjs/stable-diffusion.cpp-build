include(FetchContent)
# FetchContent_MakeAvailable was not added until CMake 3.14
if(${CMAKE_VERSION} VERSION_LESS 3.14)
    include(add_FetchContent_MakeAvailable.cmake)
endif()

set(SD_GIT_TAG  fb7d9217bb5dd6ac82b4fb1f312381f63b65ba1a)
set(SD_GIT_URL  https://github.com/Cyberhan123/stable-diffusion.cpp)

FetchContent_Declare(
  sd
  GIT_REPOSITORY    ${SD_GIT_URL}
  GIT_TAG           ${SD_GIT_TAG}
)

FetchContent_MakeAvailable(sd)