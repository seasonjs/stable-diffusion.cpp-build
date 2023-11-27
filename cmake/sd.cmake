include(FetchContent)
# FetchContent_MakeAvailable was not added until CMake 3.14
if(${CMAKE_VERSION} VERSION_LESS 3.14)
    include(add_FetchContent_MakeAvailable.cmake)
endif()

set(SD_GIT_TAG  1ac4ad2ba21c7f1c96c061f452bde129522a85c1)
set(SD_GIT_URL  https://github.com/Cyberhan123/stable-diffusion.cpp)
set(BUILD_SHARED_LIBS OFF)

FetchContent_Declare(
  sd
  GIT_REPOSITORY    ${SD_GIT_URL}
  GIT_TAG           ${SD_GIT_TAG}
)
FetchContent_MakeAvailable(sd)
