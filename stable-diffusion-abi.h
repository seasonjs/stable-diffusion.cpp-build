#ifndef STABLE_DIFFUSION_ABI_H
#define STABLE_DIFFUSION_ABI_H

#include "stable-diffusion.h"

#ifdef STABLE_DIFFUSION_SHARED
#if defined(_WIN32) && !defined(__MINGW32__)
#ifdef STABLE_DIFFUSION_BUILD
    #define STABLE_DIFFUSION_API __declspec(dllexport)
#else
    #define STABLE_DIFFUSION_API __declspec(dllimport)
#endif
#else
#define STABLE_DIFFUSION_API __attribute__((visibility("default")))
#endif
#else
    #define STABLE_DIFFUSION_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

STABLE_DIFFUSION_API uint8_t* get_image_data(const sd_image_t* images, int index);

STABLE_DIFFUSION_API uint32_t get_image_width(const sd_image_t* images, int index);

STABLE_DIFFUSION_API uint32_t get_image_height(const sd_image_t* images, int index);

STABLE_DIFFUSION_API uint32_t get_image_channel(const sd_image_t* images, int index);

#ifdef __cplusplus
}
#endif

#endif //STABLE_DIFFUSION_ABI_H
