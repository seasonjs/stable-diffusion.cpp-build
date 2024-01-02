#ifndef STABLE_DIFFUSION_ABI_H
#define STABLE_DIFFUSION_ABI_H

#include "stable-diffusion.h"
#include <string>

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
struct sd_ctx_t;

struct stable_diffusion_full_params {
    const char* negative_prompt;
    int clip_skip;
    float cfg_scale;
    int width;
    int height;
    sample_method_t sample_method;
    int sample_steps;
    float strength;
    int64_t seed;
    int batch_count;
};

// These methods are used in binding in other languages,golang, python,etc.
// Use setter to handle purego max args limit less than 9
// see https://github.com/ebitengine/purego/pull/7
//     https://github.com/ebitengine/purego/blob/4db9e9e813d0f24f3ccc85a843d2316d2d2a70c6/func.go#L104
STABLE_DIFFUSION_API struct stable_diffusion_full_params* stable_diffusion_full_default_params_ref();

STABLE_DIFFUSION_API void stable_diffusion_full_params_set_negative_prompt(
    struct stable_diffusion_full_params* params,
    const char* negative_prompt
);

STABLE_DIFFUSION_API void stable_diffusion_full_params_set_clip_skip(
    struct stable_diffusion_full_params* params,
    int clip_skip
);

void stable_diffusion_full_params_set_cfg_scale(
    struct stable_diffusion_full_params* params,
    float cfg_scale
);

STABLE_DIFFUSION_API void stable_diffusion_full_params_set_width(
    struct stable_diffusion_full_params* params,
    int width
);

STABLE_DIFFUSION_API void stable_diffusion_full_params_set_height(
    struct stable_diffusion_full_params* params,
    int height
);

STABLE_DIFFUSION_API void stable_diffusion_full_params_set_sample_method(
    struct stable_diffusion_full_params* params,
    const char* sample_method
);

STABLE_DIFFUSION_API void stable_diffusion_full_params_set_sample_steps(
    struct stable_diffusion_full_params* params,
    int sample_steps
);


STABLE_DIFFUSION_API void stable_diffusion_full_params_set_seed(
    struct stable_diffusion_full_params* params,
    int64_t seed
);

STABLE_DIFFUSION_API void stable_diffusion_full_params_set_batch_count(
    struct stable_diffusion_full_params* params,
    int batch_count
);

STABLE_DIFFUSION_API void stable_diffusion_full_params_set_strength(
    struct stable_diffusion_full_params* params,
    float strength
);

STABLE_DIFFUSION_API sd_ctx_t* stable_diffusion_init(
    const char* model_path,
    const char* vae_path,
    const char* taesd_path,
    const char* lora_model_dir,
    bool vae_decode_only,
    bool vae_tiling,
    bool free_params_immediately,
    int n_threads,
    const char* wtype,
    const char* rng_type,
    const char* schedule
);

STABLE_DIFFUSION_API const sd_image_t* stable_diffusion_predict_image(
    struct sd_ctx_t* ctx,
    const struct stable_diffusion_full_params* params,
    const char* prompt
);

STABLE_DIFFUSION_API const sd_image_t* stable_diffusion_image_predict_image(
    struct sd_ctx_t* ctx,
    const struct stable_diffusion_full_params* params,
    sd_image_t* init_image,
    const char* prompt
);

STABLE_DIFFUSION_API void stable_diffusion_set_log_callback(sd_log_cb_t sd_log_cb);

STABLE_DIFFUSION_API const char* stable_diffusion_get_system_info();

STABLE_DIFFUSION_API void stable_diffusion_free(struct sd_ctx_t* ctx);

STABLE_DIFFUSION_API void stable_diffusion_free_full_params(struct stable_diffusion_full_params* params);

STABLE_DIFFUSION_API void stable_diffusion_free_buffer(const uint8_t* buffer);

#ifdef __cplusplus
}
#endif

#endif //STABLE_DIFFUSION_ABI_H
