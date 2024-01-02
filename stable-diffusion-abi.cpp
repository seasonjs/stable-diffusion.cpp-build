#include "stable-diffusion-abi.h"

#include "stable-diffusion.h"
#include <string>
#include <cstring>
#include <map>

/*================================================= StableDiffusion ABI API  =============================================*/
const static std::map<std::string, enum sd_log_level_t> SDLogLevelMap = {
    {"DEBUG", SD_LOG_DEBUG},
    {"INFO", SD_LOG_INFO},
    {"WARN", SD_LOG_INFO},
    {"ERROR", SD_LOG_ERROR},
};

const static std::map<std::string, enum rng_type_t> RNGTypeMap = {
    {"STD_DEFAULT_RNG", STD_DEFAULT_RNG},
    {"CUDA_RNG", CUDA_RNG},
};

const static std::map<std::string, enum sample_method_t> SampleMethodMap = {
    {"EULER_A", EULER_A},
    {"EULER", EULER},
    {"HEUN", HEUN},
    {"DPM2", DPM2},
    {"DPMPP2S_A", DPMPP2S_A},
    {"DPMPP2M", DPMPP2M},
    {"DPMPP2Mv2", DPMPP2Mv2},
    {"LCM", LCM},
    {"N_SAMPLE_METHODS", N_SAMPLE_METHODS},
};

const static std::map<std::string, enum schedule_t> ScheduleMap = {
    {"DEFAULT", DEFAULT},
    {"DISCRETE", DISCRETE},
    {"KARRAS", KARRAS},
    {"N_SCHEDULES", N_SCHEDULES},
};

const static std::map<std::string, enum sd_type_t> SDTypeMap = {
    {"SD_TYPE_COUNT", SD_TYPE_COUNT},
    {"SD_TYPE_F32", SD_TYPE_F32},
    {"SD_TYPE_F16", SD_TYPE_F16},
    {"SD_TYPE_Q4_0", SD_TYPE_Q4_0},
    {"SD_TYPE_Q4_1", SD_TYPE_Q4_1},
    // SD_TYPE_Q4_2 = 4, support has been removed
    // SD_TYPE_Q4_3 (5) support has been removed
    {"SD_TYPE_Q5_0", SD_TYPE_Q5_0},
    {"SD_TYPE_Q5_1", SD_TYPE_Q5_1},
    {"SD_TYPE_Q8_0", SD_TYPE_Q8_0},
    {"SD_TYPE_Q8_1", SD_TYPE_Q8_1},
    // k-quantizations
    {"SD_TYPE_Q2_K", SD_TYPE_Q2_K},
    {"SD_TYPE_Q3_K", SD_TYPE_Q3_K},
    {"SD_TYPE_Q4_K", SD_TYPE_Q4_K},
    {"SD_TYPE_Q5_K", SD_TYPE_Q5_K},
    {"SD_TYPE_Q6_K", SD_TYPE_Q6_K},
    {"SD_TYPE_Q8_K", SD_TYPE_Q8_K},
    {"SD_TYPE_I8", SD_TYPE_I8},
    {"SD_TYPE_I16", SD_TYPE_I16},
    {"SD_TYPE_I32", SD_TYPE_I32},
};

void stable_diffusion_full_params_set_negative_prompt(
    struct stable_diffusion_full_params* params,
    const char* negative_prompt
) {
    params->negative_prompt = negative_prompt;
}

void stable_diffusion_full_params_set_clip_skip(
    struct stable_diffusion_full_params* params,
    int clip_skip
) {
    params->clip_skip = clip_skip;
}

void stable_diffusion_full_params_set_cfg_scale(
    struct stable_diffusion_full_params* params,
    const float cfg_scale
) {
    params->cfg_scale = cfg_scale;
}

void stable_diffusion_full_params_set_width(
    struct stable_diffusion_full_params* params,
    const int width
) {
    params->width = width;
}

void stable_diffusion_full_params_set_height(
    struct stable_diffusion_full_params* params,
    const int height
) {
    params->height = height;
}

void stable_diffusion_full_params_set_sample_method(
    struct stable_diffusion_full_params* params,
    const char* sample_method
) {
    const auto e_sample_method = SampleMethodMap.find(std::string(sample_method));
    if (e_sample_method != SampleMethodMap.end()) {
        params->sample_method = e_sample_method->second;
    }
}

void stable_diffusion_full_params_set_sample_steps(
    struct stable_diffusion_full_params* params,
    int sample_steps
) {
    params->sample_steps = sample_steps;
}


void stable_diffusion_full_params_set_seed(
    struct stable_diffusion_full_params* params,
    int64_t seed
) {
    params->seed = seed;
}

void stable_diffusion_full_params_set_batch_count(
    struct stable_diffusion_full_params* params,
    int batch_count
) {
    params->batch_count = batch_count;
}

void stable_diffusion_full_params_set_strength(
    struct stable_diffusion_full_params* params,
    float strength
) {
    params->strength = strength;
}

struct stable_diffusion_full_params* stable_diffusion_full_default_params_ref() {
    return new stable_diffusion_full_params{
        "",
        -1,
        0.0f,
        0,
        0,
        EULER_A,
        0,
        0.0f,
        1,
        1
    };
}


struct sd_ctx_t* stable_diffusion_init(
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
) {
    auto e_rng_type = RNGTypeMap.find(std::string(rng_type));
    if (e_rng_type != RNGTypeMap.end()) {
        e_rng_type = RNGTypeMap.find("CUDA_RNG");
    }

    auto e_wtype = SDTypeMap.find(std::string(wtype));
    if (e_wtype != SDTypeMap.end()) {
        e_wtype = SDTypeMap.find("DEFAULT");
    }

    auto e_schedule = ScheduleMap.find(std::string(schedule));

    if (e_schedule != ScheduleMap.end()) {
        e_schedule = ScheduleMap.find("DEFAULT");
    }

    return new_sd_ctx(
        model_path,
        vae_path,
        taesd_path,
        lora_model_dir,
        vae_decode_only,
        vae_tiling,
        free_params_immediately,
        n_threads,
        e_wtype->second,
        e_rng_type->second,
        e_schedule->second
    );
};

const sd_image_t* stable_diffusion_predict_image(
    struct sd_ctx_t* ctx,
    const struct stable_diffusion_full_params* params,
    const char* prompt
) {
    const auto result = txt2img(
        ctx,
        prompt,
        params->negative_prompt,
        params->clip_skip,
        params->cfg_scale,
        params->width,
        params->height,
        params->sample_method,
        params->sample_steps,
        params->seed,
        params->batch_count
    );
    // const auto image_size = params->width * params->height * 3;
    // std::vector<uint8_t> images;
    // images.reserve(image_size * params->batch_count);
    // for (const auto img: result) {
    //     if (img != nullptr) {
    //         std::copy_n(img, image_size, std::back_inserter(images));
    //     }
    // };
    // const auto buffer = new uint8_t[image_size * params->batch_count];
    // std::memcpy(buffer, images.data(), images.size());
    return result;
};

const sd_image_t* stable_diffusion_image_predict_image(
    struct sd_ctx_t* ctx,
    const struct stable_diffusion_full_params* params,
    sd_image_t* init_image,
    const char* prompt
) {
    auto result = img2img(
        ctx,
        *init_image,
        prompt,
        params->negative_prompt,
        params->clip_skip,
        params->cfg_scale,
        params->width,
        params->height,
        params->sample_method,
        params->sample_steps,
        params->strength,
        params->seed,
        params->batch_count
    );

    return result;
};

void stable_diffusion_set_log_callback(sd_log_cb_t sd_log_cb) {
    sd_set_log_callback(sd_log_cb, nullptr);
};

const char* stable_diffusion_get_system_info() {
    const std::string info = sd_get_system_info();
    const size_t length = info.size() + 1;
    const auto buffer = new char[length];
    std::memcpy(buffer, info.c_str(), length);
    return buffer;
};

void stable_diffusion_free(struct sd_ctx_t* ctx) {
    free_sd_ctx(ctx);
};

void stable_diffusion_free_full_params(struct stable_diffusion_full_params* params) {
    if (params != nullptr) {
        delete params;
        params = nullptr;
    }
}

void stable_diffusion_free_buffer(const uint8_t* buffer) {
    if (buffer != nullptr) {
        delete [] buffer;
        buffer = nullptr;
    }
}
