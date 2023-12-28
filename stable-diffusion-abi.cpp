#include "stable-diffusion-abi.h"

#include "stable-diffusion.h"
#include "util.h"
#include <string>
#include <cstring>
#include <map>
#include <algorithm>
#include <iterator>
#include <vector>

/*================================================= StableDiffusion ABI API  =============================================*/
const static std::map<std::string, enum SDLogLevel> SDLogLevelMap = {
    {"DEBUG", DEBUG},
    {"INFO", INFO},
    {"WARN", WARN},
    {"ERROR", ERROR},
};

const static std::map<std::string, enum RNGType> RNGTypeMap = {
    {"STD_DEFAULT_RNG", STD_DEFAULT_RNG},
    {"CUDA_RNG", CUDA_RNG},
};

const static std::map<std::string, enum SampleMethod> SampleMethodMap = {
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

const static std::map<std::string, enum Schedule> ScheduleMap = {
    {"DEFAULT", DEFAULT},
    {"DISCRETE", DISCRETE},
    {"KARRAS", KARRAS},
    {"N_SCHEDULES", N_SCHEDULES},
};

const static std::map<std::string, enum ggml_type> ggmlTypeMap = {
    {"DEFAULT", GGML_TYPE_COUNT},
    {"F32", GGML_TYPE_F32},
    {"F16", GGML_TYPE_F16},
    {"Q4_0", GGML_TYPE_Q4_0},
    {"Q4_1", GGML_TYPE_Q4_1},
    {"Q5_0", GGML_TYPE_Q5_0},
    {"Q5_1", GGML_TYPE_Q5_1},
    {"Q8_0", GGML_TYPE_Q8_0},
};

void stable_diffusion_full_params_set_negative_prompt(
    struct stable_diffusion_full_params* params,
    const char* negative_prompt
) {
    params->negative_prompt.assign(negative_prompt);
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
        0.0f,
        0,
        0,
        EULER_A,
        0,
        0,
        1,
        0.0f
    };
}

struct stable_diffusion_ctx {
    StableDiffusion* sd;
};

struct stable_diffusion_ctx* stable_diffusion_init(
    const int n_threads,
    const bool vae_decode_only,
    const char * taesd_path,
    const char * esrgan_path,
    const bool free_params_immediately,
    const bool vae_tiling,
    const char* lora_model_dir,
    const char* rng_type
) {
    const auto s = std::string(rng_type);
    const auto it = RNGTypeMap.find(s);
    const auto ctx = new stable_diffusion_ctx{};
    if (it != RNGTypeMap.end()) {
        const auto sd = new StableDiffusion(
            n_threads,
            vae_decode_only,
            std::string(taesd_path),
            std::string(esrgan_path),
            free_params_immediately,
            vae_tiling,
            std::string(lora_model_dir),
            it->second
        );
        ctx->sd = sd;
    }
    return ctx;
};


bool stable_diffusion_load_from_file(
    const struct stable_diffusion_ctx* ctx,
    const char* file_path,
    const char* vae_path,
    const char* wtype,
    const char* schedule,
    const int clip_skip
) {
     auto e_wtype=ggmlTypeMap.find(std::string(wtype));
    if (e_wtype!=ggmlTypeMap.end()){
        e_wtype=ggmlTypeMap.find("DEFAULT");
    }

    const auto e_schedule = ScheduleMap.find(std::string(schedule));
    if (e_schedule != ScheduleMap.end()) {
        return ctx->sd->load_from_file(
                std::string(file_path),
                std::string(vae_path),
                e_wtype->second ,
                e_schedule->second,
                clip_skip
                );
    }
    return false;
};

const uint8_t* stable_diffusion_predict_image(
    const struct stable_diffusion_ctx* ctx,
    const struct stable_diffusion_full_params* params,
    const char* prompt
) {
    const auto result = ctx->sd->txt2img(
        std::string(prompt),
        params->negative_prompt,
        params->cfg_scale,
        params->width,
        params->height,
        params->sample_method,
        params->sample_steps,
        params->seed,
        params->batch_count
    );
    const auto image_size = params->width * params->height * 3;
    std::vector<uint8_t> images;
    images.reserve(image_size * params->batch_count);
    for (const auto img: result) {
        if (img != nullptr) {
            std::copy_n(img, image_size, std::back_inserter(images));
        }
    };
    const auto buffer = new uint8_t[image_size * params->batch_count];
    std::memcpy(buffer, images.data(), images.size());
    return buffer;
};

const uint8_t* stable_diffusion_image_predict_image(
    const struct stable_diffusion_ctx* ctx,
    const struct stable_diffusion_full_params* params,
    const uint8_t* init_image,
    const char* prompt
) {
    const auto result = ctx->sd->img2img(
        init_image,
        std::string(prompt),
        params->negative_prompt,
        params->cfg_scale,
        params->width,
        params->height,
        params->sample_method,
        params->sample_steps,
        params->strength,
        params->seed
    );
    const auto image_size = params->width * params->height * 3;
    std::vector<uint8_t> images;
    images.reserve(image_size);
    for (const auto img: result) {
        if (img != nullptr) {
            std::copy_n(img, image_size, std::back_inserter(images));
        }
    };
    const auto buffer = new uint8_t[image_size];
    std::memcpy(buffer, images.data(), images.size());
    return buffer;
};

void stable_diffusion_set_log_level(const char* level) {
    const auto ll = std::string(level);
    const auto it = SDLogLevelMap.find(ll);
    if (it != SDLogLevelMap.end()) {
        set_sd_log_level(it->second);
    }
};

const char* stable_diffusion_get_system_info() {
    const std::string info = sd_get_system_info();
    const size_t length = info.size() + 1;
    const auto buffer = new char[length];
    std::memcpy(buffer, info.c_str(), length);
    return buffer;
};

void stable_diffusion_free(struct stable_diffusion_ctx* ctx) {
    if (ctx!= nullptr){
        if (ctx->sd!= nullptr){
            delete ctx->sd;
            ctx->sd= nullptr;
        }
        delete ctx;
        ctx = nullptr;
    }
};

void stable_diffusion_free_full_params( struct stable_diffusion_full_params* params) {
    if (params!= nullptr){
        delete params;
        params = nullptr;
    }
}

void stable_diffusion_free_buffer(uint8_t* buffer) {
    if (buffer!= nullptr){
        delete [] buffer;
        buffer= nullptr;
    }
}
