#include "stable-diffusion-abi.h"

#include "stable-diffusion.h"
#include "base64.hpp"
#include <string>
#include <cstring>
#include <map>
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

// Use setter to handle purego max args limit less than 9
// see https://github.com/ebitengine/purego/pull/7
//     https://github.com/ebitengine/purego/blob/4db9e9e813d0f24f3ccc85a843d2316d2d2a70c6/func.go#L104
struct sd_txt2img_options {
    const char* prompt;
    const char* negative_prompt;
    float cfg_scale;
    int width;
    int height;
    const char* sample_method;
    int sample_steps;
    int64_t seed;
    int batch_count;
};

struct sd_img2img_options {
    uint8_t* init_img;
    const char* prompt;
    const char* negative_prompt;
    float cfg_scale;
    int width;
    int height;
    const char* sample_method;
    int sample_steps;
    float strength;
    int64_t seed;
};

sd_txt2img_options* new_sd_txt2img_options() {
    const auto opt = new sd_txt2img_options{};
    return opt;
};

sd_img2img_options* new_sd_img2img_options() {
    const auto opt = new sd_img2img_options{};
    return opt;
};

// Implementation for txt2img options setters
void set_txt2img_prompt(sd_txt2img_options* opt, const char* prompt) {
    opt->prompt = prompt;
}

void set_txt2img_negative_prompt(sd_txt2img_options* opt, const char* negative_prompt) {
    opt->negative_prompt = negative_prompt;
}

void set_txt2img_cfg_scale(sd_txt2img_options* opt, const float cfg_scale) {
    opt->cfg_scale = cfg_scale;
}

void set_txt2img_size(sd_txt2img_options* opt, const int width, const int height) {
    opt->width = width;
    opt->height = height;
}

void set_txt2img_sample_method(sd_txt2img_options* opt, const char* sample_method) {
    opt->sample_method = sample_method;
}

void set_txt2img_sample_steps(sd_txt2img_options* opt, const int sample_steps) {
    opt->sample_steps = sample_steps;
}

void set_txt2img_seed(sd_txt2img_options* opt, const int64_t seed) {
    opt->seed = seed;
}

void set_img2img_init_img(sd_img2img_options* opt, const char* init_img) {
    const auto init_image = code::base64_decode<std::vector<uint8_t>, std::string>(init_img);
    opt->init_img = new uint8_t[init_image.size()];
    std::memcpy(opt->init_img, init_image.data(), init_image.size() * sizeof(uint8_t));
}

void set_img2img_prompt(sd_img2img_options* opt, const char* prompt) {
    opt->prompt = prompt;
}

void set_img2img_negative_prompt(sd_img2img_options* opt, const char* negative_prompt) {
    opt->negative_prompt = negative_prompt;
}

void set_img2img_cfg_scale(sd_img2img_options* opt, const float cfg_scale) {
    // Assuming cfg_scale is a floating point number in string format
    opt->cfg_scale = cfg_scale;
}

void set_img2img_size(sd_img2img_options* opt, const int width, const int height) {
    opt->width = width;
    opt->height = height;
}

void set_img2img_sample_method(sd_img2img_options* opt, const char* sample_method) {
    opt->sample_method = sample_method;
}

void set_img2img_sample_steps(sd_img2img_options* opt, const int sample_steps) {
    opt->sample_steps = sample_steps;
}

void set_img2img_strength(sd_img2img_options* opt, const float strength) {
    // Assuming strength is a floating point number
    opt->strength = strength;
}

void set_img2img_seed(sd_img2img_options* opt, const int64_t seed) {
    opt->seed = seed;
}

void* create_stable_diffusion(
    const int n_threads,
    const bool vae_decode_only,
    const bool free_params_immediately,
    const char* lora_model_dir,
    const char* rng_type
) {
    const auto s = std::string(rng_type);
    const auto it = RNGTypeMap.find(s);
    if (it != RNGTypeMap.end()) {
        return new StableDiffusion(
            n_threads,
            vae_decode_only,
            free_params_immediately,
            std::string(lora_model_dir),
            it->second
        );
    }
    return nullptr;
};

void destroy_stable_diffusion(void* sd) {
    const auto s = static_cast<StableDiffusion *>(sd);
    delete s;
};

bool load_from_file(void* sd, const char* file_path, const char* schedule) {
    const auto s = static_cast<StableDiffusion *>(sd);
    const auto sc = std::string(schedule);
    const auto it = ScheduleMap.find(sc);
    if (it != ScheduleMap.end()) {
        return s->load_from_file(std::string(file_path), it->second);
    }
    return false;
};

const char* txt2img(void* sd, const sd_txt2img_options* opt) {
    const auto sm = std::string(opt->sample_method);
    const auto it = SampleMethodMap.find(sm);
    if (it != SampleMethodMap.end()) {
        const auto s = static_cast<StableDiffusion *>(sd);
        const auto result = s->txt2img(
            std::string(opt->prompt),
            std::string(opt->negative_prompt),
            opt->cfg_scale,
            opt->width,
            opt->height,
            it->second,
            opt->sample_steps,
            opt->seed,
            opt->batch_count
        );
        const auto str = code::base64_encode<std::string, std::vector<uint8_t *>>(result, false);
        const auto buffer = new char[str.size()];
        std::memcpy(buffer, str.c_str(), str.size());
        return buffer;
    }
    delete opt;
    return nullptr;
};

const char* img2img(void* sd, const sd_img2img_options* opt) {
    const auto sm = std::string(opt->sample_method);
    const auto it = SampleMethodMap.find(sm);
    if (it != SampleMethodMap.end()) {
        const auto s = static_cast<StableDiffusion *>(sd);
        const auto result = s->img2img(
            /* const std::vector<uint8_t>& init_img */ opt->init_img,
                                                       /* const std::string &prompt */ std::string(opt->prompt),
                                                       /* const std::string &negative_prompt */
                                                       std::string(opt->negative_prompt),
                                                       /* float cfg_scale */ opt->cfg_scale,
                                                       /* int width */ opt->width,
                                                       /* int height */ opt->height,
                                                       /* SampleMethod sample_method */ it->second,
                                                       /* int sample_steps */ opt->sample_steps,
                                                       /* float strength */ opt->strength,
                                                       /* int64_t seed */ opt->seed
        );
        const auto str = code::base64_encode<std::string, std::vector<uint8_t *>>(result, false);
        const auto buffer = new char[str.size()];
        std::memcpy(buffer, str.c_str(), str.size());
        return buffer;
    }
    delete opt;
    return nullptr;
};

void set_stable_diffusion_log_level(const char* level) {
    const auto ll = std::string(level);
    const auto it = SDLogLevelMap.find(ll);
    if (it != SDLogLevelMap.end()) {
        set_sd_log_level(it->second);
    }
};

const char* get_stable_diffusion_system_info() {
    const std::string info = sd_get_system_info();
    const size_t length = info.size() + 1;
    const auto buffer = new char[length];
    std::memcpy(buffer, info.c_str(), length);
    return buffer;
};

void free_buffer(const char* buffer) {
    delete [] buffer;
}
