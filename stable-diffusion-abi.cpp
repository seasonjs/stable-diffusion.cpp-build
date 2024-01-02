#include "stable-diffusion-abi.h"
#include "stable-diffusion.h"
#include <string>

uint8_t* get_image_data(const sd_image_t* images, int index) {
    return images[index].data;
}

uint32_t get_image_width(const sd_image_t* images, int index) {
    return images[index].width;
}

uint32_t get_image_height(const sd_image_t* images, int index) {
    return images[index].height;
}

uint32_t get_image_channel(const sd_image_t* images, int index) {
    return images[index].channel;
}
