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

void sd_images_free(const sd_image_t* images) {
    if (images != nullptr) {
        delete []images;
    }
    images = nullptr;
}

void sd_image_free(sd_image_t* image) {
    if (image != nullptr) {
        delete image;
    }
    image = nullptr;
}

// sd_image_t new_image() {
//
// }
//
// void set_image_data(sd_image_t image, uint8_t* data) {
//     image.data = data;
// }
//
// void set_image_width(sd_image_t image, uint32_t width) {
//     image.width = width;
// }
//
// void set_image_height(sd_image_t image, uint32_t height) {
//     imageheight = height;
// }
//
// void set_image_channel(sd_image_t image, uint32_t channel) {
//     image->channel = channel;
// }
