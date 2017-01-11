#include <taichi/visualization/image_buffer.h>
#include <taichi/math/linalg.h>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_FAILURE_USERMSG
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
#define STB_TRUETYPE_IMPLEMENTATION
#include <stb_truetype.h>

TC_NAMESPACE_BEGIN

template<typename T>
void Array2D<T>::load(const std::string &filename) {
    int channels;
    FILE *f = fopen(filename.c_str(), "rb");
    assert_info(f != nullptr, "Image file not found: " + filename);
    real *data = stbi_loadf(filename.c_str(), &this->width, &this->height, &channels, 0);
    P(stbi_is_hdr(filename.c_str()));
    assert_info(data != nullptr, "Image file load failed: " + filename + " # Msg: " + std::string(stbi_failure_reason()));
    assert_info(channels == 3 || channels == 4, "Image must have channel 3 or 4: " + filename);
    this->initialize(this->width, this->height);
    for (int i = 0; i < this->width; i++) {
        for (int j = 0; j < this->height; j++) {
            real *pixel = data + ((this->height - 1 - j) * this->width + i) * channels;
            (*this)[i][j][0] = pixel[0];
            (*this)[i][j][1] = pixel[1];
            (*this)[i][j][2] = pixel[2];
            if (channels == 4 && same_type<T, Vector4>())
                (*this)[i][j][3] = pixel[3];
        }
    }
    stbi_image_free(data);
}

template<typename T>
void Array2D<T>::write(const std::string &filename)
{
    int comp = 3;
    std::vector<unsigned char> data(this->width * this->height * comp);
    for (int i = 0; i < this->width; i++) {
        for (int j = 0; j < this->height; j++) {
            for (int k = 0; k < comp; k++) {
                data[j * this->width * comp + i * comp + k] =
                        (unsigned char)(255.0f * clamp(this->data[i * this->height + (this->height - j - 1)][k], 0.0f, 1.0f));
            }
        }
    }
    int write_result = stbi_write_png(filename.c_str(), this->width, this->height, comp, &data[0], comp * this->width);
    // assert_info((bool)write_result, "Can not write image file");
}

template<typename T>
void Array2D<T>::write_text(const std::string &font_fn, const std::string &content_, real size,
                                      int dx, int dy) {
    std::vector<unsigned char> buffer(24 << 20, (unsigned char)0);
    std::vector<unsigned char> screen_buffer((size_t)(this->width * this->height), (unsigned char)0);

    static stbtt_fontinfo font;
    int i, j, ascent, baseline, ch = 0;
    float xpos = 2; // leave a little padding in case the character extends left
    static float scale;
    // TODO: cache loaded fonts?
    FILE *font_file = fopen(font_fn.c_str(), "rb");
    assert_info(font_file != nullptr, "Font file not found: " + std::string(font_fn));
    fread(&buffer[0], 1, 24 << 20, font_file);
    stbtt_InitFont(&font, &buffer[0], 0);

    scale = stbtt_ScaleForPixelHeight(&font, size);

    stbtt_GetFontVMetrics(&font, &ascent, 0, 0);
    baseline = (int)(ascent*scale);
    const std::string c_content = content_;
    const char *content = c_content.c_str();
    while (content[ch]) {
        int advance, lsb, x0, y0, x1, y1;
        float x_shift = xpos - (float)floor(xpos);
        stbtt_GetCodepointHMetrics(&font, content[ch], &advance, &lsb);
        stbtt_GetCodepointBitmapBoxSubpixel(&font, content[ch], scale, scale, x_shift, 0, &x0, &y0, &x1, &y1);
        stbtt_MakeCodepointBitmapSubpixel(&font, &screen_buffer[0] + this->width * (baseline + y0) + (int)xpos + x0,
                                          x1 - x0, y1 - y0, 200, scale, scale, x_shift, 0, content[ch]);
        // note that this stomps the old data, so where character boxes overlap (e.g. 'lj') it's wrong
        // because this API is really for baking character bitmaps into textures. if you want to render
        // a sequence of characters, you really need to render each bitmap to a temp buffer, then
        // "alpha blend" that into the working buffer
        xpos += (advance * scale);
        if (content[ch + 1])
            xpos += scale * stbtt_GetCodepointKernAdvance(&font, content[ch], content[ch + 1]);
        ++ch;
    }
    if (dy < 0) {
        dy = this->height + dy - 1;
    }
    for (j = 0; j < this->height; ++j) {
        for (i = 0; i < this->width; ++i) {
            int x = dx + i, y = dy + j;
            float alpha = screen_buffer[(this->height - j - 1) * this->width + i] / 255.0f;
            (*this)[x][y] = lerp(alpha, this->get(x, y), T(1.0f));
        }
    }
}

template
void Array2D<Vector3>::write_text(const std::string &font_fn, const std::string &content_, real size,
                                      int dx, int dy);
template
void Array2D<Vector4>::write_text(const std::string &font_fn, const std::string &content_, real size,
                                      int dx, int dy);

template void Array2D<Vector3>::load(const std::string &filename);
template void Array2D<Vector4>::load(const std::string &filename);
template void Array2D<Vector3>::write(const std::string &filename);
template void Array2D<Vector4>::write(const std::string &filename);

TC_NAMESPACE_END
