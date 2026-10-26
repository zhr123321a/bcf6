// Independent reference decoder; no training/autograd logic.
// Keep the vendored header unchanged, with its original license and SHA.
#define BCDEC_IMPLEMENTATION
#include "../third_party/bcdec/bcdec.h"
#include <cstring>

#ifdef _WIN32
#define BCF6_EXPORT __declspec(dllexport)
#else
#define BCF6_EXPORT __attribute__((visibility("default")))
#endif

extern "C" BCF6_EXPORT int bcf6_decode_image(const unsigned char* data, int width, int height, int is_signed, float* output) {
    if (!data || !output || width <= 0 || height <= 0) return 1;
    const int bw = (width + 3) / 4;
    const int bh = (height + 3) / 4;
    for (int by = 0; by < bh; ++by) {
        for (int bx = 0; bx < bw; ++bx) {
            float tile[48];  // 4 x 4 texels x 3 channels, row-major float32.
            // bcdec pitch is in FLOATS: 4 RGB texels = 12, not 48 bytes.
            bcdec_bc6h_float(data + (by * bw + bx) * 16, tile, 12, is_signed);
            for (int y = 0; y < 4 && by*4+y < height; ++y) {
                for (int x = 0; x < 4 && bx*4+x < width; ++x) {
                    std::memcpy(output + ((by*4+y)*width+bx*4+x)*3, tile+(y*4+x)*3, 3*sizeof(float));
                }
            }
        }
    }
    return 0;
}
