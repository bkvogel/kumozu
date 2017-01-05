/*
 * Copyright (c) 2005-2015, Brian K. Vogel
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are those
 * of the authors and should not be interpreted as representing official policies,
 * either expressed or implied, of the FreeBSD Project.
 *
 */
#include "PlotUtilities.h"

using namespace std;

namespace kumozu {

MatrixF convert_zero_contered_to_rgb(const MatrixF& image_zero_centered,
                                  bool normalize,
                                     std::vector<float> color_pos,
                                     std::vector<float> color_neg) {
    if (image_zero_centered.order() != 2) {
        error_exit("convert_zero_contered_to_rgb(): supplied images matrix is wrong order.");
    }
    const int height = image_zero_centered.extent(0);
    const int width = image_zero_centered.extent(1);
    MatrixF mod_input = image_zero_centered;

    // First normalize the input matrix, if specified:
    if (normalize) {
        float max_abs = 0.0f;
        for (int i = 0; i < image_zero_centered.size(); ++i) {
            if (std::abs(image_zero_centered[i]) > max_abs) {
                max_abs = std::abs(image_zero_centered[i]);
            }
        }
        if (max_abs > 1.0f) {
            scale(mod_input, 1.0f/max_abs);
        }

    }
    MatrixF image_rgb(3, height, width);
    for (int channel = 0; channel < 3; ++channel) {
        for (int r = 0; r < height; ++r) {
            for (int c = 0; c < width; ++c) {
                auto val = image_zero_centered(r,c);
                if (val >= 0) {
                    image_rgb(channel, r, c) = color_pos[channel]*val;
                } else {
                    image_rgb(channel, r, c) = -color_neg[channel]*val;
                }
            }
        }
    }
    return image_rgb;
}

}
