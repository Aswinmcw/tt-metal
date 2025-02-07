// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

//
// This header contains tile conversion functions used in tests on the host.
//

#pragma once

#include <cstdint>
#include <vector>
#include "common/assert.hpp"
#include "tt_metal/third_party/tracy/public/tracy/Tracy.hpp"
#include "math.hpp"

using namespace std;
enum TensorLayout {
    LIN_ROW_MAJOR = 0, // standard element-wise row-major
    TILED32_SWIZZLED = 1, // row-major of tiles 32x32, each tile is row-major-swizzled
    TILED32_4FACES = 2,  // rowm major of tiles 32x32, each tile is 4 faces, each face is row-major, faces are swizzled
};

template <class T, template<typename> typename BufferType>
std::vector<T> convert_to_tile_layout(const BufferType<T>& data) {
    ZoneScoped;
    std::vector<T> result;
    TT_ASSERT(data.size() % (32 * 32) == 0);
    int num_tiles = data.size() / (32 * 32);
    for(int tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        std::vector<T> top_left;
        std::vector<T> top_right;
        std::vector<T> bottom_left;
        std::vector<T> bottom_right;

        int index = tile_idx * (32 * 32);
        for(int row = 0; row < 32; row++) {
            for(int col = 0; col < 32; col++) {
                if(row < 16 and col < 16) {
                    top_left.push_back(data[index]);
                } else if(row < 16 and col >= 16) {
                    top_right.push_back(data[index]);
                } else if(row >= 16 and col < 16) {
                    bottom_left.push_back(data[index]);
                } else if(row >= 16 and col >= 16) {
                    bottom_right.push_back(data[index]);
                } else {
                    TT_ASSERT(false);
                }
                index++;
            }
        }
        TT_ASSERT(top_left.size() == 16 * 16);
        TT_ASSERT(top_right.size() == 16 * 16);
        TT_ASSERT(bottom_left.size() == 16 * 16);
        TT_ASSERT(bottom_right.size() == 16 * 16);

        result.insert(result.end(), top_left.begin(), top_left.end());
        result.insert(result.end(), top_right.begin(), top_right.end());
        result.insert(result.end(), bottom_left.begin(), bottom_left.end());
        result.insert(result.end(), bottom_right.begin(), bottom_right.end());
    }

    return result;
}

template <class T, template<typename> typename BufferTyp>
std::vector<T> convert_to_flat_layout(const BufferTyp<T>& data) {
    ZoneScoped;
    std::vector<T> result;
    TT_ASSERT(data.size() % (32 * 32) == 0);
    int num_tiles = data.size() / (32 * 32);
    for(int tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        int tile_start = tile_idx * (32 * 32);
        for(int face_y = 0; face_y < 2; face_y++) {
            for(int row = 0; row < 16; row++) {
                int start = tile_start + face_y * (16 * 32) + row * 16;
                for(int face_x = 0; face_x < 2; face_x++) {
                    int offset = face_x * (16 * 16);
                    for(int col = offset; col < offset + 16; col++) {
                        result.push_back(data[start + col]);
                    }
                }
            }
        }
    }

    return result;
}


// Converts a 32-swizzled tilized row-major tensor to a linear 32-zero-padded row-major tensor
template<typename T, template<typename> typename BufferType>
inline std::vector<T> untilize_nchw(const BufferType<T>& in, const std::vector<std::uint32_t>& shape) {
    ZoneScoped;
    TT_ASSERT(shape.size() == 4);
    TT_ASSERT(shape[2] % 32 == 0 && shape[3] % 32 == 0);

    std::vector<T> result;
    // Untilize into row major
    int N = shape[0], C = shape[1], H = shape[2], W = shape[3];
    result.resize(N*C*H*W);
    uint32_t linear = 0;
    for (int n = 0; n < N; n++)
    for (int c = 0; c < C; c++)
    for (int hs32 = 0; hs32 < H; hs32 += 32) // iterate over h with stride 32
    for (int ws32 = 0; ws32 < W; ws32 += 32) // iterate over w with stride 32
    for (int h32 = 0; h32 < 32; h32++) // hs32 + h32 = h
    for (int w32 = 0; w32 < 32; w32++) { // ws32 + w32 = w
        T val = in[linear];
        auto w = w32 + ws32;
        auto h = h32 + hs32;
        auto offs = w + h*W + c*H*W + n*C*H*W;
        result[offs] = val;
        linear ++;
    }

    return result;
}

inline std::uint32_t round_up_to_mul32(std::uint32_t val) { return ((val & 31) == 0) ? val : (val | 31)+1; }

// Converts a linear non-zero-padded row-major tensor to zero-padded-32 32-swizzled tilized row-major tensor
template<typename T, template<typename> typename BufferType>
inline std::vector<T> tilize_nchw(const BufferType<T>& in_rowmajor, const std::vector<std::uint32_t>& shape) {
    ZoneScoped;
    int N = shape[0], C = shape[1], H = shape[2], W = shape[3];
    int NCHW = N*C*H*W;
    int OH = round_up_to_mul32(H);
    int OW = round_up_to_mul32(W);
    std::vector<T> tilized_result;
    tilized_result.resize(N*C*OH*OW);
    std::fill(tilized_result.begin(), tilized_result.end(), 0);
    int out_index = 0;
    for (int n = 0; n < N; n++)
    for (int c = 0; c < C; c++)
    for (int hs32 = 0; hs32 < H; hs32 += 32)
    for (int ws32 = 0; ws32 < W; ws32 += 32)
    for (int h32 = 0; h32 < 32; h32++)
    for (int w32 = 0; w32 < 32; w32++) {
        auto w = w32+ws32;
        auto h = h32+hs32;
        auto in_offs = w + h*W + c*H*W + n*C*H*W;
        auto val = (w >= W || h >= H || c >= C || in_offs >= NCHW) ? 0 : in_rowmajor[in_offs];
        int out_w = (out_index % OW);
        int out_h = (out_index / OW) % OH;
        int out_c = (out_index / (OH*OW)) % C;
        int out_n = (out_index / (C*OH*OW));
        TT_ASSERT(w < OW);
        TT_ASSERT(h < OH);
        TT_ASSERT(out_n < N);
        int out_offs = out_w + out_h*OW + out_c*OH*OW + out_n*C*OH*OW;
        tilized_result[out_offs] = val;
        out_index++;
    }
    TT_ASSERT(tilized_result.size() == N*C*OH*OW);

    return tilized_result;
}

struct TensAddr {
    vector<std::uint32_t> sh;

    std::uint32_t numel() const {
        std::uint32_t prod = 1;
        for (int j = 0; j < sh.size(); j ++)
            prod *= sh[j];
        return prod;
    }

    TensAddr(vector<std::uint32_t> shape) : sh(shape) {}
    int offs(int n, int c, int h, int w) {
        TT_ASSERT(std::uint32_t(n) < sh[0] && std::uint32_t(c) < sh[1] && std::uint32_t(h) < sh[2] && std::uint32_t(w) < sh[3]);
        return w + sh[3]*h + sh[2]*sh[3]*c + sh[1]*sh[2]*sh[3]*n;
    }
};

template<typename T, template<typename> typename BufferType>
inline vector<T> convert_layout(const BufferType<T>& inp, const vector<uint32_t>& shape, TensorLayout inL, TensorLayout outL) {
    ZoneScoped;
    switch (inL) {
        case TILED32_SWIZZLED:
            if (outL == TILED32_4FACES) {
                return convert_to_tile_layout<T>(inp);
            } else if (outL == LIN_ROW_MAJOR) {
                return untilize_nchw<T>(inp, shape);
            } else
                TT_ASSERT(false && "Unsupported conversion.");
        break;
        case LIN_ROW_MAJOR:
            if (outL == TILED32_SWIZZLED) {
                return tilize_nchw<T>(inp, shape);
            } else if (outL == TILED32_4FACES) {
                auto swiz32 = convert_layout<T>(inp, shape, inL, TILED32_SWIZZLED);
                return convert_layout<T>(swiz32, shape, TILED32_SWIZZLED, outL);
            } else
                TT_ASSERT(false && "Unsupported conversion.");
        break;
        case TILED32_4FACES:
            if (outL == TILED32_SWIZZLED) {
                return convert_to_flat_layout<T>(inp);
            } else if (outL == LIN_ROW_MAJOR) {
                auto swiz32 = convert_layout<T>(inp, shape, inL, TILED32_SWIZZLED);
                return untilize_nchw<T>(swiz32, shape);
            } else {
                TT_ASSERT(false && "Unsupported conversion");
            }
        break;
        default:
            TT_ASSERT(false && "Unsupported conversion");
    }
    return vector<T>();
}
