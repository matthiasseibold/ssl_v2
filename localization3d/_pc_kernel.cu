/*
 * _pc_kernel.cu — GPU point-cloud processing kernel for visualize_bboxes.py
 *
 * One CUDA thread per ZED pixel.  Replaces the following CPU numpy pipeline:
 *   isfinite filter → ZED→OCV coord transform → pinhole projection →
 *   webcam ROI check → heatmap colour blend
 *
 * Compiled with:
 *   nvcc -arch=sm_75 --shared -Xcompiler -fPIC -O3 -o _pc_kernel.so _pc_kernel.cu
 *
 * Device pointers are passed as unsigned long long so that Python ctypes can
 * hand them in without needing a ctypes pointer type per array dtype.
 *
 * Invalid points (NaN/Inf depth) have NaN written to xyz_out so that Open3D
 * silently skips them — no separate valid[] buffer or CPU compaction needed.
 */

#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------

__global__ void _process_pc_kernel(
    const float*   xyzrgba,   // (N, 4) flat: X, Y, Z, RGBA-packed
    const uint8_t* bgra,      // (N, 4) flat: B, G, R, A  (ZED VIEW.LEFT)
    const uint8_t* heatmap,   // (H_wc * W_wc * 3) flat: B, G, R  (OpenCV BGR)
    int H_wc, int W_wc,
    const float* R,           // (9,) row-major webcam rotation  (world→cam)
    const float* t,           // (3,) webcam translation
    double fx, double fy, double cx, double cy,
    int roi_x0, int roi_y0, int roi_x1, int roi_y1,
    float blend, int show_heatmap,
    float*   xyz_out,         // (N * 3) flat: OCV-frame positions (NaN if invalid)
    float*   rgb_out,         // (N * 3) flat: RGB colours in [0, 1]
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float x = xyzrgba[i * 4 + 0];
    float y = xyzrgba[i * 4 + 1];
    float z = xyzrgba[i * 4 + 2];

    // Validity: ZED marks missing depth as NaN; also reject ±inf.
    // Write NaN to xyz_out so the caller can pass the full array to Open3D
    // without a separate CPU compaction step.
    if (!isfinite(x) || !isfinite(y) || !isfinite(z)) {
        xyz_out[i * 3 + 0] = NAN;
        xyz_out[i * 3 + 1] = NAN;
        xyz_out[i * 3 + 2] = NAN;
        return;
    }

    // ZED RIGHT_HANDED_Z_UP → OpenCV: x→x, y→−z, z→y
    float xo = x, yo = -z, zo = y;
    xyz_out[i * 3 + 0] = xo;
    xyz_out[i * 3 + 1] = yo;
    xyz_out[i * 3 + 2] = zo;

    // ZED VIEW.LEFT is BGRA: channel 0=B, 1=G, 2=R.
    const float inv255 = 1.0f / 255.0f;
    float r_f = bgra[i * 4 + 2] * inv255;
    float g_f = bgra[i * 4 + 1] * inv255;
    float b_f = bgra[i * 4 + 0] * inv255;
    rgb_out[i * 3 + 0] = r_f;
    rgb_out[i * 3 + 1] = g_f;
    rgb_out[i * 3 + 2] = b_f;

    if (!show_heatmap) return;

    // Pinhole projection: OCV-frame point → webcam pixel (u, v).
    float px = R[0]*xo + R[1]*yo + R[2]*zo + t[0];
    float py = R[3]*xo + R[4]*yo + R[5]*zo + t[1];
    float pz = R[6]*xo + R[7]*yo + R[8]*zo + t[2];
    if (pz <= 0.0f) return;

    int u = (int)(fx * px / pz + cx);
    int v = (int)(fy * py / pz + cy);
    if (u < roi_x0 || u >= roi_x1 || v < roi_y0 || v >= roi_y1) return;

    // Heatmap lookup (OpenCV BGR channel order).
    int hm_idx = (v * W_wc + u) * 3;
    uint8_t hm_b = heatmap[hm_idx + 0];
    uint8_t hm_g = heatmap[hm_idx + 1];
    uint8_t hm_r = heatmap[hm_idx + 2];
    if (hm_b >= 224 && hm_g >= 224 && hm_r >= 224) return;  // near-white → keep ZED colour

    // Blend heatmap colour over ZED colour; output remains RGB.
    float inv_blend = 1.0f - blend;
    rgb_out[i * 3 + 0] = blend * hm_r * inv255 + inv_blend * r_f;
    rgb_out[i * 3 + 1] = blend * hm_g * inv255 + inv_blend * g_f;
    rgb_out[i * 3 + 2] = blend * hm_b * inv255 + inv_blend * b_f;
}

// ---------------------------------------------------------------------------
// C-callable launcher (extern "C" so ctypes can find it by name)
// Device pointers are passed as unsigned long long (64-bit CUDA device address).
// ---------------------------------------------------------------------------

extern "C" int launch_process_pc(
    unsigned long long xyzrgba_ptr,
    unsigned long long bgra_ptr,
    unsigned long long heatmap_ptr,
    int H_wc, int W_wc,
    unsigned long long R_ptr,
    unsigned long long t_ptr,
    double fx, double fy, double cx, double cy,
    int roi_x0, int roi_y0, int roi_x1, int roi_y1,
    float blend, int show_heatmap,
    unsigned long long xyz_out_ptr,
    unsigned long long rgb_out_ptr,
    int N, int grid_size, int block_size
) {
    _process_pc_kernel<<<grid_size, block_size>>>(
        reinterpret_cast<const float*>  (xyzrgba_ptr),
        reinterpret_cast<const uint8_t*>(bgra_ptr),
        reinterpret_cast<const uint8_t*>(heatmap_ptr),
        H_wc, W_wc,
        reinterpret_cast<const float*>  (R_ptr),
        reinterpret_cast<const float*>  (t_ptr),
        fx, fy, cx, cy,
        roi_x0, roi_y0, roi_x1, roi_y1,
        blend, show_heatmap,
        reinterpret_cast<float*>  (xyz_out_ptr),
        reinterpret_cast<float*>  (rgb_out_ptr),
        N
    );
    return static_cast<int>(cudaGetLastError());
}
