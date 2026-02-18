#include "cubemapencoder.h"

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>

#include <c10/cuda/CUDAGuard.h> // support multiple GPUs

#include <algorithm>
#include <stdexcept>

#include <cstdio>

// OpenGL use left-bottom as origin
// OpenCV use left-top as origin

#define LEFT_TOP_AS_ORIGIN

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be a contiguous tensor")
#define CHECK_IS_INT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Int, #x " must be an int tensor")
#define CHECK_IS_FLOATING(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float || x.scalar_type() == at::ScalarType::Half || x.scalar_type() == at::ScalarType::Double, #x " must be a floating tensor")

__device__ void EdgeTable(int L, int flag, int* index_xy)
{
    int input_face = index_xy[0];
	int input_x = index_xy[1];
	int input_y = index_xy[2];
// #ifdef LEFT_TOP_AS_ORIGIN
	if (input_face == 0) {
		if (flag == 1) { index_xy[0] = 4; index_xy[1] = L - 1; index_xy[2] = input_y; }
		else if (flag == 2) { index_xy[0] = 5; index_xy[1] = 0; index_xy[2] = input_y; }
		else if (flag == 4) { index_xy[0] = 3; index_xy[1] = L - 1; index_xy[2] = input_x; }
		else { index_xy[0] = 2; index_xy[1] = L - 1; index_xy[2] = input_x; }
	}
	else if (input_face == 1) {
		if (flag == 1) { index_xy[0] = 5; index_xy[1] = L - 1; index_xy[2] = input_y; }
		else if (flag == 2) { index_xy[0] = 4; index_xy[1] = 0; index_xy[2] = input_y; }
		else if (flag == 4) { index_xy[0] = 3; index_xy[1] = 0; index_xy[2] = L - 1 - input_x; }
		else { index_xy[0] = 2; index_xy[1] = 0; index_xy[2] = L - 1 - input_x; }
	}
	else if (input_face == 2) {
		if (flag == 1) { index_xy[0] = 1; index_xy[1] = L -1-input_y; index_xy[2] = L-1; }
		else if (flag == 2) { index_xy[0] = 0; index_xy[1] = input_y; index_xy[2] = L-1; }
		else if (flag == 4) { index_xy[0] = 4; index_xy[1] = input_x; index_xy[2] = L - 1; }
		else { index_xy[0] = 5; index_xy[1] = L - 1 - input_x; index_xy[2] = L - 1; }
	}
	else if (input_face == 3) {
		if (flag == 1) { index_xy[0] = 1; index_xy[1] = L-1-input_y; index_xy[2] = 0; }
		else if (flag == 2) { index_xy[0] = 0; index_xy[1] = input_y; index_xy[2] = 0; }
		else if (flag == 4) { index_xy[0] = 4; index_xy[1] = input_x; index_xy[2] = 0; }
		else { index_xy[0] = 5; index_xy[1] = L-1-input_x;  index_xy[2] = 0; }
	}
	else if (input_face == 4) {
		if (flag == 1) { index_xy[0] = 1; index_xy[1] = L - 1; index_xy[2] = input_y; }
		else if (flag == 2) { index_xy[0] = 0; index_xy[1] = 0; index_xy[2] = input_y; }
		else if (flag == 4) { index_xy[0] = 3; index_xy[1] = input_x; index_xy[2] =0; }
		else { index_xy[0] = 2; index_xy[1] = input_x; index_xy[2] = 0; }
	}
	else {
		if (flag == 1) { index_xy[0] = 0; index_xy[1] = L - 1; index_xy[2] = input_y; }
		else if (flag == 2) { index_xy[0] = 1; index_xy[1] = 0; index_xy[2] = input_y; }
		else if (flag == 4) { index_xy[0] = 3; index_xy[1] = L-1-input_x; index_xy[2] = L-1; }
		else { index_xy[0] = 2; index_xy[1] = L - 1 - input_x; index_xy[2] = L-1; }
	}
}

template<typename scalar_t>
__device__ void Compute_Cubemap_UV(scalar_t vx, scalar_t vy, scalar_t vz, scalar_t* uv, int* index)
{
    int max_dim = 0;
    scalar_t x_ = abs(vx);
    scalar_t y_ = abs(vy);
    scalar_t z_ = abs(vz);

    scalar_t max_v = x_;
    if (y_ > max_v)
    {
        max_v = y_;
        max_dim = 1;
    }
    if (z_ > max_v)
    {
        max_v = z_;
        max_dim = 2;
    }

    if (max_dim == 0)
    {
        uv[0] = vz / vx;
        uv[1] = vy / vx;
        // +X
        if (vx >= 0.f)
        {
            *index = 0;
            uv[0] = -uv[0];
            uv[1] = -uv[1];
        }
        // -X
        else
        {
            *index = 1;
            uv[0] = -uv[0];
        }
    }
    else if (max_dim == 1)
    {
        uv[0] = vx / vy;
        uv[1] = vz / vy;
        // +Y
        if (vy >= 0.f)
        {
            *index = 2;
        }
        // -Y
        else
        {
            *index = 3;
            uv[0] = -uv[0];
            uv[1] = -uv[1];
        }
    }
    else
    {
        uv[0] = vx / vz;
        uv[1] = vy / vz;
        // +Z
        if (vz >= 0.f)
        {
            *index = 4;
            uv[1] = -uv[1];
        }
        // -Z
        else
        {
            *index = 5;
        }
    }
}

template<typename scalar_t>
__device__ bool Compute_Seamless_Index(int index, uint32_t L, const scalar_t* uv, int* index_xy, scalar_t* kxky, int* out_flag)
{
    scalar_t loc_uv[2]; // copy uv
	loc_uv[0] = uv[0]; loc_uv[1] = uv[1];
	
	int uy_0, ux_0;
	int uy_1, ux_1;
	scalar_t kx, ky;
	int flag = 0;
	bool is_vertex = false;

// #ifdef LEFT_TOP_AS_ORIGIN
	loc_uv[1] = -loc_uv[1];
// #endif

    // 256 * 256 좌표계로 변환
	loc_uv[0] = (loc_uv[0] * 0.5f + 0.5f) * scalar_t(L);
	loc_uv[1] = (loc_uv[1] * 0.5f + 0.5f) * scalar_t(L);
    // ux, uy -> loc_uv의 위 아래 경계값
	ux_0 = int(floor(loc_uv[0] - 0.5f)); uy_0 = int(floor(loc_uv[1] - 0.5f));
	ux_1 = ux_0 + 1; uy_1 = uy_0 + 1;
    // kx, ky -> 가중치
	kx = loc_uv[0] - scalar_t(ux_0) - 0.5f;
	ky = loc_uv[1] - scalar_t(uy_0) - 0.5f;

    // clamping
	if (ux_0 < 0) ux_0 = 0;
	if (ux_0 >= L) ux_0 = L - 1;
	if (ux_1 < 0) ux_1 = 0;
	if (ux_1 >= L) ux_1 = L - 1;

	if (uy_0 < 0) uy_0 = 0;
	if (uy_0 >= L) uy_0 = L - 1;
	if (uy_1 < 0) uy_1 = 0;
	if (uy_1 >= L) uy_1 = L - 1;

    // 왼쪽
	if (loc_uv[0] < 0.5f) {
		flag = flag | 0x01;
		kx = 0.5f - loc_uv[0]; // 가중치 왜 변화?
	}
    // 오른쪽
	else if (loc_uv[0] >= scalar_t(L) - 0.5f) {
		flag = flag | 0x02;
	}
    // 위쪽
	if (loc_uv[1] < 0.5f) {
		flag = flag | 0x04;
		ky = 0.5f - loc_uv[1];
	}
    // 아래쪽
	else if (loc_uv[1] >= scalar_t(L) - 0.5f) {
		flag = flag | 0x08;
	}
    // index, ux, uy 순으로 저장
	if ((flag & 0x03) && (flag & 0x0C)) { // vertex case
		is_vertex = true;
		index_xy[0] = index; index_xy[1] = ux_0; index_xy[2] = uy_0;
		index_xy[3] = index; index_xy[4] = ux_0; index_xy[5] = uy_0; EdgeTable(L, flag & 0x03, &index_xy[3]);
		index_xy[6] = index; index_xy[7] = ux_0; index_xy[8] = uy_0; EdgeTable(L, flag & 0x0C, &index_xy[6]);	
	}
	else if ((flag & 0x03)) { // edge case u style
		index_xy[0] = index; index_xy[1] = ux_0; index_xy[2] = uy_0;
		index_xy[3] = index; index_xy[4] = ux_0; index_xy[5] = uy_0; EdgeTable(L, flag, &index_xy[3]);
		index_xy[6] = index; index_xy[7] = ux_0; index_xy[8] = uy_1;
		index_xy[9] = index; index_xy[10] = ux_0; index_xy[11] = uy_1; EdgeTable(L, flag, &index_xy[9]);	
	}
	else if ((flag & 0x0C)) { // edge case v style
		index_xy[0] = index; index_xy[1] = ux_0; index_xy[2] = uy_0;
		index_xy[3] = index; index_xy[4] = ux_1; index_xy[5] = uy_0;
		index_xy[6] = index; index_xy[7] = ux_0; index_xy[8] = uy_0; EdgeTable(L, flag, &index_xy[6]);
		index_xy[9] = index; index_xy[10] = ux_1; index_xy[11] = uy_0; EdgeTable(L, flag, &index_xy[9]);	
	}
	else {
		index_xy[0] = index; index_xy[1] = ux_0; index_xy[2] = uy_0;
		index_xy[3] = index; index_xy[4] = ux_1; index_xy[5] = uy_0;
		index_xy[6] = index; index_xy[7] = ux_0; index_xy[8] = uy_1;
		index_xy[9] = index; index_xy[10] = ux_1; index_xy[11] = uy_1;
	}
	kxky[0] = kx; kxky[1] = ky;
    *out_flag = flag;

	return is_vertex;
}

template<typename scalar_t>
__global__ void Cubemap_Bilinear_Seamless_Kernel(
    const scalar_t* __restrict__ inputs, // __restrict: 컴파일러 최적화 키워드, 메모리에서 가져온 값을 레지스터에 캐싱함
    const scalar_t* __restrict__ cubemap,
    const scalar_t* __restrict__ fail_value,
    scalar_t* __restrict__ outputs,
    const uint32_t B, const uint32_t C, const uint32_t L
)
{
    uint32_t n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < B)
    {
        // get normal
        scalar_t vx = inputs[n * 3];
        scalar_t vy = inputs[n * 3 + 1];
        scalar_t vz = inputs[n * 3 + 2];

        // for uv
        scalar_t uv[2];
        int cube_idx;

        // for seamless
        int index_xy[3 * 4];
        scalar_t kxky[2];
        int flag;

        // if normal = 0, return fail value
        if (vx == 0.f && vy == 0.f && vz == 0.f)
        {
            for (int ic = 0; ic < C; ++ic)
                outputs[ic * B + n] = fail_value[ic];
            return;
        }
        // compute uv
        Compute_Cubemap_UV(vx, vy, vz, uv, &cube_idx);
        // compute seamless idx
        bool is_vertex = Compute_Seamless_Index(cube_idx, L, uv, index_xy, kxky, &flag);
        
        // compute color
        // what is bilinear interpolation? I need more accurate understanding
        for (int ic = 0; ic < C; ++ic)
        {
            scalar_t v00, v01, v10, v11;
			v00 = cubemap[((index_xy[0] * C + ic)* L + index_xy[2]) * L + index_xy[1]];
			v01 = cubemap[((index_xy[3] * C + ic)* L + index_xy[5]) * L + index_xy[4]];
			v10 = cubemap[((index_xy[6] * C + ic)* L + index_xy[8]) * L + index_xy[7]];
			if (is_vertex)
            {
				v11 = (v00 + v01 + v10) / 3.f;
            }
			else
            {
				v11 = cubemap[((index_xy[9] * C + ic)* L + index_xy[11]) * L + index_xy[10]];
            }
			outputs[ic * B + n] = (1 - kxky[1])*((1 - kxky[0])* v00 + kxky[0] * v01) + kxky[1] * ((1 - kxky[0])*v10 + kxky[0] * v11);
        }
    }
}

// forward
void cubemap_encode_forward(const at::Tensor inputs, const at::Tensor cubemap, const at::Tensor fail_value,
                            at::Tensor outputs, const uint32_t B, const uint32_t C, const uint32_t L
)
{
    // check cuda
    CHECK_CUDA(inputs);
	CHECK_CUDA(cubemap);
	CHECK_CUDA(fail_value);
	CHECK_CUDA(outputs);
    
    // check contiguous
	CHECK_CONTIGUOUS(inputs);
	CHECK_CONTIGUOUS(cubemap);
	CHECK_CONTIGUOUS(fail_value);
	CHECK_CONTIGUOUS(outputs);
    
    // check data type
	CHECK_IS_FLOATING(inputs);
	CHECK_IS_FLOATING(cubemap);
	CHECK_IS_FLOATING(fail_value);
	CHECK_IS_FLOATING(outputs);

    // OptionalCUDAGuard: 현재 스레드가 작업을 수행할 GPU Device 설정하고, 작업이 끝나면 이전 Device 상태로 자동 복구.
    // device of: 입력된 텐서가 어느 Device에 위치해있는지 찾아주는 함수
    const at::cuda::OptionalCUDAGuard device_guard(device_of(cubemap));
    static constexpr uint32_t threads = 256;
    uint32_t blocks = uint32_t((B + threads - 1) / threads);

    // compute cubemap bilinear seamless kernel
    // Cubemap_Bilinear_Seamless_Kernel<scalar_t><<<blocks, threads>>>(
	// 				inputs.data_ptr<scalar_t>(),
	// 				cubemap.data_ptr<scalar_t>(),
	// 				fail_value.data_ptr<scalar_t>(),
	// 				outputs.data_ptr<scalar_t>(),
	// 				B, C, L
	// 			);
	// 이 매크로 필요함. 본래 코드의 매크로는 무시할 것.
	AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    cubemap.scalar_type(), "cubemap_encode_forward", ([&] {
				Cubemap_Bilinear_Seamless_Kernel<scalar_t><<<blocks, threads>>>(
					inputs.data_ptr<scalar_t>(),
					cubemap.data_ptr<scalar_t>(),
					fail_value.data_ptr<scalar_t>(),
					outputs.data_ptr<scalar_t>(),
					B, C, L
				);
    }));
}

template<typename scalar_t>
__device__ void Compute_Cubemap_UV_Backward(
	int index, scalar_t x, scalar_t y, scalar_t z, 
	scalar_t * uv, scalar_t * grad_xyz
)
{
	int face = index / 2;
	if (face == 0) {//0,1
		if (index == 0) {	uv[0] = -uv[0]; uv[1] = -uv[1];	}
		else {	uv[0] = -uv[0]; 	}
		grad_xyz[0] = -(z * uv[0] + y * uv[1]) / (x*x);
		grad_xyz[1] = 1.f / x * uv[1];
		grad_xyz[2] = 1.f / x * uv[0];
	}
	else if (face == 1) {//2,3
		if (index == 2) {}
		else { uv[0] = -uv[0]; uv[1] = -uv[1]; }
		grad_xyz[0] = 1.f / y * uv[0];
		grad_xyz[1] = -(x * uv[0] + z * uv[1]) / (y*y);
		grad_xyz[2] = 1.f / y * uv[1];
	}
	else if (face == 2) {//4,5
		if (index == 4) { uv[1] = -uv[1]; }
		else {}
		grad_xyz[0] = 1.f / z * uv[0];
		grad_xyz[1] = 1.f / z * uv[1];
		grad_xyz[2] = -(x*uv[0] + y * uv[1]) / (z*z);
	}
}

template<typename scalar_t>
__global__ void Cubemap_Bilinear_Seamless_Backward_Kernel(
	const scalar_t * __restrict__ grad_outputs, // [C,B]
	const scalar_t * __restrict__ cubemap, // [6,C,L,L]
	const scalar_t * __restrict__ inputs, // [B,3]
	scalar_t * __restrict__ grad_cubemap, // [6,C,L,L]
	scalar_t * __restrict__ grad_inputs, // [B,3]
	scalar_t * __restrict__ grad_fail, // [C]
	const uint32_t B, const uint32_t C, const uint32_t L
)
{
	const uint32_t n = blockIdx.x * blockDim.x + threadIdx.x;
	if (n < B)
    {
		scalar_t vx = inputs[n * 3 + 0];
		scalar_t vy = inputs[n * 3 + 1];
		scalar_t vz = inputs[n * 3 + 2];

		if (vx == 0.f && vy == 0.f && vz == 0.f) {
			for (int iC = 0; iC < C; iC++) {
				atomicAdd(grad_fail + iC, grad_outputs[iC * B + n]);
			}
			grad_inputs[n * 3 + 0] = 0;
		    grad_inputs[n * 3 + 1] = 0;
		    grad_inputs[n * 3 + 2] = 0;
			return;
		}
		scalar_t uv[2];
        int cube_idx;

		int index_xy[3 * 4];
        scalar_t kxky[2];
        int flag;

		Compute_Cubemap_UV(vx, vy, vz, uv, &cube_idx);
		bool is_vertex = Compute_Seamless_Index(cube_idx, L, uv, index_xy, kxky, &flag);
		scalar_t grad_view_[3] = { 0,0,0 };

		for (int ic = 0; ic < C; ++ic)
        {
			scalar_t v00, v01, v10, v11;
			//scalar_t grad_input = grad_image[(n*C + iC)*h*w + idx];
			scalar_t grad_input = grad_outputs[ic * B + n];
			v00 = cubemap[((index_xy[0] * C + ic)* L + index_xy[2]) * L + index_xy[1]];
			v01 = cubemap[((index_xy[3] * C + ic)* L + index_xy[5]) * L + index_xy[4]];
			v10 = cubemap[((index_xy[6] * C + ic)* L + index_xy[8]) * L + index_xy[7]];
			
            if (is_vertex)
            {
				v11 = (v00 + v01 + v10) / 3.f;
				scalar_t extra_g = kxky[1] * kxky[0] / 3.f;
				atomicAdd(grad_cubemap + ((index_xy[0] * C + ic)* L + index_xy[2]) * L + index_xy[1], ((1 - kxky[1])* (1 - kxky[0]) + extra_g) * grad_input);
				atomicAdd(grad_cubemap + ((index_xy[3] * C + ic)* L + index_xy[5]) * L + index_xy[4], ((1 - kxky[1])*kxky[0] + extra_g) * grad_input);
				atomicAdd(grad_cubemap + ((index_xy[6] * C + ic)* L + index_xy[8]) * L + index_xy[7], ((kxky[1] * (1 - kxky[0])) + extra_g) * grad_input);
			}
			else
            {
				v11 = cubemap[((index_xy[9] * C + ic)* L + index_xy[11]) * L + index_xy[10]];
				atomicAdd(grad_cubemap + ((index_xy[0] * C + ic)* L + index_xy[2]) * L + index_xy[1], (1 - kxky[1])* (1 - kxky[0]) * grad_input);
				atomicAdd(grad_cubemap + ((index_xy[3] * C + ic)* L + index_xy[5]) * L + index_xy[4], (1 - kxky[1])* kxky[0] * grad_input);
				atomicAdd(grad_cubemap + ((index_xy[6] * C + ic)* L + index_xy[8]) * L + index_xy[7], kxky[1] * (1 - kxky[0]) * grad_input);
				atomicAdd(grad_cubemap + ((index_xy[9] * C + ic)* L + index_xy[11]) * L + index_xy[10], kxky[1] * kxky[0] * grad_input);
			}
			scalar_t loc_grad[2];// ux,uy
			loc_grad[0] = (1 - kxky[1]) * (v01 - v00) + kxky[1] * (v11 - v10);
			loc_grad[1] = (1 - kxky[0]) * (v10 - v00) + kxky[0] * (v11 - v01);
			loc_grad[0] *= 0.5f * scalar_t(L) * grad_input;
			loc_grad[1] *= 0.5f * scalar_t(L) * grad_input;
			if (flag & 0x01)
            {
				loc_grad[0] = -loc_grad[0];
			}
			if (flag & 0x04)
            {
				loc_grad[1] = -loc_grad[1];
			}
// #ifdef LEFT_TOP_AS_ORIGIN
			loc_grad[1] = -loc_grad[1];
// #endif
			scalar_t loc_grad_view[3];
			Compute_Cubemap_UV_Backward(
				cube_idx, vx, vy, vz, loc_grad, loc_grad_view
			);
			grad_view_[0] += loc_grad_view[0];
			grad_view_[1] += loc_grad_view[1];
			grad_view_[2] += loc_grad_view[2];
		}
		grad_inputs[n * 3 + 0] = grad_view_[0];
		grad_inputs[n * 3 + 1] = grad_view_[1];
		grad_inputs[n * 3 + 2] = grad_view_[2];
	}
}

// backward
void cubemap_encode_backward(const at::Tensor grad_outputs, const at::Tensor inputs, const at::Tensor cubemap,
                            at::Tensor grad_cubemap, at::Tensor grad_inputs, at::Tensor grad_fail,
                            const uint32_t B, const uint32_t C, const uint32_t L
)
{
    CHECK_CUDA(grad_outputs);
	CHECK_CUDA(inputs);
	CHECK_CUDA(cubemap);
	CHECK_CUDA(grad_cubemap);
	CHECK_CUDA(grad_inputs);
	CHECK_CUDA(grad_fail);

	CHECK_CONTIGUOUS(grad_outputs);
	CHECK_CONTIGUOUS(inputs);
	CHECK_CONTIGUOUS(cubemap);
	CHECK_CONTIGUOUS(grad_cubemap);
	CHECK_CONTIGUOUS(grad_inputs);
	CHECK_CONTIGUOUS(grad_fail);

	CHECK_IS_FLOATING(grad_outputs);
	CHECK_IS_FLOATING(inputs);
	CHECK_IS_FLOATING(cubemap);
	CHECK_IS_FLOATING(grad_cubemap);
	CHECK_IS_FLOATING(grad_inputs);
	CHECK_IS_FLOATING(grad_fail);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(cubemap));
	static constexpr uint32_t threads = 256;
	uint32_t blocks = uint32_t((B + threads - 1) / threads);

    #define scalar_t float
    Cubemap_Bilinear_Seamless_Backward_Kernel<scalar_t><<<blocks, threads>>>(
					grad_outputs.data<scalar_t>(), cubemap.data<scalar_t>(), inputs.data<scalar_t>(),
					grad_cubemap.data<scalar_t>(), grad_inputs.data<scalar_t>(), grad_fail.data<scalar_t>(),
					B, C, L
				);
    #undef scalar_t
}