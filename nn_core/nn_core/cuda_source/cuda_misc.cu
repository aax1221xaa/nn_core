#define CUDA_API_PER_THREAD_DEFAULT_STEAM 
#include "cuda_misc.cuh"

#ifndef __CUDACC__
#define __CUDACC__
#endif

#include <device_functions.h>
#include <device_launch_parameters.h>


__constant__ uint __cmem[CONST_ELEM_SIZE];


/*******************************************
											  
			   kernel functions			  

*******************************************/

__global__ void __transpose(
	const nn_type* input,
	nn_type* output,
	cuint* c_trans_ranks,
	cuint* c_dims,
	cuint* c_steps,
	uint n_ranks,
	cuint total_size
) {
	cuint tidx = blockIdx.x * blockDim.x + threadIdx.x;
	
	uint quot = tidx;
	uint src_index = 0;

	while (n_ranks) {
		--n_ranks;

		cuint rank = c_trans_ranks[n_ranks];
		cuint dim = c_dims[rank];
		cuint curr_dim = quot % dim;

		src_index += c_steps[rank] * curr_dim;

		quot /= dim;
	}

	if (tidx < total_size) {
		output[tidx] = input[src_index];
	}
}

__global__ void __padding_dilation_2d(
	const nn_type* input,
	nn_type* output,
	cuint in_w,
	cuint in_h,
	cuint in_c,
	cuint out_w,
	cuint out_h,
	cuint stride_x,
	cuint stride_y,
	cuint offset_x,
	cuint offset_y
) {
	cuint in_cx = blockIdx.x * blockDim.x + threadIdx.x;
	cuint in_cy = blockIdx.y * blockDim.y + threadIdx.y;
	cuint in_cz = blockIdx.z;

	cuint out_cx = in_cx * stride_x + offset_x;
	cuint out_cy = in_cy * stride_y + offset_y;

	if (in_cx < in_w && in_cy < in_h) {
		output[(out_cy * out_w * in_c) + (out_cx * in_c) + in_cz] = input[(in_cy * in_w * in_c) + (in_cx * in_c) + in_cz];
	}
}

__global__ void __add_bias_32x32(
	const nn_type* data_a,
	const nn_type* data_b,
	nn_type* data_c,
	cuint n,
	cuint c
) {
	cuint cx = blockIdx.x * blockDim.x + threadIdx.x;
	cuint cy = blockIdx.y * blockDim.y + threadIdx.y;

	cuint addr = cy * c + cx;

	__shared__ nn_type share_b[BLOCK_32];

	if (threadIdx.y == 0) share_b[threadIdx.x] = cx < c ? data_b[cx] : 0.f;
	__syncthreads();

	if (cx < c && cy < n) {
		data_c[addr] = data_a[addr] + share_b[threadIdx.x];
	}
}

__global__ void __add_bias_16x16x4(
	const nn_type* data_a,
	const nn_type* data_b,
	nn_type* data_c,
	cuint h,
	cuint w,
	cuint c
) {
	cuint cx = blockIdx.x * blockDim.x + threadIdx.x;
	cuint cy = blockIdx.y * blockDim.y + threadIdx.y;
	cuint cz = blockIdx.z * blockDim.z + threadIdx.z;

	cuint addr = (cy * w * c) + (cx * c) + cz;

	__shared__ nn_type share_b[BLOCK_4];

	if (threadIdx.x == 0 && threadIdx.y == 0) share_b[threadIdx.z] = cz < c ? data_b[cz] : 0.f;
	__syncthreads();

	if (cx < w && cy < h && cz < c) {
		data_c[addr] = data_a[addr] + share_b[threadIdx.z];
	}
}

__global__ void __add_bias_8x8x16(
	const nn_type* data_a,
	const nn_type* data_b,
	nn_type* data_c,
	cuint h,
	cuint w,
	cuint c
) {
	cuint cx = blockIdx.x * blockDim.x + threadIdx.x;
	cuint cy = blockIdx.y * blockDim.y + threadIdx.y;
	cuint cz = blockIdx.z * blockDim.z + threadIdx.z;

	cuint addr = (cy * w * c) + (cx * c) + cz;

	__shared__ nn_type share_b[BLOCK_16];

	if (threadIdx.x == 0 && threadIdx.y == 0) share_b[threadIdx.z] = cz < c ? data_b[cz] : 0.f;
	__syncthreads();

	if (cx < w && cy < h && cz < c) {
		data_c[addr] = data_a[addr] + share_b[threadIdx.z];
	}
}

__global__ void __sum_gradient_1d(
	const nn_type* a,
	nn_type* b,
	cuint n,
	cuint c
) {
	cuint cx = blockIdx.x * blockDim.x + threadIdx.x;
	cuint tidx = threadIdx.y * BLOCK_32 + threadIdx.x;

	__shared__ nn_type sm[BLOCK_32 * BLOCK_32];
	
	sm[tidx] = 0.f;
	__syncthreads();

	for (uint i = 0; i < n; i += BLOCK_32) {
		cuint cy = threadIdx.y + i;

		if (cx < c && cy < n) {
			sm[tidx] += a[cy * c + cx];
		}
	}

#pragma unroll
	for (uint i = BLOCK_32 / 2; i > 0; i /= 2) {
		cuint half_side = (threadIdx.y + i) * BLOCK_32 + threadIdx.x;

		__syncthreads();
		if (threadIdx.y < i) sm[tidx] += sm[half_side];
	}

	if (cx < c && threadIdx.y == 0) b[cx] = sm[threadIdx.x];
}

__global__ void __sum_gradient_2d(
	const nn_type* a,
	nn_type* b,
	cuint n,
	cuint h,
	cuint w,
	cuint c
) {
	/*
	threads = [4, 16, 16]
	blocks = [c]
	*/

	__shared__ nn_type sm[BLOCK_4 * BLOCK_16 * BLOCK_16];

	cuint tidx = threadIdx.z * (BLOCK_16 * BLOCK_16) + threadIdx.y * BLOCK_16 + threadIdx.x;
	cuint cidx = blockIdx.x;

	sm[tidx] = 0.f;
	__syncthreads();

	for (uint z = 0; z < n; z += BLOCK_4) {
		cuint tz = threadIdx.z + z;
		cuint nidx = tz * (c * h * w);

		for (uint y = 0; y < h; y += BLOCK_16) {
			cuint ty = threadIdx.y + y;
			cuint yidx = ty * (w * c);

			for (uint x = 0; x < w; x += BLOCK_16) {
				cuint tx = threadIdx.x + x;
				cuint xidx = tx * c;

				if (tz < n && ty < h && tx < w) sm[tidx] += a[nidx + yidx + xidx + cidx];
			}
		}
	}

#pragma unroll
	for (uint i = BLOCK_1024 / 2; i > 0; i /= 2) {
		__syncthreads();

		if (tidx < i) sm[tidx] += sm[tidx + i];
	}

	b[blockIdx.x] = sm[0];
}

__global__ void __add_tensor(
	const nn_type* a_input,
	const nn_type* b_input,
	nn_type* output,
	cuint size
) {
	cuint idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size) output[idx] = a_input[idx] + b_input[idx];
}

__global__ void __add_tensor2(
	const nn_type* a_input,
	const nn_type b_input,
	nn_type* output,
	cuint size
) {
	cuint idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size) output[idx] = a_input[idx] + b_input;
}

__global__ void __sub_tensor(
	const nn_type* a_input,
	const nn_type* b_input,
	nn_type* output,
	cuint size
) {
	cuint idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size) output[idx] = a_input[idx] - b_input[idx];
}

__global__ void __sub_tensor2(
	const nn_type* a_input,
	const nn_type b_input,
	nn_type* output,
	cuint size
) {
	cuint idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size) output[idx] = a_input[idx] - b_input;
}

__global__ void __sub_tensor3(
	const nn_type a_input,
	const nn_type* b_input,
	nn_type* output,
	cuint size
) {
	cuint idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size) output[idx] = a_input - b_input[idx];
}

__global__ void __mul_tensor(
	const nn_type* a_input,
	const nn_type* b_input,
	nn_type* output,
	cuint size
) {
	cuint idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size) output[idx] = a_input[idx] * b_input[idx];
}

__global__ void __mul_tensor2(
	const nn_type* a_input,
	const nn_type b_input,
	nn_type* output,
	cuint size
) {
	cuint idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size) output[idx] = a_input[idx] * b_input;
}

__global__ void __div_tensor(
	const nn_type* a_input,
	const nn_type* b_input,
	nn_type* output,
	cuint size
) {
	cuint idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size) output[idx] = a_input[idx] / b_input[idx];
}

__global__ void __div_tensor2(
	const nn_type* a_input,
	const nn_type b_input,
	nn_type* output,
	cuint size
) {
	cuint idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size) output[idx] = a_input[idx] / b_input;
}

__global__ void __inv_tensor(
	const nn_type* a_input,
	nn_type* output,
	cuint size
) {
	cuint idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size) output[idx] = 1. / a_input[idx];
}

/*******************************************

			    host functions

*******************************************/

void set_const_mem(cuint* h_mem, size_t len, size_t offset) {
#if _DEBUG
	check_cuda(cudaMemcpyToSymbol(__cmem, h_mem, sizeof(uint) * len, sizeof(uint) * offset));
#else
	cudaMemcpyToSymbol(__cmem, h_mem, sizeof(uint) * len, sizeof(uint) * offset);
#endif
}

cuint* get_const_mem(size_t len, size_t offset) {
	uint* ptr = NULL;

#if _DEBUG
	check_cuda(cudaGetSymbolAddress((void**)&ptr, __cmem));
#else
	cudaGetSymbolAddress((void**)&ptr, __cmem);
#endif

	return ptr + offset;
}

void transpose_param_init(
	const GpuTensor<nn_type>& input,
	const std::vector<uint>& ranks,
	cuint** c_dims,
	cuint** c_steps,
	cuint** c_ranks
) {
	const NN_Shape shape = input.get_shape();
	uint* ptr = new uint[shape.ranks()];
	size_t offset = 0;
	size_t len = (size_t)shape.ranks();

	int i = 0;

	for (const int& n : shape) ptr[i++] = (uint)n;

	set_const_mem(ptr, len, offset);
	*c_dims = get_const_mem(len, offset);

	offset += len;

	uint step = 1;

	while (i) {
		--i;

		cuint dim = ptr[i];
		ptr[i] = step;
		step *= dim;
	}

	set_const_mem(ptr, len, offset);
	*c_steps = get_const_mem(len, offset);

	offset += len;

	for (const int& n : ranks) ptr[i++] = n;

	set_const_mem(ptr, len, offset);
	*c_ranks = get_const_mem(len, offset);

	delete[] ptr;
}

void transpose(
	const GpuTensor<nn_type>& input,
	GpuTensor<nn_type>& output,
	const std::vector<uint>& ranks
) {
	dim3 threads(BLOCK_1024);
	dim3 blocks = get_grid_size(threads, (uint)input.get_shape().total_size());

	cuint* c_dims = NULL;
	cuint* c_steps = NULL;
	cuint* c_trans_ranks = NULL;

	transpose_param_init(input, ranks, &c_dims, &c_steps, &c_trans_ranks);

	__transpose<<<blocks, threads>>>(
		(nn_type*)input.get_ptr(),
		(nn_type*)output.get_ptr(),
		c_trans_ranks,
		c_dims,
		c_steps,
		(uint)input.get_shape().ranks(),
		(uint)input.get_shape().total_size()
	);
#if _DEBUG
	check_cuda(cudaDeviceSynchronize());
	check_cuda(cudaGetLastError());
#endif
}

void padding_dilation(
	cudaStream_t s,
	const nn_type* input,
	nn_type* output,
	const NN_Tensor4dShape& in,
	const NN_Tensor4dShape& out,
	cuint offset_x,
	cuint offset_y,
	cuint stride_x,
	cuint stride_y
) {
	dim3 threads(BLOCK_32, BLOCK_32);
	dim3 blocks = get_grid_size(threads, in._w, in._h, in._c);

	__padding_dilation_2d<<<blocks, threads, 0, s>>>(
		input,
		output,
		(uint)in._w,
		(uint)in._h,
		(uint)in._c,
		(uint)out._w,
		(uint)out._h,
		stride_x,
		stride_y,
		offset_x,
		offset_y
	);
}

void add_bias_1d(
	const GpuTensor<nn_type>& input,
	const GpuTensor<nn_type>& bias,
	GpuTensor<nn_type>& output
) {
	const NN_Shape& in = input.get_shape();

	dim3 threads(BLOCK_32, BLOCK_32);
	dim3 blocks = get_grid_size(threads, in[1], in[0]);

	__add_bias_32x32<<<blocks, threads>>>(
		input.get_ptr(),
		bias.get_ptr(),
		output.get_ptr(),
		(uint)in[0],
		(uint)in[1]
	);
#if _DEBUG
	check_cuda(cudaDeviceSynchronize());
	check_cuda(cudaGetLastError());
#endif
}

void add_bias_2d(
	NN_Stream& s,
	const GpuTensor<nn_type>& input,
	const GpuTensor<nn_type>& bias,
	GpuTensor<nn_type>& output
) {
	const NN_Tensor4dShape& in = input.get_shape().get_4d_shape();
	const nn_type* in_data = input.get_ptr();
	const nn_type* bias_data = bias.get_ptr();
	nn_type* out_data = output.get_ptr();

	cudaStream_t* p_st = s.get_stream();

	if (in._h >= BLOCK_16 && in._w >= BLOCK_16 || in._c <= BLOCK_4) {
		dim3 threads(BLOCK_16, BLOCK_16, BLOCK_4);
		dim3 blocks = get_grid_size(threads, in._w, in._h, in._c);

		for (int i = 0; i < in._n; ++i) {
			cuint index = in._h * in._w * in._c * i;
			const nn_type* d_in = in_data + index;
			nn_type* d_out = out_data + index;
			
			__add_bias_16x16x4<<<blocks, threads, 0, p_st[i % STREAMS]>>>(
				d_in,
				bias_data,
				d_out,
				(uint)in._h,
				(uint)in._w,
				(uint)in._c
			);
		}
	}
	else {
		dim3 threads(BLOCK_8, BLOCK_8, BLOCK_16);
		dim3 blocks = get_grid_size(threads, in._w, in._h, in._c);

		for (int i = 0; i < in._n; ++i) {
			cuint index = in._h * in._w * in._c;
			const nn_type* d_in = in_data + index;
			nn_type* d_out = out_data + index;

			__add_bias_8x8x16<<<blocks, threads, 0, p_st[i % STREAMS]>>>(
				d_in,
				bias_data,
				d_out,
				(uint)in._h,
				(uint)in._w,
				(uint)in._c
			);
		}
	}
}

void sum_gradient_1d(
	const GpuTensor<nn_type>& input,
	GpuTensor<nn_type>& output
) {
	const NN_Shape& in = input.get_shape();

	dim3 threads(BLOCK_32, BLOCK_32);
	dim3 blocks = get_grid_size(threads, in[1]);

	__sum_gradient_1d<<<blocks, threads>>>(
		input.get_ptr(),
		output.get_ptr(),
		(uint)in[0],
		(uint)in[1]
	);
}

void sum_gradient_2d(
	const GpuTensor<nn_type>& input,
	GpuTensor<nn_type>& output
) {
	const NN_Tensor4dShape& in = input.get_shape().get_4d_shape();

	dim3 threads(BLOCK_16, BLOCK_16, BLOCK_4);
	dim3 blocks(in._c);

	__sum_gradient_2d<<<blocks, threads>>>(
		input.get_ptr(),
		output.get_ptr(),
		(uint)in._n,
		(uint)in._h,
		(uint)in._w,
		(uint)in._c
	);
}

void add_tensor(
	const GpuTensor<nn_type>& a_input,
	const GpuTensor<nn_type>& b_input,
	GpuTensor<nn_type>& output
) {
	const NN_Shape& a_shape = a_input.get_shape();
	const NN_Shape& b_shape = b_input.get_shape();
	const NN_Shape& out_shape = output.get_shape();

	if (a_shape != b_shape || b_shape != out_shape) {
		ErrorExcept(
			"[add_tensor] shapes of input and output are different. %s != %s != %s",
			shape_to_str(a_shape),
			shape_to_str(b_shape),
			shape_to_str(out_shape)
		);
	}

	cuint len = (uint)a_shape.total_size();
	const dim3 threads(BLOCK_1024);
	const dim3 blocks = get_grid_size(threads, len);

	__add_tensor<<<blocks, threads>>>(
		a_input.get_ptr(),
		b_input.get_ptr(),
		output.get_ptr(),
		len
	);
}

void add_tensor(
	const GpuTensor<nn_type>& input,
	const nn_type scalar,
	GpuTensor<nn_type>& output
) {
	const NN_Shape& in_shape = input.get_shape();
	const NN_Shape& out_shape = output.get_shape();

	if (in_shape != out_shape) {
		ErrorExcept(
			"[add_tensor] shapes of input and output are different. %s != %s",
			shape_to_str(in_shape),
			shape_to_str(out_shape)
		);
	}

	cuint len = (uint)input.get_shape().total_size();
	const dim3 threads(BLOCK_1024);
	const dim3 blocks = get_grid_size(threads, len);

	__add_tensor2<<<blocks, threads>>>(
		input.get_ptr(),
		scalar,
		output.get_ptr(),
		len
	);
}

void sub_tensor(
	const GpuTensor<nn_type>& a_input,
	const GpuTensor<nn_type>& b_input,
	GpuTensor<nn_type>& output
) {
	const NN_Shape& a_shape = a_input.get_shape();
	const NN_Shape& b_shape = b_input.get_shape();
	const NN_Shape& out_shape = output.get_shape();

	if (a_shape != b_shape || a_shape != out_shape) {
		ErrorExcept(
			"[sub_tensor] shapes of input and output are different. %s != %s != %s",
			shape_to_str(a_shape),
			shape_to_str(b_shape),
			shape_to_str(out_shape)
		);
	}

	cuint len = (uint)a_shape.total_size();
	const dim3 threads(BLOCK_1024);
	const dim3 blocks = get_grid_size(threads, len);

	__sub_tensor<<<blocks, threads>>>(
		a_input.get_ptr(),
		b_input.get_ptr(),
		output.get_ptr(),
		len
	);
}

void sub_tensor(
	const GpuTensor<nn_type>& input,
	const nn_type scalar,
	GpuTensor<nn_type>& output
) {
	const NN_Shape& in_shape = input.get_shape();
	const NN_Shape& out_shape = output.get_shape();

	if (in_shape != out_shape) {
		ErrorExcept(
			"[sub_tensor] shapes of input and output are different. %s != %s",
			shape_to_str(in_shape),
			shape_to_str(out_shape)
		);
	}

	cuint len = (uint)in_shape.total_size();
	const dim3 threads(BLOCK_1024);
	const dim3 blocks = get_grid_size(threads, len);

	__sub_tensor2<<<blocks, threads>>>(
		input.get_ptr(),
		scalar,
		output.get_ptr(),
		len
	);
}

void sub_tensor(
	const nn_type scalar,
	const GpuTensor<nn_type>& input,
	GpuTensor<nn_type>& output
) {

}

void mul_tensor(
	const GpuTensor<nn_type>& a_input,
	const GpuTensor<nn_type>& b_input,
	GpuTensor<nn_type>& output
) {

}

void mul_tensor(
	const GpuTensor<nn_type>& input,
	const nn_type scalar,
	GpuTensor<nn_type>& output
) {
	const NN_Shape shape = output.get_shape();
	cuint len = (uint)shape.total_size();

	dim3 threads(BLOCK_1024);
	dim3 blocks = get_grid_size(threads, len);

	__mul_tensor2<<<blocks, threads>>>(
		input.get_ptr(),
		scalar,
		output.get_ptr(),
		len
	);
}

void div_tensor(
	const GpuTensor<nn_type>& a_input,
	const GpuTensor<nn_type>& b_input,
	GpuTensor<nn_type>& output
) {
	const NN_Shape shape = output.get_shape();
	cuint len = (uint)shape.total_size();

	dim3 threads(BLOCK_1024);
	dim3 blocks = get_grid_size(threads, len);

	__div_tensor<<<blocks, threads>>>(
		a_input.get_ptr(),
		b_input.get_ptr(),
		output.get_ptr(),
		len
	);
}

void div_tensor(
	const GpuTensor<nn_type>& input,
	const nn_type scalar,
	GpuTensor<nn_type>& output
) {
	cuint len = (uint)output.get_shape().total_size();
	dim3 threads(BLOCK_1024);
	dim3 blocks = get_grid_size(threads, len);

	__div_tensor2<<<blocks, threads>>>(
		input.get_ptr(),
		scalar,
		output.get_ptr(),
		len
	);
#if _DEBUG
	check_cuda(cudaDeviceSynchronize());
	check_cuda(cudaGetLastError());
#endif
}

void inv_tensor(
	const GpuTensor<nn_type>& input,
	GpuTensor<nn_type>& output
) {
	cuint len = (uint)output.get_shape().total_size();
	dim3 threads(BLOCK_1024);
	dim3 blocks = get_grid_size(threads, len);

	__inv_tensor << <blocks, threads >> > (
		input.get_ptr(),
		output.get_ptr(),
		len
		);
}