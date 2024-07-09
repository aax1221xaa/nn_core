#define CUDA_API_PER_THREAD_DEFAULT_STEAM 
#include "maxpool.cuh"
#include "cuda_misc.cuh"

#ifndef __CUDACC__
#define __CUDACC__
#endif

#include <device_functions.h>
#include <device_launch_parameters.h>


/**********************************************/
/*											  */
/*				 kernel function			  */
/*										      */
/**********************************************/

__global__ void __maxpool_2d(
	const nn_type* a,
	nn_type* b,
	uint* mark,
	cuint a_h,
	cuint a_w,
	cuint b_h,
	cuint b_w,
	cuint k_h,
	cuint k_w,
	cuint st_h,
	cuint st_w,
	cuint tile_h,
	cuint tile_w
) {
	extern __shared__ nn_type sm[];

	cuint out_x = blockIdx.x * blockDim.x + threadIdx.x;
	cuint out_y = blockIdx.y * blockDim.y + threadIdx.y;
	cuint x0 = blockIdx.x * blockDim.x * st_w;
	cuint y0 = blockIdx.y * blockDim.y * st_h;
	cuint z0 = blockIdx.z;

	const nn_type* pa = a + ((z0 * a_w * a_h) + (y0 * a_w) + x0);
	nn_type* pb = b + (b_w * b_h * z0);
	uint* pmark = mark + (b_w * b_h * z0);

	for (uint h = 0; h < tile_h; h += blockDim.y) {
		cuint ty = threadIdx.y + h;
		cuint in_y = ty + y0;

		for (uint w = 0; w < tile_w; w += blockDim.x) {
			cuint tx = threadIdx.x + w;
			cuint in_x = tx + x0;

			if (tx < tile_w && ty < tile_h && in_x < a_w && in_y < a_h) {
				sm[ty * tile_w + tx] = pa[ty * a_w + tx];
			}
		}
	}
		
	__syncthreads();

	if (out_x < b_w && out_y < b_h) {
		nn_type val = -FLT_MAX;
		uint index = 0;

		for (uint h = 0; h < k_h; ++h) {
			cuint ty = threadIdx.y * st_h + h;
			for (uint w = 0; w < k_w; ++w) {
				cuint tx = threadIdx.x * st_w + w;
				nn_type sm_val = sm[ty * tile_w + tx];

				if (sm_val > val) {
					val = sm_val;
					index = h * k_w + w;
				}
			}
		}

		pb[out_y * b_w + out_x] = val;
		pmark[out_y * b_w + out_x] = index;
	}
}

/*
__global__ void __d_maxpool(
	const float* a,
	const uint* indice,
	float* b,
	cuint a_h,
	cuint a_w,
	cuint b_h,
	cuint b_w,
	cuint k_w,
	cuint st_h,
	cuint st_w
) {
	cuint cx = blockIdx.x * blockDim.x + threadIdx.x;
	cuint cy = blockIdx.y * blockDim.y + threadIdx.y;
	cuint cz = blockIdx.z;

	const float* p_a = a + (cz * (a_h * a_w) + (cy * st_h * a_w) + (cx * st_w));
	const uint* p_indice = indice + (cz * (b_h * b_w) + (cy * b_w) + cx);
	float* p_b = b + (cz * (b_h * b_w) + (cy * b_w) + cx);

	if (cx < b_w && cy < b_h) {
		cuint index = *p_indice;
		cuint x = index % k_w;
		cuint y = index / k_w;

		*p_b = p_a[y * a_w + x];
	}
}
*/

/**********************************************

				    Maxpool2d

**********************************************/
/*
void maxpool2d(
	cudaStream_t s,
	const nn_type* input,
	nn_type* output,
	uint* max_indice,
	cuint in_h,
	cuint in_w,
	cuint out_h,
	cuint out_w,
	cuint h_kernel,
	cuint w_kernel,
	cuint h_stride,
	cuint w_stride,
	cuint h_tile,
	cuint w_tile
) {
	size_t smem_size = sizeof(nn_type) * h_tile * w_tile;
	dim3 threads(BLOCK_32, BLOCK_32);
	dim3 blocks = get_grid_size(threads, in_w, in_h);

	__maxpool_2d<<<blocks, threads, smem_size, s>>>(
		input,
		output,
		max_indice,
		in_h,
		in_w,
		out_h,
		out_w,
		h_kernel,
		w_kernel,
		h_stride,
		w_stride,
		h_tile,
		w_tile
	);
}
*/
/**********************************************/
/*                                            */
/*                NN_Maxpool2D                */
/*                                            */
/**********************************************/

NN_Maxpool2D::NN_Maxpool2D(const NN_Shape& k_size, const NN_Shape& stride, const Pad pad, const char* name) :
	_pad(pad),
	_k_size(k_size),
	_stride(stride),
	NN_Layer(name)
{
}

void NN_Maxpool2D::get_output_shape(const NN_List<NN_Shape>& input_shape, NN_List<NN_Shape>& output_shape) {
	const NN_Shape& shape = input_shape[0].val();

	int n = shape[0];
	int c = shape[1];
	int h = 0;
	int w = 0;

	if (_pad == Pad::SAME) {
		h = (int)ceil((float)(shape[2] - _k_size[0]) / _stride[0] + 1);
		w = (int)ceil((float)(shape[3] - _k_size[1]) / _stride[1] + 1);
	}
	else {
		h = (int)floorf((float)(shape[2] - _k_size[0]) / _stride[0] + 1);
		w = (int)floorf((float)(shape[3] - _k_size[1]) / _stride[1] + 1);
	}

	output_shape.append(NN_Shape({ n, c, h, w }));
	if (n > 0) _indice = GpuTensor<uint>::zeros({ n, c, h, w });
}

void NN_Maxpool2D::build(const NN_List<NN_Shape>& input_shape, NN_Link* p_node) {

}

void NN_Maxpool2D::run(NN_Stream& st, const NN_List<GpuTensor<nn_type>>& input, NN_List<GpuTensor<nn_type>>& output) {
	const nn_type* m_input = input[0].val().get_ptr();
	nn_type* m_output = output[0].val().get_ptr();
	uint* m_indice = _indice.get_ptr();

	const NCHW in = input[0].val().get_shape().get_nchw();
	const NCHW out = output[0].val().get_shape().get_nchw();

	int tile_h = (BLOCK_32 - 1) * _stride[0] + _k_size[0];
	int tile_w = (BLOCK_32 - 1) * _stride[1] + _k_size[1];

	size_t smem_size = sizeof(nn_type) * tile_h * tile_w;
	dim3 threads(BLOCK_32, BLOCK_32);
	dim3 blocks = get_grid_size(threads, in.w, in.h, in.c);

	cuint kh = _k_size[0];
	cuint kw = _k_size[1];
	cuint sh = _stride[0];
	cuint sw = _stride[1];
	
	cudaStream_t* p_st = st.get_stream();

	for (int n = 0; n < in.n; ++n) {
		const nn_type* in_data = m_input + (n * in.c * in.h * in.w);
		nn_type* out_data = m_output + (n * out.c * out.h * out.w);
		uint* indice = m_indice + (n * out.c * out.h * out.w);

		__maxpool_2d<<<blocks, threads, smem_size, p_st[n % STREAMS]>>>(
			in_data,
			out_data,
			indice,
			in.h,
			in.w,
			out.h,
			out.w,
			kh,
			kw,
			sh,
			sw,
			tile_h,
			tile_w
		);
	}
	check_cuda(cudaDeviceSynchronize());
	check_cuda(cudaGetLastError());
}

NN_Backward* NN_Maxpool2D::create_backward(NN_Optimizer* optimizer) {
	return new NN_dMaxpool2D(this, optimizer);
}


/**********************************************/
/*                                            */
/*                NN_dMaxpool2D               */
/*                                            */
/**********************************************/

NN_dMaxpool2D::NN_dMaxpool2D(NN_Maxpool2D* maxpool, NN_Optimizer* optimizer) :
	NN_Backward(optimizer),
	_maxpool(maxpool)
{

}

void NN_dMaxpool2D::get_dinput_shape(const NN_List<NN_Shape>& dout_shape, NN_List<NN_Shape>& din_shape) {

}

void NN_dMaxpool2D::run(
	NN_Stream& st,
	const NN_List<GpuTensor<nn_type>>& input,
	const NN_List<GpuTensor<nn_type>>& doutput,
	NN_List<GpuTensor<nn_type>>& dinput
) {

}


/**********************************************

				  D_Maxpool2d

**********************************************/
/*
void d_maxpool2d(
	cudaStream_t* s,
	const nn_type* d_output,
	nn_type* d_input,
	const nn_shape& out_shape,
	const nn_shape& in_shape,
	cuint* max_indice,
	cuint w_kernel,
	cuint h_stride,
	cuint w_stride
) {
	dim3 threads(BLOCK_32, BLOCK_32);
	dim3 blocks = get_grid_size(threads, out_shape[3], out_shape[2], out_shape[1]);

	for (int i = 0; i < out_shape[0]; ++i) {
		const nn_type* d_doutput = d_output + (i * out_shape[1] * out_shape[2] * out_shape[3]);
		cuint* d_indice = max_indice + (i * out_shape[1] * out_shape[2] * out_shape[3]);
		nn_type* d_dinput = d_input + (i * in_shape[1] * in_shape[2] * in_shape[3]);

		__d_maxpool<<<blocks, threads, 0, s[i % STREAMS]>>>(
			d_doutput,
			d_indice,
			d_dinput,
			out_shape[2],
			out_shape[3],
			in_shape[2],
			in_shape[3],
			w_kernel,
			h_stride,
			w_stride
		);
	}
}
*/