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
	cuint a_c,
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

	const nn_type* pa = a + ((y0 * a_w * a_c) + (x0 * a_c) + z0);
	nn_type* pb = b + z0;
	uint* pmark = mark + z0;

	for (uint h = 0; h < tile_h; h += blockDim.y) {
		cuint ty = threadIdx.y + h;
		cuint in_y = ty + y0;

		for (uint w = 0; w < tile_w; w += blockDim.x) {
			cuint tx = threadIdx.x + w;
			cuint in_x = tx + x0;

			if (tx < tile_w && ty < tile_h && in_x < a_w && in_y < a_h) {
				sm[ty * tile_w + tx] = pa[(ty * a_w * a_c) + (tx * a_c)];
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

		pb[(out_y * b_w * a_c) + out_x * a_c] = val;
		pmark[(out_y * b_w * a_c) + out_x * a_c] = index;
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

uint NN_Maxpool2D::special_i = 0;

NN_Maxpool2D::NN_Maxpool2D(const NN_Shape& k_size, const NN_Shape& stride, const std::string& pad, const std::string& name) :
	_pad(pad),
	_k_size(k_size),
	_stride(stride),
	NN_Layer(name, "max_pooling2d")
{
}

void NN_Maxpool2D::get_output_shape(const NN_List<NN_Shape>& input_shape, NN_List<NN_Shape>& output_shape) {
	const NN_Shape& shape = input_shape[0].val();

	++special_i;

	int n = shape[0];
	int h = 0;
	int w = 0;
	int c = shape[3];

	if (_pad == "same") {
		h = (int)ceil(float(shape[1]) / _stride[1]);
		w = (int)ceil(float(shape[2]) / _stride[0]);
	}
	else {
		h = (int)ceil(float(shape[1] - _k_size[1] + 1) / _stride[1]);
		w = (int)ceil(float(shape[2] - _k_size[0] + 1) / _stride[0]);
	}

	output_shape.append(NN_Shape({ n, h, w, c }));
	if (n > 0) _indice = GpuTensor<uint>::zeros(NN_Shape({ n, h, w, c }));
}

void NN_Maxpool2D::build(const NN_List<NN_Shape>& input_shape, NN_List<GpuTensor<nn_type>>& weights) {

}

void NN_Maxpool2D::run(NN_Stream& st, const NN_List<GpuTensor<nn_type>>& input, NN_List<GpuTensor<nn_type>>& output) {
	const nn_type* m_input = input[0].val().get_ptr();
	nn_type* m_output = output[0].val().get_ptr();
	uint* m_indice = _indice.get_ptr();

	const NN_Tensor4dShape in = input[0].val().get_shape().get_4d_shape();
	const NN_Tensor4dShape out = output[0].val().get_shape().get_4d_shape();

	int tile_h = (BLOCK_32 - 1) * _stride[0] + _k_size[0];
	int tile_w = (BLOCK_32 - 1) * _stride[1] + _k_size[1];

	size_t smem_size = sizeof(nn_type) * tile_h * tile_w;
	dim3 threads(BLOCK_32, BLOCK_32);
	dim3 blocks = get_grid_size(threads, in._w, in._h, in._c);

	cuint kh = _k_size[0];
	cuint kw = _k_size[1];
	cuint sh = _stride[0];
	cuint sw = _stride[1];
	
	cudaStream_t* p_st = st.get_stream();

	for (uint n = 0; n < (uint)in._n; ++n) {
		const nn_type* in_data = m_input + (n * in._c * in._h * in._w);
		nn_type* out_data = m_output + (n * out._c * out._h * out._w);
		uint* indice = m_indice + (n * out._c * out._h * out._w);

		__maxpool_2d<<<blocks, threads, smem_size, p_st[n % STREAMS]>>>(
			in_data,
			out_data,
			indice,
			(uint)in._h,
			(uint)in._w,
			(uint)in._c,
			(uint)out._h,
			(uint)out._w,
			kh,
			kw,
			sh,
			sw,
			(uint)tile_h,
			(uint)tile_w
		);
#if _DEBUG
		check_cuda(cudaStreamSynchronize(p_st[n % STREAMS]));
		check_cuda(cudaGetLastError());
#endif
	}
#if _DEBUG
	check_cuda(cudaDeviceSynchronize());
	check_cuda(cudaGetLastError());
#endif
}

NN_Backward* NN_Maxpool2D::create_backward(std::vector<bool>& mask) {
	return new NN_dMaxpool2D(*this);
}


/**********************************************/
/*                                            */
/*                NN_dMaxpool2D               */
/*                                            */
/**********************************************/

NN_dMaxpool2D::NN_dMaxpool2D(NN_Maxpool2D& layer) :
	NN_Backward_t(layer)
{
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