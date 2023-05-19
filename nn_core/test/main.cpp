#include "../nn_core/cpp_source/nn_tensor.h"
#include "../nn_core/cuda_source/maxpool.cuh"
#include <time.h>
#include "tbb/tbb.h"
#include "vld.h"

using namespace tbb;


void maxpool2d_cpu(
	const tensor<float> input,
	tensor<float> output,
	tensor<uint> mark,
	cuint kernel_w,
	cuint kernel_h,
	cuint stride_w,
	cuint stride_h
) {
	for (uint n = 0; n < output.n; ++n) {
		const float* n_in = input.data + (n * input.c * input.h * input.w);
		float* n_out = output.data + (n * output.c * output.h * output.w);
		uint* n_mark = mark.data + (n * mark.c * mark.h * mark.w);

		parallel_for(
			blocked_range3d<uint>(
				0, output.c,
				0, output.h,
				0, output.w
				),
			[&](const blocked_range3d<uint>& e) {

			for (uint c = e.pages().begin(); c < e.pages().end(); ++c) {
				const float* c_in = n_in + (c * input.h * input.w);
				float* c_out = n_out + (c * output.h * output.w);
				uint* c_mark = n_mark + (c * mark.h * mark.w);

				for (uint y = e.rows().begin(); y < e.rows().end(); ++y) {
					const float* y_in = c_in + (y * stride_h * input.w);
					float* y_out = c_out + (y * output.w);
					uint* y_mark = c_mark + (y * mark.w);

					for (uint x = e.cols().begin(); x < e.cols().end(); ++x) {
						const float* x_in = y_in + x * stride_w;
						float* x_out = y_out + x;
						uint* x_mark = y_mark + x;

						float val = 0;
						float max_val = -(FLT_MAX);
						uint index = 0;

						for (uint h = 0; h < kernel_h; ++h) {
							for (uint w = 0; w < kernel_w; ++w) {
								val = x_in[h * input.w + w];

								if (val > max_val) {
									max_val = val;
									index = h * kernel_w + w;
								}
							}
						}
						*x_out = max_val;
						*x_mark = index;
					}
				}
			}
		});
	}
}

template <typename _T>
uint verify_data(
	const tensor<_T> data_1,
	const tensor<_T> data_2
) {
	uint count = 0;

	parallel_for(
		blocked_range<uint>(
			0, data_1.n * data_1.c * data_1.h * data_1.w
		),
		[&](const blocked_range<uint>& e) {

		for (uint i = e.begin(); i < e.end(); ++i) {
			if (data_1.data[i] != data_2.data[i]) ++count;
		}
	});

	return count;
}


int main() {
	try {
		int kernel_h = 4;
		int kernel_w = 4;
		int stride_h = 2;
		int stride_w = 2;

		int batch = 64;
		int channel = 32;
		int in_h = 224;
		int in_w = 224;
		int out_h = (in_h - kernel_h) / stride_h + 1;
		int out_w = (in_w - kernel_w) / stride_w + 1;

		Tensor<float> input({ batch, channel, in_h, in_w });
		Tensor<float> output({ batch, channel, out_h, out_w });
		Tensor<uint> mark({ batch, channel, out_h, out_w });

		Tensor<float> output2({ batch, channel, out_h, out_w });
		Tensor<uint> mark2({ batch, channel, out_h, out_w });

		tensor<float> d_input = { NULL, batch, channel, in_h, in_w};
		tensor<float> d_output = { NULL, batch, channel, out_h, out_w };
		tensor<uint> d_mark = { NULL, batch, channel, out_h, out_w };

		cudaStream_t streams[STREAMS] = { NULL, };

		parallel_for(blocked_range<uint>(0, input._len),
			[&](const blocked_range<uint>& e) {
			for (uint i = e.begin(); i < e.end(); ++i) {
				input._data[i] = roundf(((float)rand() / RAND_MAX) * 10.f);
			}
		});

		check_cuda(cudaMalloc(&d_input.data, sizeof(float) * input._len));
		check_cuda(cudaMalloc(&d_output.data, sizeof(float) * output._len));
		check_cuda(cudaMalloc(&d_mark.data, sizeof(uint) * mark._len));

		check_cuda(cudaMemcpy(d_input.data, input._data, sizeof(float) * input._len, cudaMemcpyHostToDevice));
		
		for (int i = 0; i < STREAMS; ++i) check_cuda(cudaStreamCreate(&streams[i]));

		clock_t start = clock();
		maxpool2d(
			streams,
			d_input,
			d_output,
			d_mark,
			kernel_w,
			kernel_h,
			stride_w,
			stride_h
		);
		
		clock_t end = clock();

		check_cuda(cudaMemcpy(output._data, d_output.data, sizeof(float) * output._len, cudaMemcpyDeviceToHost));
		check_cuda(cudaMemcpy(mark._data, d_mark.data, sizeof(uint) * mark._len, cudaMemcpyDeviceToHost));

		check_cuda(cudaFree(d_input.data));
		check_cuda(cudaFree(d_output.data));
		check_cuda(cudaFree(d_mark.data));
		for (int i = 0; i < STREAMS; ++i) check_cuda(cudaStreamDestroy(streams[i]));
		
		std::cout << "elapsed tile = " << (long)(end - start) << "ms" << std::endl;

		tensor<float> h_input = { input._data, batch, channel, in_h, in_w };
		tensor<float> h_output = { output2._data, batch, channel, out_h, out_w };
		tensor<uint> h_mark = { mark2._data, batch, channel, out_h, out_w };

		maxpool2d_cpu(h_input, h_output, h_mark, kernel_w, kernel_h, stride_w, stride_h);

		tensor<float> g_output = { output._data, batch, channel, out_h, out_w };
		tensor<uint> g_mark = { mark._data, batch, channel, out_h, out_w };

		uint count = verify_data(h_output, g_output);
		printf("fails data count = %d\n", count);
		
		count = verify_data(h_mark, g_mark);
		printf("fails indice count = %d\n", count);
	}
	catch (Exception& p) {
		p.Put();
	}

	return 0;
}