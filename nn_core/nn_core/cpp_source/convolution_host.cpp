//#include "convolution_host.h"
//#include <tbb/tbb.h>
//
//using namespace tbb;
//
//
//int get_output_size_host(
//	int input_size,
//	int kernel_size,
//	int pad_size,
//	int stride
//) {
//	return (input_size + (2 * pad_size) - kernel_size) / stride + 1;
//}
//
//void check_conv_2d_host(
//	const Tensor& d_input,
//	const Tensor& d_kernel,
//	const Tensor& d_output,
//	int st_w,
//	int st_h
//) {
//	int out_h = get_output_size_host(d_input.h, d_kernel.h, 0, st_h);
//	int out_w = get_output_size_host(d_input.w, d_kernel.w, 0, st_w);
//
//	if (d_output.h != out_h || d_output.w != out_w) {
//		ErrorExcept(
//			"[check_conv_2d_host] invalid output dimension %s",
//			dim_to_str(d_output)
//		);
//	}
//	else if (d_kernel.c != d_input.c || d_kernel.n != d_output.c) {
//		ErrorExcept(
//			"[check_conv_2d_host] invalid channels input: %s, kernel: %s, output: %s",
//			dim_to_str(d_input),
//			dim_to_str(d_kernel),
//			dim_to_str(d_output)
//		);
//	}
//}
//
//void check_dilation_2d_host(
//	const Tensor& input,
//	const Tensor& output,
//	uint scale,
//	int offset_x,
//	int offset_y
//) {
//	int out_w = input.w * scale + offset_x;
//	int out_h = input.h * scale + offset_y;
//
//	if (out_w > output.w || out_h > output.h) {
//		ErrorExcept(
//			"[check_dilation_2d_host] output is too small. output: %s, expect output: [%d, %d, %d, %d]",
//			dim_to_str(output),
//			output.n,
//			output.c,
//			out_h,
//			out_w
//		);
//	}
//}
//
//void check_correl_2d_host(
//	const Tensor& d_doutput,
//	const Tensor& d_kernel,
//	const Tensor& d_dinput
//) {
//	int d_in_w = d_doutput.w - d_kernel.w + 1;
//	int d_in_h = d_doutput.h - d_kernel.h + 1;
//
//	if (
//		d_doutput.c != d_kernel.n ||
//		d_dinput.c != d_kernel.c ||
//		d_dinput.w != d_in_w ||
//		d_dinput.h != d_in_h
//		) {
//		ErrorExcept(
//			"[check_correl_2d_host] invalid (d_output, kernel, d_input) size. d_doutput: %s, d_tkernel: %s, d_dinput: %s",
//			dim_to_str(d_doutput), dim_to_str(d_kernel), dim_to_str(d_dinput)
//		);
//	}
//}
//
//void check_kernel_conv_2d_host(
//	const Tensor& doutput,
//	const Tensor& input,
//	Tensor& gradient
//) {
//	int in_h = input.h - doutput.h + 1;
//	int in_w = input.w - doutput.w + 1;
//
//	if (gradient.h != in_h ||
//		gradient.w != in_w ||
//		gradient.n != doutput.c ||
//		gradient.c != input.c ||
//		input.n != doutput.n) {
//
//		ErrorExcept(
//			"[check_kernel_conv_2d_host] invalid tensor arguments size. input: %s, doutput: %s, gradient: %s",
//			dim_to_str(input),
//			dim_to_str(doutput),
//			dim_to_str(gradient)
//		);
//	}
//}
//
//
//void conv_2d_host(const Tensor & h_input, const Tensor & h_kernel, Tensor & h_output, int st_w, int st_h)
//{
//	check_conv_2d_host(
//		h_input,
//		h_kernel,
//		h_output,
//		st_w,
//		st_h
//	);
//
//	uint* indices = new uint[h_kernel.c * h_kernel.h * h_kernel.w];
//
//	for (int c = 0; c < h_kernel.c; ++c) {
//		uint* p_indices = indices + (c * h_kernel.h * h_kernel.w);
//		for (int h = 0; h < h_kernel.h; ++h) {
//			for (int w = 0; w < h_kernel.w; ++w) {
//				p_indices[h * h_kernel.w + w] = (c * h_input.h * h_input.w) + (h * h_input.w) + w;
//			}
//		}
//	}
//
//	parallel_for(
//		blocked_range2d<uint>(0, h_output.c, 0, h_output.h * h_output.w),
//		[&](blocked_range2d<uint>& q) {
//
//		cuint k_len = h_kernel.c * h_kernel.h * h_kernel.w;
//
//		for (int batch = 0; batch < h_output.n; ++batch) {
//			float* h_in = h_input.data + (batch * h_input.c * h_input.h * h_input.w);
//			float* h_out = h_output.data + (batch * h_output.c * h_output.h * h_output.w);
//
//			for (uint m = q.rows().begin(); m < q.rows().end(); ++m) {
//				float* p_out = h_out + (m * h_output.h * h_output.w);
//				float* p_kernel = h_kernel.data + (m * h_kernel.c * h_kernel.h * h_kernel.w);
//
//				for (uint k = q.cols().begin(); k < q.cols().end(); ++k) {
//					float sum_ = 0.f;
//
//					cuint x0 = (k % h_output.w) * st_w;
//					cuint y0 = (k / h_output.w) * st_h;
//	
//					float* p_in = h_in + (y0 * h_input.w + x0);
//
//					for (uint e = 0; e < k_len; ++e) {
//						sum_ += p_in[indices[e]] * p_kernel[e];
//					}
//
//					p_out[k] = sum_;
//				}
//			}
//		}
//	}
//	);
//
//	delete[] indices;
//}
//
//void correl_2d_host(const Tensor & h_doutput, const Tensor & h_kernel, Tensor & h_dinput)
//{
//	check_correl_2d_host(
//		h_doutput,
//		h_kernel,
//		h_dinput
//	);
//
//	uint* indices = new uint[get_elem_size(h_kernel)];
//
//	for (int n = 0; n < h_kernel.n; ++n) {
//		uint* p_indices = indices + (n * h_kernel.h * h_kernel.w);
//		for (int h = 0; h < h_kernel.h; ++h) {
//			for (int w = 0; w < h_kernel.w; ++w) {
//				p_indices[h * h_kernel.w + w] = (n * h_doutput.h * h_doutput.w) + ((h_kernel.h - h - 1) * h_doutput.w) + (h_kernel.w - w - 1);
//			}
//		}
//	}
//
//	Tensor t_kernel;
//	set_host_tensor(t_kernel, h_kernel.c, h_kernel.n, h_kernel.h, h_kernel.w);
//
//	transpose_host(h_kernel, t_kernel);
//
//	parallel_for(
//		blocked_range2d<uint>(0, h_dinput.c, 0, h_dinput.h * h_dinput.w),
//		[&](blocked_range2d<uint>& q) {
//
//		cuint k_len = t_kernel.c * t_kernel.h * t_kernel.w;
//
//		for (int batch = 0; batch < h_dinput.n; ++batch) {
//			float* h_din = h_dinput.data + (batch * h_dinput.c * h_dinput.h * h_dinput.w);
//			float* h_dout = h_doutput.data + (batch * h_doutput.c * h_doutput.h * h_doutput.w);
//
//			for (uint m = q.rows().begin(); m < q.rows().end(); ++m) {
//				float* p_din = h_din + (m * h_dinput.h * h_dinput.w);
//				float* p_kernel = t_kernel.data + (m * t_kernel.c * t_kernel.h * t_kernel.w);
//
//				for (uint k = q.cols().begin(); k < q.cols().end(); ++k) {
//					float sum_ = 0.f;
//
//					cuint x0 = k % h_dinput.w;
//					cuint y0 = k / h_dinput.w;
//
//					float* p_dout = h_dout + (y0 * h_doutput.w + x0);
//
//					for (uint e = 0; e < k_len; ++e) {
//						sum_ += p_dout[indices[e]] * p_kernel[e];
//					}
//
//					p_din[k] = sum_;
//				}
//			}
//		}
//	}
//	);
//
//	delete[] indices;
//	free_tensor(t_kernel);
//}
//
//void transpose_host(
//	const Tensor& h_input,
//	Tensor& h_output
//) {
//	if (get_elem_size(h_input) != get_elem_size(h_output)) {
//		ErrorExcept(
//			"[transpose_host] invalid input, output size. input: %s, output: %s",
//			dim_to_str(h_input),
//			dim_to_str(h_output)
//		);
//	}
//
//	parallel_for(
//		blocked_range<uint>(0, get_elem_size(h_input)),
//		[&](blocked_range<uint>& q) {
//
//		for (uint i = q.begin(); i < q.end(); ++i) {
//			uint wh_idx = i % (h_input.w * h_input.h);
//			uint page = i / (h_input.w * h_input.h);
//
//			cuint c = page % h_input.c;
//			cuint n = page / h_input.c;
//
//			float* p_output = h_output.data + (c * h_output.c * h_output.h * h_output.w) + n * (h_output.h * h_output.w);
//
//			p_output[wh_idx] = h_input.data[i];
//		}
//	}
//	);
//}
//
//void dilation_2d_host(const Tensor & h_input, Tensor & h_output, uint scale, int offset_x, int offset_y)
//{
//	check_dilation_2d_host(
//		h_input,
//		h_output,
//		scale,
//		offset_x,
//		offset_y
//	);
//
//	memset(h_output.data, 0, get_mem_size(h_output));
//
//	parallel_for(
//		blocked_range3d<uint>(
//			0, h_input.n * h_output.c,
//			0, h_input.h,
//			0, h_input.w
//			),
//		[&](blocked_range3d<uint>& q) {
//
//		for (uint i = q.pages().begin(); i < q.pages().end(); ++i) {
//			float* p_in = h_input.data + (i * h_input.h * h_input.w);
//			float* p_out = h_output.data + (i * h_output.h * h_output.w);
//
//			for (uint h = q.rows().begin(); h < q.rows().end(); ++h) {
//				for (uint w = q.cols().begin(); w < q.cols().end(); ++w) {
//					p_out[(h * scale + offset_y) * h_output.w + (w * scale + offset_x)] = p_in[h * h_input.w + w];
//				}
//			}
//		}
//	}
//	);
//}
//
//void kernel_conv_2d_host(
//	const Tensor& h_doutput,
//	const Tensor& h_input,
//	Tensor& h_gradient
//){
//	check_kernel_conv_2d_host(
//		h_doutput,
//		h_input,
//		h_gradient
//	);
//
//	memset(h_gradient.data, 0, get_mem_size(h_gradient));
//
//	MemBlock<uint> indices;
//	create_host_memblock(indices, h_doutput.h * h_doutput.w);
//
//	for (uint h = 0; h < h_doutput.h; ++h) {
//		for (uint w = 0; w < h_doutput.w; ++w) {
//			indices.data[h * h_doutput.w + w] = h * h_input.w + w;
//		}
//	}
//
//	parallel_for(
//		blocked_range3d<uint>(
//			0, h_input.n,
//			0, h_gradient.n,
//			0, h_gradient.c * h_gradient.h * h_gradient.w
//			),
//		[&](blocked_range3d<uint>& q) {
//
//		cuint h_dout_wh = h_doutput.h * h_doutput.w;
//
//		for (uint batch = q.pages().begin(); batch < q.pages().end(); ++batch) {
//			float* p_dout = h_doutput.data + (batch * h_doutput.c * h_doutput.h * h_doutput.w);
//			float* p_in = h_input.data + (batch * h_input.c * h_input.h * h_input.w);
//
//			for (uint m = q.rows().begin(); m < q.rows().end(); ++m) {
//				float* pdout = p_dout + (m * h_doutput.h * h_doutput.w);
//				float* pgrad = h_gradient.data + (m * h_gradient.c * h_gradient.h * h_gradient.w);
//
//				for (uint k = q.cols().begin(); k < q.cols().end(); ++k) {
//					cuint x0 = k % h_gradient.w;
//					cuint y0 = (k / h_gradient.w) % h_gradient.h;
//					cuint c0 = k / (h_gradient.w * h_gradient.h);
//
//					float* pin = p_in + (c0 * (h_input.h * h_input.w) + (y0 * h_input.w) + x0);
//					float _sum = 0.f;
//
//					for (uint e = 0; e < h_dout_wh; ++e) {
//						_sum += pdout[e] * pin[indices.data[e]];
//					}
//
//					pgrad[k] += _sum;
//				}
//			}
//		}
//	}
//	);
//
//	free_memblock(indices);
//}
