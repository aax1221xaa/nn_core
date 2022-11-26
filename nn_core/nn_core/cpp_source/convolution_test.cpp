#include "convolution_test.h"
#include <tbb/tbb.h>

using namespace tbb;


int get_output_size_host(
	int input_size,
	int kernel_size,
	int pad_size,
	int stride
) {
	return (input_size + (2 * pad_size) - kernel_size) / stride + 1;
}

void check_conv_2d_host(
	const Tensor& d_input,
	const Tensor& d_kernel,
	const Tensor& d_output,
	int st_w,
	int st_h
) {
	int out_h = get_output_size_host(d_input.h, d_kernel.h, 0, st_h);
	int out_w = get_output_size_host(d_input.w, d_kernel.w, 0, st_w);

	if (d_output.h != out_h || d_output.w != out_w) {
		ErrorExcept(
			"[check_conv_2d] invalid output dimension %s",
			dim_to_str(d_output)
		);
	}
	else if (d_kernel.c != d_input.c || d_kernel.n != d_output.c) {
		ErrorExcept(
			"[check_conv_2d] invalid channels input: %s, kernel: %s, output: %s",
			dim_to_str(d_input),
			dim_to_str(d_kernel),
			dim_to_str(d_output)
		);
	}
}

void check_dilation_2d_host(
	const Tensor& input,
	const Tensor& output,
	uint scale,
	int offset_x,
	int offset_y
) {
	int out_w = input.w * scale + offset_x;
	int out_h = input.h * scale + offset_y;

	if (out_w > output.w || out_h > output.h) {
		ErrorExcept(
			"[check_dilation_2d] output is too small. output: %s, expect output: [%d, %d, %d, %d]",
			dim_to_str(output),
			output.n,
			output.c,
			out_h,
			out_w
		);
	}
}

void check_correl_2d_host(
	const Tensor& d_doutput,
	const Tensor& d_kernel,
	const Tensor& d_dinput
) {
	int d_in_w = d_doutput.w - d_kernel.w + 1;
	int d_in_h = d_doutput.h - d_kernel.h + 1;

	if (
		d_doutput.c != d_kernel.n ||
		d_dinput.c != d_kernel.c ||
		d_dinput.w != d_in_w ||
		d_dinput.h != d_in_h
		) {
		ErrorExcept(
			"[check_correl_2d] invalid (d_output, kernel, d_input) size. d_doutput: %s, d_tkernel: %s, d_dinput: %s",
			dim_to_str(d_doutput), dim_to_str(d_kernel), dim_to_str(d_dinput)
		);
	}
}

void check_kernel_conv_2d_host(
	const Tensor& d_input,
	const Tensor& d_doutput,
	Tensor& d_gradient
) {
	int in_h = d_input.h - d_doutput.h + 1;
	int in_w = d_input.w - d_doutput.w + 1;

	if (d_gradient.h != in_h ||
		d_gradient.w != in_w ||
		d_gradient.n != d_doutput.c ||
		d_gradient.c != d_input.c ||
		d_input.n != d_doutput.n) {

		ErrorExcept(
			"[check_kernel_conv_2d] invalid tensor arguments size. d_input: %s, d_doutput: %s, gradient: %s",
			dim_to_str(d_input),
			dim_to_str(d_doutput),
			dim_to_str(d_gradient)
		);
	}
}


void conv_2d_host(const Tensor & h_input, const Tensor & h_kernel, Tensor & h_output, int st_w, int st_h)
{
	check_conv_2d_host(
		h_input,
		h_kernel,
		h_output,
		st_w,
		st_h
	);

	uint* indices = new uint[h_kernel.c * h_kernel.h * h_kernel.w];

	for (int c = 0; c < h_kernel.c; ++c) {
		uint* p_indices = indices + (c * h_kernel.h * h_kernel.w);
		for (int h = 0; h < h_kernel.h; ++h) {
			for (int w = 0; w < h_kernel.w; ++w) {
				p_indices[h * h_kernel.w + w] = (c * h_input.h * h_input.w) + (h * h_input.w) + w;
			}
		}
	}

	parallel_for(
		blocked_range2d<uint>(0, h_output.c, 0, h_output.h * h_output.w),
		[&](blocked_range2d<uint>& q) {

		cuint k_len = h_kernel.c * h_kernel.h * h_kernel.w;

		for (int batch = 0; batch < h_output.n; ++batch) {
			float* h_in = h_input.data + (batch * h_input.c * h_input.h * h_input.w);
			float* h_out = h_output.data + (batch * h_output.c * h_output.h * h_output.w);

			for (uint m = q.rows().begin(); m < q.rows().end(); ++m) {
				float* p_out = h_out + (m * h_output.h * h_output.w);
				float* p_kernel = h_kernel.data + (m * h_kernel.c * h_kernel.h * h_kernel.w);

				for (uint k = q.cols().begin(); k < q.cols().end(); ++k) {
					float sum_ = 0.f;

					cuint x0 = k % h_output.w;
					cuint y0 = k / h_output.w;
	
					float* p_in = h_in + (y0 * h_input.w + x0);

					for (uint e = 0; e < k_len; ++e) {
						sum_ += p_in[indices[e]] * p_kernel[e];
					}

					p_out[m * h_output.w + k] = sum_;
				}
			}
		}
	}
	);

	delete[] indices;
}

void correl_2d_host(const Tensor & h_doutput, const Tensor & h_kernel, Tensor & h_dinput)
{
	check_correl_2d_host(
		h_doutput,
		h_kernel,
		h_dinput
	);

	uint* indices = new uint[get_elem_size(h_kernel)];

	for (int n = 0; n < h_kernel.n; ++n) {
		uint* p_indices = indices + (n * h_kernel.h * h_kernel.w);
		for (int h = 0; h < h_kernel.h; ++h) {
			for (int w = 0; w < h_kernel.w; ++w) {
				p_indices[h * h_kernel.w + w] = (n * h_doutput.h * h_doutput.w) + ((h_kernel.h - h - 1) * h_doutput.w) + (h_kernel.w - w - 1);
			}
		}
	}

	Tensor t_kernel;
	create_host_tensor(t_kernel, h_kernel.c, h_kernel.n, h_kernel.h, h_kernel.w);

	parallel_for(
		blocked_range2d<uint>(0, h_dinput.c, 0, h_dinput.h * h_dinput.w),
		[&](blocked_range2d<uint>& q) {

		cuint k_len = t_kernel.c * t_kernel.h * t_kernel.w;

		for (int batch = 0; batch < h_dinput.n; ++batch) {
			float* h_din = h_dinput.data + (batch * h_dinput.c * h_dinput.h * h_dinput.w);
			float* h_dout = h_doutput.data + (batch * h_doutput.c * h_doutput.h * h_doutput.w);

			for (uint m = q.rows().begin(); m < q.rows().end(); ++m) {
				float* p_din = h_din + (m * h_dinput.h * h_dinput.w);
				float* p_kernel = t_kernel.data + (m * t_kernel.c * t_kernel.h * t_kernel.w);

				for (uint k = q.cols().begin(); k < q.cols().end(); ++k) {
					float sum_ = 0.f;

					cuint x0 = k % h_dinput.w;
					cuint y0 = k / h_dinput.w;

					float* p_dout = h_dout + (y0 * h_doutput.w + x0);

					for (uint e = 0; e < k_len; ++e) {
						sum_ += p_dout[indices[e]] * p_kernel[e];
					}

					p_din[m * h_dinput.w + k] = sum_;
				}
			}
		}
	}
	);

	delete[] indices;
}

void transpose_host(
	const Tensor& h_input,
	Tensor& h_output
) {
	parallel_for(
		blocked_range<uint>(0, get_elem_size(h_input)),
		[&](blocked_range<uint>& q) {

		for (uint i = q.begin(); i < q.end(); ++i) {
			uint wh_idx = i % (h_input.w * h_input.h);
			uint count = i / (h_input.w * h_input.h);

			cuint row = count % h_input.n;
			cuint col = count / h_input.n;

			float* p_input = h_input.data + (row * (h_input.w * h_input.h * h_input.n) + col * (h_input.w * h_input.h));

			h_output.data[i] = p_input[wh_idx];
		}
	}
	);
}

void dilation_2d_host(const Tensor & h_input, Tensor & h_output, uint scale, int offset_x, int offset_y)
{
}

void kernel_conv_2d_host(const Tensor & h_input, const Tensor & h_doutput, Tensor & h_gradient)
{
}
