#include <iostream>
#include <random>
#include <time.h>
#include <tbb/tbb.h>
#include <vld.h>

#include "cuda_source/convolution.cuh"
#include "cpp_source/convolution_host.h"
#include "cuda_source/maxpool.cuh"

using namespace std;
using namespace tbb;


void cpu_conv_2d_test(
	Tensor& input, 
	Tensor& kernel, 
	Tensor& output,
	int st_w,
	int st_h
) {
	clock_t start, end;
	
	start = clock();
	conv_2d_host(
		input,
		kernel,
		output,
		st_w,
		st_h
	);
	end = clock();

	printf("cpu elapsed time: %dms\n", int(end - start));
}

void cpu_correl_2d_test(
	Tensor& doutput,
	Tensor& kernel,
	Tensor& dinput
) {
	Tensor dpad;

	set_host_tensor(dpad, doutput.n, doutput.c, dinput.h + kernel.h - 1, dinput.w + kernel.w - 1);
	dilation_2d_host(doutput, dpad, 2, kernel.h - 1, kernel.w - 1);

	correl_2d_host(
		dpad,
		kernel,
		dinput
	);

	free_tensor(dpad);
}

void cpu_kernel_conv_2d_test(
	Tensor& doutput,
	Tensor& input,
	Tensor& gradient
) {
	kernel_conv_2d_host(
		doutput,
		input,
		gradient
	);
}


void gpu_kernel_conv_2d_test(
	Tensor& doutput,
	Tensor& input,
	Tensor& gradient
) {
	Tensor d_dout, d_in, d_grad;
	Stream s;

	set_like_tensor(d_dout, doutput, GPU, false);
	set_like_tensor(d_in, input, GPU, false);
	set_like_tensor(d_grad, gradient, GPU, true);

	create_streams(s, doutput.n);

	kernel_conv_2d(
		s,
		d_dout,
		d_in,
		d_grad
	);

	copy_tensor(d_grad, gradient);

	free_tensor(d_dout);
	free_tensor(d_in);
	free_tensor(d_grad);
	free_streams(s);
}

void gpu_correl_2d_test(
	Tensor& doutput,
	Tensor& kernel,
	Tensor& dinput
) {
	Stream s;
	Tensor d_doutput, d_kernel, d_dinput, d_dpad;

	create_streams(s, doutput.n);

	set_like_tensor(d_doutput, doutput, GPU, false);
	set_like_tensor(d_kernel, kernel, GPU, false);
	set_like_tensor(d_dinput, dinput, GPU, true);

	set_dev_tensor(d_dpad, doutput.n, doutput.c, dinput.h + kernel.h - 1, dinput.w + kernel.w - 1);

	dilation_2d(
		s,
		d_doutput,
		d_dpad,
		2,
		kernel.h - 1,
		kernel.w - 1
	);
	correl_2d(
		s,
		d_dpad,
		d_kernel,
		d_dinput
	);

	copy_tensor(d_dinput, dinput);

	free_tensor(d_doutput);
	free_tensor(d_kernel);
	free_tensor(d_dinput);
	free_tensor(d_dpad);
	free_streams(s);
}

void gpu_conv_2d_test(
	Tensor& input,
	Tensor& kernel,
	Tensor& output,
	int st_w,
	int st_h
) {
	clock_t start, end;
	Tensor d_input, d_kernel, d_output;
	Stream s;

	set_dev_tensor(
		d_input,
		input.n,
		input.c,
		input.h,
		input.w
	);
	set_dev_tensor(
		d_kernel,
		kernel.n,
		kernel.c,
		kernel.h,
		kernel.w
	);
	set_dev_tensor(
		d_output,
		output.n,
		output.c,
		output.h,
		output.w
	);
	create_streams(s, output.n);

	copy_tensor(input, d_input);
	copy_tensor(kernel, d_kernel);

	start = clock();
	conv_2d(
		s,
		d_input,
		d_kernel,
		d_output,
		st_w,
		st_h
	);
	end = clock();

	printf("gpu elapsed time: %dms\n", int(end - start));

	copy_tensor(d_output, output);

	free_tensor(d_input);
	free_tensor(d_kernel);
	free_tensor(d_output);
	free_streams(s);
}


int main() {
	const int n = 1;
	const int ih = 17;
	const int iw = 17;
	const int ic = 2;

	const int k = 3;
	const int stride = 1;

	const int oh = (ih - k) / stride + 1;
	const int ow = (iw - k) / stride + 1;
	const int oc = 3;

	Tensor hin, hout, hk;

	set_host_tensor(hin, n, ic, ih, iw);
	set_host_tensor(hout, n, oc, oh, ow);
	set_host_tensor(hk, oc, ic, k, k);

	random_device rd;
	mt19937 gen(rd());
	uniform_int_distribution<int> dis(1, 5);

	try {
		
		parallel_for(
			blocked_range<uint>(0, n * oc * oh * ow),
			[&](blocked_range<uint>& q) {
			
			for (uint i = q.begin(); i < q.end(); ++i) {
				hout.data[i] = float(dis(gen));
			}
		}
		);
		parallel_for(
			blocked_range<uint>(0, n * ic * ih * iw),
			[&](blocked_range<uint>& q) {

			for (uint i = q.begin(); i < q.end(); ++i) {
				hin.data[i] = float(dis(gen));
			}
		}
		);


		cpu_kernel_conv_2d_test(
			hout,
			hin,
			hk
		);
		print_tensor(hk);

		gpu_kernel_conv_2d_test(
			hout,
			hin,
			hk
		);
		print_tensor(hk);
	}
	catch (Exception& e) {
		e.Put();
	}

	free_tensor(hin);
	free_tensor(hout);
	free_tensor(hk);

	return 0;
}