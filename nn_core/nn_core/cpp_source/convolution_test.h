#pragma once

#include "cuda_common.h"


void conv_2d_host(
	const Tensor& h_input,
	const Tensor& h_kernel,
	Tensor& h_output,
	int st_w,
	int st_h
);

void correl_2d_host(
	const Tensor& h_doutput,
	const Tensor& h_kernel,
	Tensor& h_dinput
);

void transpose_host(
	const Tensor& h_input,
	Tensor& h_output
);

void dilation_2d_host(
	const Tensor& h_input,
	Tensor& h_output,
	uint scale,
	int offset_x,
	int offset_y
);

void kernel_conv_2d_host(
	const Tensor& h_input,
	const Tensor& h_doutput,
	Tensor& h_gradient
);