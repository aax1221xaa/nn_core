#include <iostream>
#include <random>
#include <time.h>
#include <tbb/tbb.h>
#include <vld.h>

#include "cuda_source/convolution.cuh"
#include "cuda_source/maxpool.cuh"

using namespace std;
using namespace tbb;


void print(const Tensor& t) {
	printf("\nTensor shape = %s", dim_to_str(t));
	for (int n = 0; n < t.n; ++n) {
		float* npt = t.data + (n * t.h * t.w * t.c);
		printf("\nn = %d\n===================================================\n", n);
		for (int c = 0; c < t.c; ++c) {
			float* cpt = npt + (c * t.h * t.w);
			printf("\nc = %d, (%dx%d)\n", c, t.h, t.w);
			for (int h = 0; h < t.h; ++h) {
				float* hpt = cpt + (h * t.w);
				for (int w = 0; w < t.w; ++w) {
					printf("%.0f ", hpt[w]);
				}
				printf("\n");
			}
		}
	}
	printf("\n");
}

int main() {
	const int n = 1;
	const int ih = 17;
	const int iw = 17;
	const int ic = 2;

	const int k = 3;
	const int stride = 2;

	const int oh = (ih - k) / stride + 1;
	const int ow = (iw - k) / stride + 1;
	const int oc = 3;

	const int doh = ih + k - 1;
	const int dow = iw + k - 1;
	const int offset = k - 1;

	Tensor hin, hout, hk, hpad;
	Tensor din, dout, dk, dpad;
	Stream st;

	create_host_tensor(hin, n, ic, ih, iw);
	create_host_tensor(hout, n, oc, oh, ow);
	create_host_tensor(hk, oc, ic, k, k);
	create_host_tensor(hpad, n, oc, doh, dow);

	create_dev_tensor(din, n, ic, ih, iw);
	create_dev_tensor(dout, n, oc, oh, ow);
	create_dev_tensor(dk, oc, ic, k, k);
	create_dev_tensor(dpad, n, oc, doh, dow);

	create_streams(st, n);

	random_device rd;
	mt19937 gen(rd());
	uniform_int_distribution<int> dis(1, 5);

	try {
		
		parallel_for(
			blocked_range<uint>(0, n * oh * ow * oc),
			[&](blocked_range<uint>& q) {
			
			for (uint i = q.begin(); i < q.end(); ++i) {
				hout.data[i] = float(dis(gen));
			}
		}
		);
		parallel_for(
			blocked_range<uint>(0, oc * k * k * ic),
			[&](blocked_range<uint>& q) {

			for (uint i = q.begin(); i < q.end(); ++i) {
				hk.data[i] = float(dis(gen));
			}
		}
		);

		copy_tensor(hout, dout);
		copy_tensor(hk, dk);

		
		dilation_2d(
			st,
			dout,
			dpad,
			stride,
			offset,
			offset
		);

		correl_2d(st, dpad, dk, din);
	
		copy_tensor(din, hin);
		copy_tensor(dpad, hpad);

		print(hout);
		print(hk);
		print(hpad);
		print(hin);
	}
	catch (Exception& e) {
		e.Put();
	}

	free_streams(st);

	free_tensor(hin);
	free_tensor(hout);
	free_tensor(hk);
	free_tensor(hpad);

	free_tensor(din);
	free_tensor(dout);
	free_tensor(dk);
	free_tensor(dpad);

	return 0;
}