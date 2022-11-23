#include <iostream>
#include <random>
#include <time.h>
#include <tbb/tbb.h>
#include <vld.h>

#include "cuda_source/convolution.cuh"
#include "cuda_source/maxpool.cuh"
#include "cuda_source/d_convolution.cuh"

using namespace std;
using namespace tbb;


void print(const Tensor* t) {
	printf("\nTensor shape = [%d, %d, %d, %d]", t->n, t->c, t->h, t->w);
	for (int n = 0; n < t->n; ++n) {
		float* npt = t->data + (n * t->h * t->w * t->c);
		printf("\nn = %d\n===================================================\n", n);
		for (int c = 0; c < t->c; ++c) {
			float* cpt = npt + (c * t->h * t->w);
			printf("\nc = %d, (%dx%d)\n", c, t->h, t->w);
			for (int h = 0; h < t->h; ++h) {
				float* hpt = cpt + (h * t->w);
				for (int w = 0; w < t->w; ++w) {
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
	const int ih = 16;
	const int iw = 16;
	const int ic = 3;

	const int k = 3;
	const int stride = 2;

	const int oh = (ih - k) / stride + 1;
	const int ow = (iw - k) / stride + 1;
	const int oc = 3;

	const int doh = ih + k - 1;
	const int dow = iw + k - 1;
	const int offset = k - 1;

	Tensor* hin = CreateHostTensor(n, ih, iw, ic);
	Tensor* hout = CreateHostTensor(n, oh, ow, oc);
	Tensor* hk = CreateHostTensor(oc, k, k, ic);
	Tensor* htk = CreateHostTensor(oc, k, k, ic);
	Tensor* hpad = CreateHostTensor(n, doh, dow, oc);

	Tensor* din = CreateDeviceTensor(n, ih, iw, ic);
	Tensor* dout = CreateDeviceTensor(n, oh, ow, oc);
	Tensor* dk = CreateDeviceTensor(oc, k, k, ic);
	Tensor* dtk = CreateDeviceTensor(oc, k, k, ic);
	Tensor* dpad = CreateDeviceTensor(n, doh, dow, oc);

	Stream* st = CreateStreams(n);

	random_device rd;
	mt19937 gen(rd());
	uniform_int_distribution<int> dis(1, 9);

	try {
		
		parallel_for(
			blocked_range<uint>(0, n * oh * ow * oc),
			[&](blocked_range<uint>& q) {
			
			for (uint i = q.begin(); i < q.end(); ++i) {
				hout->data[i] = float(dis(gen));
			}
		}
		);
		parallel_for(
			blocked_range<uint>(0, oc * k * k * ic),
			[&](blocked_range<uint>& q) {

			for (uint i = q.begin(); i < q.end(); ++i) {
				hk->data[i] = float(dis(gen));
			}
		}
		);

		MemCpy(hout, dout);
		MemCpy(hk, dk);
		checkCuda(cudaMemset(dpad->data, 0, GetTotalSize(dpad)));

		dilation_2d(
			st,
			dout,
			dpad,
			stride,
			offset,
			offset
		);
		correl_2d(
			st,
			dpad,
			dk,
			din,
			dtk
		);

		MemCpy(din, hin);
		MemCpy(dpad, hpad);
		MemCpy(dtk, htk);

		print(hout);
		print(hpad);
		print(hk);
		print(htk);
		print(hin);
	}
	catch (Exception& e) {
		e.Put();
	}

	FreeStreams(&st);
	FreeTensor(&hin);
	FreeTensor(&hout);
	FreeTensor(&hk);
	FreeTensor(&htk);
	FreeTensor(&hpad);

	FreeTensor(&din);
	FreeTensor(&dout);
	FreeTensor(&dk);
	FreeTensor(&dtk);
	FreeTensor(&dpad);

	return 0;
}