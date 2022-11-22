#include <iostream>
#include <random>
#include <time.h>
#include <tbb/tbb.h>
#include <vld.h>

#include "cuda_source/convolution.cuh"
#include "cuda_source/maxpool.cuh"
#include "cuda_source/k_convolution.cuh"

using namespace std;
using namespace tbb;


int main() {
	const int n = 64;
	const int ih = 122;
	const int iw = 122;
	const int ic = 32;

	const int k = 3;

	const int oh = ih - k + 1;
	const int ow = iw - k + 1;
	const int oc = 64;

	Tensor* hin = CreateHostTensor(n, ih, iw, ic);
	Tensor* hout = CreateHostTensor(n, oh, ow, oc);
	Tensor* hk = CreateHostTensor(oc, k, k, ic);

	Tensor* din = CreateDeviceTensor(n, ih, iw, ic);
	Tensor* dout = CreateDeviceTensor(n, oh, ow, oc);
	Tensor* dk = CreateDeviceTensor(oc, k, k, ic);

	Stream* st = CreateStreams(n);

	random_device rd;
	mt19937 gen(rd());
	uniform_real_distribution<float> dis(0, 1);

	try {
		
		parallel_for(
			blocked_range<uint>(0, n * ih * iw * ic),
			[&](blocked_range<uint>& q) {
			
			for (uint i = q.begin(); i < q.end(); ++i) {
				hin->data[i] = dis(gen);
			}
		}
		);
		parallel_for(
			blocked_range<uint>(0, n * oh * ow * oc),
			[&](blocked_range<uint>& q) {

			for (uint i = q.begin(); i < q.end(); ++i) {
				hout->data[i] = dis(gen);
			}
		}
		);

		MemCpy(hin, din);
		MemCpy(hout, dout);

		clock_t _start, _end;

		printf("start!!!!!!!!!!!!\n");
		_start = clock();

		kernel_conv_2d(
			st,
			din,
			dout,
			dk
		);
		
		_end = clock();
		printf("elapsed time = %dms\n", int(_end - _start));

		MemCpy(dk, hk);
	}
	catch (Exception& e) {
		e.Put();
	}

	FreeStreams(&st);
	FreeTensor(&hin);
	FreeTensor(&hout);
	FreeTensor(&hk);

	FreeTensor(&din);
	FreeTensor(&dout);
	FreeTensor(&dk);

	return 0;
}