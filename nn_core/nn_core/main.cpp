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


#if _DEBUG

int main() {
	const int n = 1;
	const int ih = 10;
	const int iw = 10;
	const int ic = 2;

	const int k = 3;

	const int oh = ih - k + 1;
	const int ow = iw - k + 1;
	const int oc = 2;

	Tensor* hin = CreateHostTensor(n, ih, iw, ic);
	Tensor* hout = CreateHostTensor(n, oh, ow, oc);
	Tensor* hk = CreateHostTensor(oc, k, k, ic);

	Tensor* din = CreateDeviceTensor(n, ih, iw, ic);
	Tensor* dout = CreateDeviceTensor(n, oh, ow, oc);
	Tensor* dk = CreateDeviceTensor(oc, k, k, ic);

	Stream* st = CreateStreams(n);

	try {
		for (int c = 0; c < ic; ++c) {
			for (int h = 0; h < ih; ++h) {
				float* p_in = hin->data + (c * hin->w * hin->h) + (h * hin->w);
				for (int w = 0; w < iw; ++w) {
					if (w < h) {
						if (c == 0) {
							p_in[w] = float(h);
						}
						else {
							p_in[w] = float(w);
						}
					}
					else {
						if (c == 0) {
							p_in[w] = float(w);
						}
						else {
							p_in[w] = float(h);
						}
					}
				}
			}
		}
		for (int c = 0; c < oc; ++c) {
			for (int h = 0; h < oh; ++h) {
				float* p_out = hout->data + (c * hout->w * hout->h) + (h * hout->w);
				for (int w = 0; w < ow; ++w) {
					if (w < h) {
						if (c == 0) {
							p_out[w] = float(h);
						}
						else {
							p_out[w] = float(w);
						}
					}
					else {
						if (c == 0) {
							p_out[w] = float(w);
						}
						else {
							p_out[w] = float(h);
						}
					}
				}
			}
		}

		MemCpy(hin, din);
		MemCpy(hout, dout);

		clock_t _start, _end;

		printf("start!!!!!!!!!!!!\n");
		_start = clock();

		kernel_conv_2d_1x1024_g_ind(
			st,
			din,
			dout,
			dk
		);

		_end = clock();
		printf("elapsed time = %dms\n", int(_end - _start));

		MemCpy(dk, hk);

		int i = 0;
		for (int _n = 0; _n < hin->n; ++_n) {
			for (int _c = 0; _c < hin->c; ++_c) {
				for (int _h = 0; _h < hin->h; ++_h) {
					for (int _w = 0; _w < hin->w; ++_w) {
						printf("%.0f ", hin->data[i++]);
					}
					printf("\n");
				}
				printf("\n\n");
			}
			printf("\n===============================================\n");
		}
		printf("\n");
		i = 0;
		for (int _n = 0; _n < hout->n; ++_n) {
			for (int _c = 0; _c < hout->c; ++_c) {
				for (int _h = 0; _h < hout->h; ++_h) {
					for (int _w = 0; _w < hout->w; ++_w) {
						printf("%.0f ", hout->data[i++]);
					}
					printf("\n");
				}
				printf("\n\n");
			}
			printf("\n===============================================\n");
		}
		printf("\n");
		i = 0;
		for (int _n = 0; _n < hk->n; ++_n) {
			for (int _c = 0; _c < hk->c; ++_c) {
				for (int _h = 0; _h < hk->h; ++_h) {
					for (int _w = 0; _w < hk->w; ++_w) {
						printf("%.0f ", hk->data[i++]);
					}
					printf("\n");
				}
				printf("\n\n");
			}
			printf("\n===============================================\n");
		}
		printf("\n");

		kernel_conv_2d_32x32_g_ind(
			st,
			din,
			dout,
			dk
		);


		i = 0;
		for (int _n = 0; _n < hk->n; ++_n) {
			for (int _c = 0; _c < hk->c; ++_c) {
				for (int _h = 0; _h < hk->h; ++_h) {
					for (int _w = 0; _w < hk->w; ++_w) {
						printf("%.0f ", hk->data[i++]);
					}
					printf("\n");
				}
				printf("\n\n");
			}
			printf("\n===============================================\n");
		}
		printf("\n");
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

#else

int main() {
	const int n = 64;
	const int ih = 256;
	const int iw = 256;
	const int ic = 3;

	const int k = 3;

	const int oh = ih - k + 1;
	const int ow = iw - k + 1;
	const int oc = 32;

	Tensor* hin = CreateHostTensor(n, ih, iw, ic);
	Tensor* hout = CreateHostTensor(n, oh, ow, oc);
	Tensor* hk = CreateHostTensor(oc, k, k, ic);

	Tensor* din = CreateDeviceTensor(n, ih, iw, ic);
	Tensor* dout = CreateDeviceTensor(n, oh, ow, oc);
	Tensor* dk = CreateDeviceTensor(oc, k, k, ic);

	Stream* st = CreateStreams(n);

	try {
		memset(hin->data, 0, GetTotalSize(hin));
		memset(hout->data, 0, GetTotalSize(hout));

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

		/*
		int i = 0;
		for (int _n = 0; _n < hk->n; ++_n) {
			for (int _c = 0; _c < hk->c; ++_c) {
				for (int _h = 0; _h < hk->h; ++_h) {
					for (int _w = 0; _w < hk->w; ++_w) {
						printf("%.0f ", hk->data[i++]);
					}
					printf("\n");
				}
				printf("\n\n");
			}
			printf("\n===============================================\n");
		}
		printf("\n");
		*/
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

#endif