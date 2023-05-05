#include <iostream>
#include <random>
#include <time.h>
#include <vld.h>

#include "cpp_source/nn_model.h"
#include "cpp_source/nn_layer.h"
#include "cpp_source/mnist.h"
#include "cuda_source/convolution.cuh"

#ifdef FIX_MODE
int main() {
	try {
		MNIST mnist("E:\\data_set\\mnist", 64);
		NN_Manager nn;

		Layer_t x_input = Input({1, 28, 28 }, -1, "input");
		
		Layer_t x = NN_Creater(NN_Conv2D(32, { 5, 5 }, { 1, 1 }, Pad::VALID, "conv2d_1"))(x_input);
		x = NN_Creater(NN_ReLU("ReLU_1"))(x);
		x = NN_Creater(NN_Maxpool2D({ 2, 2 }, { 2, 2 }, "maxpool2d_1"))(x);				// [none, 12, 12, 32]
		x = NN_Creater(NN_Conv2D(64, { 5, 5 }, { 1, 1 }, Pad::VALID, "conv2d_2"))(x);
		x = NN_Creater(NN_ReLU("ReLU_2"))(x);
		x = NN_Creater(NN_Maxpool2D({ 2, 2 }, { 2, 2 }, "maxpool2d_2"))(x);				// [none, 4, 4, 16]
		x = NN_Creater(NN_Flatten("Flatten"))(x);										// [none, 512]
		x = NN_Creater(NN_Dense(512, "Dense_1"))(x);
		x = NN_Creater(NN_ReLU("ReLU_3"))(x);
		x = NN_Creater(NN_Dense(256, "Dense_2"))(x);
		x = NN_Creater(NN_ReLU("ReLU_4"))(x);
		x = NN_Creater(NN_Dense(10, "Dense_3"))(x);
		x = NN_Creater(NN_ReLU("ReLU_5"))(x);

		Model model = Model(x_input, x, "model_1");
		model.summary();

		Tensor<nn_type> x_tensor({ 60000, 1, 28, 28 });

		for (uint i = 0; i < x_tensor._len; ++i) x_tensor._data[i] = mnist.train_x._data[i] / 255.f;
		
		clock_t start = clock();
		std::vector<Tensor<nn_type>> output = model.predict<nn_type>(x_tensor, 128, 60000 / 128);
		clock_t end = clock();

		printf("elapsed time: %ld ms.\n", end - start);
	
		/*
		for (uint i = 0; i < 128; ++i) {
			for (uint j = 0; j < 784; ++j) {
				sample._data[i * 784 + j] = ((float)mnist.train_x._data[i * 784 + j]) / 255.f;
			}
		}
		clock_t start = clock();
		std::vector<Tensor<nn_type>> output = model.predict({ sample });
		clock_t end = clock();

		printf("\n\nelapsed time: %ld ms.\n", end - start);
		*/
	}
	catch (const Exception& e) {
		cudaDeviceReset();
		e.Put();
	}

	cudaDeviceReset();

	return 0;
}

#endif

#ifndef FIX_MODE
int main() {
	try {
		NN_Manager nn;

		Layer_t x_input = Input({ 32 }, -1, "input");

		Layer_t x = NN_Creater(NN_Dens(16, "Dense_1"))(x_input);
		x = NN_Creater(NN_ReLU("ReLU_1"))(x);

		Model model = Model(x_input, x, "model_1");
		model.summary();

		Tensor<nn_type> x_tensor({ 16, 32 });

		for (uint i = 0; i < x_tensor._len; ++i) x_tensor._data[i] = (float)i * 0.1f;

		std::vector<Tensor<nn_type>> output = model.predict<nn_type>(x_tensor, 16, 1);

		std::cout << output[0];


		/*
		for (uint i = 0; i < 128; ++i) {
			for (uint j = 0; j < 784; ++j) {
				sample._data[i * 784 + j] = ((float)mnist.train_x._data[i * 784 + j]) / 255.f;
			}
		}
		clock_t start = clock();
		std::vector<Tensor<nn_type>> output = model.predict({ sample });
		clock_t end = clock();

		printf("\n\nelapsed time: %ld ms.\n", end - start);
		*/
	}
	catch (const Exception& e) {
		e.Put();
	}

	return 0;
}
#endif