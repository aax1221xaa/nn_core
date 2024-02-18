#include <iostream>
#include <random>
#include <time.h>
#include <tbb/tbb.h>
#include <vld.h>

#include "cpp_source/nn_model.h"
#include "cpp_source/nn_layer.h"
#include "cuda_source/convolution.cuh"

int main() {
	try {
		NN_Manager nn;

		Layer_t x_input = Input({1, 28, 28 }, -1, "input");
		Layer_t x2_input = Input({ 1, 28, 28 }, -1, "input2");
		Layer_t x3_input = Input({ 1, 28, 28 }, -1, "input3");

		Layer_t x = NN_Creater(NN_Dense(1024, "Dense_1_1"))(x_input);
		x = NN_Creater(NN_Dense(512, "Dense_1_2"))(x);
		x = NN_Creater(NN_Dense(256, "Dense_1_3"))(x);

		Layer_t x2 = NN_Creater(NN_Dense(1024, "Dense_2_1"))(x2_input);
		x2 = NN_Creater(NN_Dense(512, "Dense_2_2"))(x2);
		x2 = NN_Creater(NN_Dense(256, "Dense_2_3"))(x2);

		Layer_t x3 = NN_Creater(NN_Dense(1024, "Dense_1_1"))(x3_input);
		x3 = NN_Creater(NN_Dense(512, "Dense_3_2"))(x3);
		x3 = NN_Creater(NN_Dense(256, "Dense_3_3"))(x3);
		
		Layer_t y = NN_Creater(NN_Dense(128, "Dense_7_1"))({ x, x2 });
		y = NN_Creater(NN_Dense(64, "Dense_7_2"))(y);

		Model model_1({ x_input, x2_input, x3_input }, { y, x3 }, "model_1");
		model_1.summary();

		Layer_t x4_input = Input({ 1, 28, 28 }, -1, "input4");
		Layer_t x5_input = Input({ 1, 28, 28 }, -1, "input5");
		Layer_t x6_input = Input({ 1, 28, 28 }, -1, "input6");

		Layer_t x4 = NN_Creater(NN_Dense(1024, "Dense_4_1"))(x4_input);
		x4 = NN_Creater(NN_Dense(512, "Dense_4_2"))(x4);
		x4 = NN_Creater(NN_Dense(256, "Dense_4_3"))(x4);

		Layer_t x5 = NN_Creater(NN_Dense(1024, "Dense_5_1"))(x5_input);
		x5 = NN_Creater(NN_Dense(512, "Dense_5_2"))(x5);
		x5 = NN_Creater(NN_Dense(256, "Dense_5_3"))(x5);

		Layer_t x6 = NN_Creater(NN_Dense(1024, "Dense_1_1"))(x6_input);
		x6 = NN_Creater(NN_Dense(512, "Dense_6_2"))(x6);
		x6 = NN_Creater(NN_Dense(256, "Dense_6_3"))(x6);

		Layer_t y2 = model_1({ x4, x5, x6 });

		Layer_t y4 = NN_Creater(NN_Dense(128, "Dense_9_1"))(y2[1]);
		Layer_t y3 = NN_Creater(NN_Dense(128, "Dense_8_1"))(y2[0]);

		Model model_2({ x4_input, x5_input, x6_input }, { y3, y4 }, "model_2");
		model_2.summary();

	}
	catch (const Exception& e) {
		cudaDeviceReset();
		e.Put();
	}

	cudaDeviceReset();

	return 0;
}