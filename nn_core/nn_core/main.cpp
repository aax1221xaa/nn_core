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

		Layer_t x_input = nn.input({ 1, 28, 28 }, -1, "input");
		Layer_t x2_input = nn.input({ 1, 28, 28 }, -1, "input2");
		Layer_t x3_input = nn.input({ 1, 28, 28 }, -1, "input3");

		Layer_t x = nn(NN_Dense(1024, "Dense_1_1"))(x_input);
		x = nn(NN_Dense(512, "Dense_1_2"))(x);
		x = nn(NN_Dense(256, "Dense_1_3"))(x);

		Layer_t x2 = nn(NN_Dense(1024, "Dense_2_1"))(x2_input);
		x2 = nn(NN_Dense(512, "Dense_2_2"))(x2);
		x2 = nn(NN_Dense(256, "Dense_2_3"))(x2);

		Layer_t x3 = nn(NN_Dense(1024, "Dense_3_1"))(x3_input);
		x3 = nn(NN_Dense(512, "Dense_3_2"))(x3);
		x3 = nn(NN_Dense(256, "Dense_3_3"))(x3);

		Layer_t y = nn(NN_Concat("Concat_1"))({ x, x2 });
		y = nn(NN_Dense(64, "Dense_7_2"))(y);

		Layer_t y2 = nn(NN_Concat("Concat_2"))({ x3, y });

		y = nn(NN_Dense(32, "Dense_4_1"))(y);
		y2 = nn(NN_Dense(32, "Dense_4_2"))(y2);

		Model model_1(nn, { x_input, x2_input }, y, "Model_1");
		Model model_2(nn, { x_input, x2_input, x3_input }, y2, "Model_2");

		model_1.summary();
		model_2.summary();
	}
	catch (Exception& e) {
		e.Put();
	}

	return 0;
}