#include <vld.h>

#include "cpp_source/nn_model.h"
#include "cpp_source/nn_layer.h"


int main() {
	try {
		NN_Manager nn;

		Layer_t x_input = nn.input({ 1, 28, 28 }, -1, "input");

		Layer_t x = nn(NN_Conv2D(32, { 5, 5 }, { 1, 1 }, Pad::VALID, "conv_1"))(x_input);
		x = nn(NN_ReLU("relu_1"))(x);
		x = nn(NN_Maxpool2D({ 2, 2 }, { 2, 2 }, Pad::VALID, "maxpool_1"))(x);
		
		x = nn(NN_Conv2D(64, { 5, 5 }, { 1, 1 }, Pad::VALID, "conv_2"))(x);
		x = nn(NN_ReLU("relu_2"))(x);
		x = nn(NN_Maxpool2D({ 2, 2 }, { 2, 2 }, Pad::VALID, "maxpool_2"))(x);

		x = nn(NN_Flat("flat"))(x);

		x = nn(NN_Dense(512, "dense_1"))(x);
		x = nn(NN_ReLU("relu_3"))(x);
		x = nn(NN_Dense(128, "dense_2"))(x);
		x = nn(NN_ReLU("relu_4"))(x);
		x = nn(NN_Dense(10, "dense_2"))(x);
		x = nn(NN_ReLU("relu_5"))(x);

		Model model(nn, x_input, x, "model_1");

		model.summary();
	}
	catch (Exception& e) {
		e.put();
	}

	cudaDeviceReset();

	return 0;
}