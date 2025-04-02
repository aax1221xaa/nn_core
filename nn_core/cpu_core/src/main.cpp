#include "header/nn_core.h"
#include <exception>

#ifdef _DEBUG
#include "vld.h"
#endif
 


std::vector<NN_OpLinker*> distance(NN_Lambda& nn, const std::vector<NN_OpLinker*>& a){
	return { nn(NN_Div())(a[0], 255.f) };
}


int main() {
	try {
		NN_Manager nn;

		Layer_t x_input = nn.input({ 28, 28, 1 }, -1, "input");
		Layer_t x = nn(NN_Lambda(nn, distance, "lambda"))(x_input);
		x = nn(NN_Conv2D(32, { 5, 5 }, { 1, 1 }, "valid", "conv2d_1"))(x);		// {24, 24, 32}
		x = nn(NN_ReLU("relu_1"))(x);
		x = nn(NN_Maxpool2D({ 2, 2 }, { 2, 2 }, "valid", "maxpool_1"))(x);		// {12, 12, 32}
		x = nn(NN_Conv2D(64, { 5, 5 }, { 1, 1 }, "valid", "conv2d_2"))(x);		// {8, 8, 64}
		x = nn(NN_ReLU("relu_2"))(x);
		x = nn(NN_Maxpool2D({ 2, 2 }, { 2, 2 }, "valid", "maxpool_2"))(x);		// {4, 4, 64}
		x = nn(NN_Flatten("flatten"))(x);
		x = nn(NN_Dense(512, "dense_1"))(x);
		x = nn(NN_ReLU("relu_3"))(x);
		x = nn(NN_Dense(10, "dense_2"))(x);
		x = nn(NN_Softmax("softmax"))(x);

		Model model(nn, x_input, x, "model_1");

		model.summary();
	}
	catch (NN_Exception& e) {
		e.put();
		cudaDeviceReset();

		return -1;
	}

	cudaDeviceReset();

	return 0;
}