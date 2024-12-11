#include "cpp_source/nn_core.h"
#include "cpp_source/mnist.h"
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

		Layer_t x_input = nn.input({ 24, 24, 1 }, -1, "input");
		Layer_t x = nn(NN_Lambda(nn, distance, "lambda"))(x_input);
		x = nn(NN_Conv2D(32, { 3, 3 }, { 1, 1 }, "same", "conv2d"))(x);
		x = nn(NN_ReLU("relu"))(x);

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