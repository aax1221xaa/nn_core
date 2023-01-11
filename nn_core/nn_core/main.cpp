#include <iostream>
#include <random>
#include <time.h>
#include <vld.h>

#include "cpp_source/nn_model.h"
#include "cpp_source/nn_layer.h"



int main() {
	NN_Manager nn;

	NN x_input = Input({ 28, 28, 1 }, 32, "input");
	NN x = Test("test_1")(x_input);
	x = Test("test_2")(x);
	x = Test("test_3")(x);
	NN y_output = Test("test_4")(x);

	NN_Model model = Model(x_input, y_output, "model_1");

	for (NN_Link* p : NN_Manager::reg_links) {
		printf("%s\n", p->op_layer->layer_name.c_str());
	}

	return 0;
}