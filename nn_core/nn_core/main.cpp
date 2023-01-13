#include <iostream>
#include <random>
#include <time.h>
#include <vld.h>

#include "cpp_source/nn_model.h"
#include "cpp_source/nn_layer.h"



int main() {
	NN_Manager nn;

	NN x_input_1 = Input({ 28, 28, 1 }, 32, "input_1");
	NN x_input_2 = Input({ 28, 28, 1 }, 32, "input_2");

	NN x = Test("test_1_1")(x_input_1);
	x = Test("test_1_2")(x);
	x = Test("test_1_3")(x);
	NN feature_1 = Test("test_1_4")(x);

	x = Test("test_2_1")(x_input_2);
	x = Test("test_2_2")(x);
	x = Test("test_2_3")(x);
	NN feature_2 = Test("test_2_4")(x);

	NN y_output = Test("test_3_1")({ feature_1, feature_2 });

	NN_Model model = Model(x_input_1, y_output, "model_1");

	for (NN_Link* p : NN_Manager::reg_links) {
		printf("%s\n", p->op_layer->layer_name.c_str());
	}

	return 0;
}