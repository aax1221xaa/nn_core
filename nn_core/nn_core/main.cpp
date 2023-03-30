#include <iostream>
#include <random>
#include <time.h>
#include <vld.h>

#include "cpp_source/nn_model.h"
#include "cpp_source/nn_layer.h"



/*
int main() {
	NN_Manager nn;

	NN x_input_1 = Input({ 28, 28, 1 }, 32, "input_1");
	NN x_input_2 = Input({ 28, 28, 1 }, 32, "input_2");

	NN x = Test("test_1_1")(x_input_1);
	x = Test("test_1_2")(x);
	x = Test("test_1_3")(x);
	NN branch_1 = Test("test_1_4")(x);

	x = Test("test_2_1")(x_input_2);
	x = Test("test_2_2")(x);
	NN branch_2 = Test("test_2_3")(x);

	x = Test("concat")({ branch_1, branch_2 });
	x = Test("test_3_1")(x);
	x = Test("test_3_2")(x);
	NN y_output = Test("test_3_3")(x);

	NN_Model& model = Model({ x_input_1, x_input_2 }, y_output, "model_1");

	x_input_1 = Input({ 28, 28, 1 }, 32, "input_1_1");
	x_input_2 = Input({ 28, 28, 1 }, 32, "input_1_2");

	x = Test("test_4_1")(x_input_1);
	x = Test("test_4_2")(x);
	NN feature_1 = Test("test_4_3")(x);

	x = Test("test_5_1")(x_input_2);
	x = Test("test_5_2")(x);
	NN feature_2 = Test("test_5_3")(x);

	x = model({ feature_1, feature_2 });
	x = Test("test_6_1")(x);
	x = Test("test_6_2")(x);
	NN y_output_2 = Test("test_6_3")(x);

	NN_Model model_2 = Model({ x_input_1, x_input_2 }, y_output_2, "model_2");

	model_2.summary();

	return 0;
}
*/

int main() {
	List<int> list({ 1, 2, {3, 4} });


	return 0;
}