#include <iostream>
#include <random>
#include <time.h>
#include <vld.h>

#include "cpp_source/nn_model.h"
#include "cpp_source/nn_layer.h"



int main() {
	
	NN_Manager nn;
	
	Layer_t x_input_1 = Input({ 28, 28, 1 }, 32, "input_1");
	Layer_t x_input_2 = Input({ 28, 28, 1 }, 32, "input_2");
	
	Layer_t x = NN_Creater(NN_Test("test_1_1"))(x_input_1);
	x = NN_Creater(NN_Test("test_1_2"))(x);
	x = NN_Creater(NN_Test("test_1_3"))(x);
	Layer_t branch_1 = NN_Creater(NN_Test("test_1_4"))(x);

	x = NN_Creater(NN_Test("test_2_1"))(x_input_2);
	x = NN_Creater(NN_Test("test_2_2"))(x);
	Layer_t branch_2 = NN_Creater(NN_Test("test_2_3"))(x);
	
	x = NN_Creater(NN_Test("concat"))({ branch_1, branch_2 });
	
	x = NN_Creater(NN_Test("test_3_1"))(x);
	x = NN_Creater(NN_Test("test_3_2"))(x);
	Layer_t y_output = NN_Creater(NN_Test("test_3_3"))(x);

	Model model({ x_input_1, x_input_2 }, { y_output }, "model_1");
	
	x_input_1 = Input({ 28, 28, 1 }, 32, "input_1_1");
	x_input_2 = Input({ 28, 28, 1 }, 32, "input_1_2");

	x = NN_Creater(NN_Test("test_4_1"))(x_input_1);
	x = NN_Creater(NN_Test("test_4_2"))(x);
	Layer_t feature_1 = NN_Creater(NN_Test("test_4_3"))(x);

	x = NN_Creater(NN_Test("test_5_1"))(x_input_2);
	x = NN_Creater(NN_Test("test_5_2"))(x);
	Layer_t feature_2 = NN_Creater(NN_Test("test_5_3"))(x);

	x = model({ feature_1, feature_2 });
	x = NN_Creater(NN_Test("test_6_1"))(x);
	x = NN_Creater(NN_Test("test_6_2"))(x);
	Layer_t y_output_2 = NN_Creater(NN_Test("test_6_3"))(x);

	Model model_2({ x_input_1, x_input_2 }, { y_output_2 }, "model_2");

	model.summary();
	model_2.summary();

	cout << "===========================================" << endl;

	NN_Tensor tensor_1;
	NN_Tensor tensor_2;

	tensor_1.test_value = 1;
	tensor_2.test_value = 2;

	vector<NN_Tensor> result = model_2.predict({ tensor_1, tensor_2 });

	for (NN_Tensor p_result : result) cout << p_result.test_value << endl;

	return 0;
}
