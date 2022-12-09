#include <iostream>
#include <random>
#include <time.h>
#include <vld.h>

#include "cpp_source/nn_layer.h"
#include "cpp_source/nn_manager.h"

class Test {
public:
	int i;
	int j;
};


int main() {
	NN_Ptr<Test> test;

	test->i = 0;

	return 0;
}