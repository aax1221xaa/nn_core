#include <iostream>
#include <random>
#include <time.h>
#include <vld.h>

#include "cpp_source/nn_layer.h"

typedef NN_Vec<NN_Coupler<NN_Link>> NN;

int main() {
	NN_Shape shape({ 1, 2, 3, 4 });

	for (int& n : shape) {
		printf("%d ", n);
	}
	printf("\n");

	return 0;
}