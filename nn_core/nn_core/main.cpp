#include <iostream>
#include <random>
#include <time.h>
#include <vld.h>

#include "cpp_source/nn_layer.h"


int main() {
	NN_Manager nn;

	NN_Link* x1 = Input({ 1, 1, 1 }, 1, "aaaa");
	NN_Link* x2 = Input({ 1, 1, 1 }, 1, "bbbb");
	
	NN_Link* conv = Conv2D()({ x1, x2 });

	NN_Link* y1 = Conv2D()(conv);
	NN_Link* y2 = Conv2D()(conv);

	auto model = Model({ x1, x2 }, { y1, y2 });

	NN_Link* _x1 = Input({ 1, 1, 1, 1 }, 1, "aaaaaa");
	NN_Link* _x2 = Input({ 1, 1, 1, 1 }, 1, "bbbbbb");

	vector<NN_Link*> y = model({ _x1, _x2 });
	

	return 0;
}