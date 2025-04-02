#include "../cpu_core/src/header/host_tensor.h"

int main() {
	HostTensor<int> tensor(NN_Shape({ 3, 3, 3 }));

	return 0;
}