#include "../header/nn_common.h"
#include "../header/Exception.h"
#include <random>


std::vector<int> random_choice(int min, int max, int amounts, bool replace) {
	if ((max - min) < amounts) {
		ErrorExcept(
			"[random_choice] Invalid amounts."
		);
	}

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<int> dist(min, max - 1);

	std::vector<int> indice(amounts);

	if (replace) {
		for (int i = 0; i < amounts; ++i) {
			indice[i] = dist(gen);
		}
	}
	else {
		std::vector<bool> mask(labs(max - min), false);

		for (int i = 0; i < amounts; ++i) {
			while (true) {
				int num = dist(gen);

				if (!mask[num - min]) {
					mask[num - min] = true;
					indice[i] = num;

					break;
				}
			}
		}
	}

	return indice;
}