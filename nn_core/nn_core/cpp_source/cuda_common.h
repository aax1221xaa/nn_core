#pragma once
#include "CudaCheck.h"
#include <vector>
#include <memory>


using namespace std;

typedef const int cint;
typedef unsigned int uint;
typedef const unsigned int cuint;

#define STR_MAX			1024

#define BLOCK_SIZE				32
#define SQR_BLOCK_SIZE			BLOCK_SIZE * BLOCK_SIZE
#define CONST_ELEM_SIZE			16384
#define CONST_MEM_SIZE			65536

#define EPSILON					1e-8


dim3 get_grid_size(const dim3 block, uint x = 1, uint y = 1, uint z = 1);