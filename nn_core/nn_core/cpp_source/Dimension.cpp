#include "Dimension.h"



/*********************************************/
/*                                           */
/*                     Dim                   */
/*                                           */
/*********************************************/

Dim::Dim() {

}

Dim::Dim(const std::vector<int>& dim_) :
	dim(dim_)
{
}

int& Dim::operator[](int axis) {
	if (axis < 0) {
		return dim[dim.max_size() - axis - 1];
	}
	
	return dim[axis];
}

bool Dim::operator==(Dim& pDim) {
	size_t pSize = pDim.GetSize();
	size_t size = GetSize();

	if (pSize != size) return false;

	for (size_t i = 0; i < size; ++i) {
		if (pDim.dim[i] != dim[i]) return false;
	}

	return true;
}

void Dim::Set(const std::vector<int>& dim_) {
	dim = dim_;
}

std::vector<int>& Dim::Get() {
	return dim;
}

const size_t Dim::GetSize() {
	return dim.size();
}