#include "Dimension.h"



/*********************************************/
/*                                           */
/*                     Dim                   */
/*                                           */
/*********************************************/

Dim::Dim() {

}

Dim::Dim(const initializer_list<int>& arr) :
	dim(arr)
{
}


const int& Dim::operator[](int axis) {
	if (axis < 0) {
		return dim[dim.size() + axis];
	}
	
	return dim[axis];
}

const bool Dim::operator==(const Dim& pDim) {
	size_t pSize = pDim.size();
	size_t cSize = size();
	bool match_flag = true;

	if (pSize != cSize) match_flag = false;

	for (size_t i = 0; i < cSize; ++i) {
		if (pDim.dim[i] != dim[i]) match_flag = false;
	}

	return match_flag;
}

void Dim::set(const initializer_list<int>& dim_) {
	dim = dim_;
}

const unsigned int Dim::size() const {
	return (unsigned int)dim.size();
}

void Dim::clear() {
	dim.clear();
}