#pragma once
#include <vector>


/*********************************************/
/*                                           */
/*                     Dim                   */
/*                                           */
/*********************************************/

class Dim {
public:
	std::vector<int> dim;

	Dim();
	Dim(const std::vector<int>& dim_);
	
	int& operator[](int axis);
	bool operator==(Dim& pDim);

	void Set(const std::vector<int>& dim_);
	std::vector<int>& Get();
	const size_t GetSize();
};