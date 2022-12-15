#pragma once
#include <vector>


using namespace std;

/*********************************************/
/*                                           */
/*                     Dim                   */
/*                                           */
/*********************************************/

class Dim {
public:
	vector<int> dim;

	Dim();
	Dim(const initializer_list<int>& arr);
	
	int& operator[](int axis);
	const bool operator==(const Dim& pDim);

	void set(const initializer_list<int>& dim_);
	const unsigned int size() const;
	void clear();
};