#pragma once
#include <vector>
#include "CudaCheck.h"


#define BLOCK_4					4
#define BLOCK_8					8
#define BLOCK_16				16
#define BLOCK_32				32
#define BLOCK_1024				1024
#define CONST_ELEM_SIZE			(65536 / sizeof(uint))

#define EPSILON					1e-8

#define STREAMS					16

typedef const int cint;
typedef unsigned int uint;
typedef const unsigned int cuint;
typedef float nn_type;

dim3 get_grid_size(const dim3 block, unsigned int x = 1, unsigned int y = 1, unsigned int z = 1);

std::vector<int> random_choice(int min, int max, int amounts, bool replace = true);



/**********************************************/
/*                                            */
/*                  NN_Stream                 */
/*                                            */
/**********************************************/

class NN_Stream {
private:
	class Container {
	public:
		cudaStream_t* _st;
		int _n_ref;
		int _amounts;

		Container() : _st(NULL), _n_ref(0), _amounts(0) {}
	}*_ptr;

	void destroy();

public:
	class Iterator {
	public:
		cudaStream_t* m_st;
		int _n_id;

		Iterator(cudaStream_t* st, int index) : m_st(st), _n_id(index) {}
		Iterator(const typename NN_Stream::Iterator& p) : m_st(p.m_st), _n_id(p._n_id) {}

		bool operator!=(const typename NN_Stream::Iterator& p) const { return _n_id != p._n_id; }
		void operator++() { ++_n_id; }
		cudaStream_t& operator*() const { return m_st[_n_id]; }
	};

	NN_Stream(int amounts = STREAMS);
	NN_Stream(const NN_Stream& p);
	NN_Stream(NN_Stream&& p);
	~NN_Stream();

	NN_Stream& operator=(const NN_Stream& p);
	NN_Stream& operator=(NN_Stream&& p);

	cudaStream_t& operator[](int index);

	typename NN_Stream::Iterator begin() const { return NN_Stream::Iterator(_ptr->_st, 0); }
	typename NN_Stream::Iterator end() const { return NN_Stream::Iterator(_ptr->_st, _ptr->_amounts); }
	void clear();

	cudaStream_t* get_stream() const;
};