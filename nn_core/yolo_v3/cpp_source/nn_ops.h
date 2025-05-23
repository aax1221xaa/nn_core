#pragma once
#include "nn_base.h"


/**********************************************/
/*                                            */
/*                    NN_Ops                  */
/*                                            */
/**********************************************/

class NN_Ops : public NN_Layer {
protected:
	nn_type _scalar;

public:
	NN_Ops(const std::string& layer_name);
	NN_Ops(nn_type scalar, const std::string& layer_name);
	NN_Ops(const NN_Ops& p);
	NN_Ops(NN_Ops&& p);

	const NN_Ops& operator=(const NN_Ops& p);
	const NN_Ops& operator=(NN_Ops&& p);

	void get_output_shape(const NN_List<NN_Shape>& input_shape, NN_List<NN_Shape>& output_shape);
	void set_output(const NN_List<NN_Shape>& output_shape, NN_List<GpuTensor<nn_type>>& input, NN_List<GpuTensor<nn_type>>& output);
};


/**********************************************/
/*                                            */
/*                    NN_Add                  */
/*                                            */
/**********************************************/

class NN_Add : public NN_Ops {
public:
	NN_Add(const std::string& layer_name = "");
	NN_Add(nn_type scalar, const std::string& layer_name = "");
	NN_Add(const NN_Add& p);
	NN_Add(NN_Add&& p);

	void run(NN_Stream& st, const NN_List<GpuTensor<nn_type>>& input, NN_List<GpuTensor<nn_type>>& output);
};


/**********************************************/
/*                                            */
/*                    NN_Sub                  */
/*                                            */
/**********************************************/

class NN_Sub : public NN_Ops {
public:
	NN_Sub(const std::string& layer_name = "");
	NN_Sub(nn_type scalar, const std::string& layer_name = "");
	NN_Sub(const NN_Sub& p);
	NN_Sub(NN_Sub&& p);

	void run(NN_Stream& st, const NN_List<GpuTensor<nn_type>>& input, NN_List<GpuTensor<nn_type>>& output);
};


/**********************************************/
/*                                            */
/*                    NN_Mul                  */
/*                                            */
/**********************************************/

class NN_Mul : public NN_Ops {
public:
	NN_Mul(const std::string& layer_name = "");
	NN_Mul(nn_type scalar, const std::string& layer_name = "");
	NN_Mul(const NN_Mul& p);
	NN_Mul(NN_Mul&& p);

	void run(NN_Stream& st, const NN_List<GpuTensor<nn_type>>& input, NN_List<GpuTensor<nn_type>>& output);
};


/**********************************************/
/*                                            */
/*                    NN_Div                  */
/*                                            */
/**********************************************/

class NN_Div : public NN_Ops {
public:
	NN_Div(const std::string& layer_name = "");
	NN_Div(nn_type scalar, const std::string& layer_name = "");
	NN_Div(const NN_Div& p);
	NN_Div(NN_Div&& p);

	void run(NN_Stream& st, const NN_List<GpuTensor<nn_type>>& input, NN_List<GpuTensor<nn_type>>& output);
};