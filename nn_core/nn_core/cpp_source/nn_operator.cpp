#include "nn_operator.h"
#include "../cuda_source/cuda_misc.cuh"


/**********************************************/
/*                                            */
/*					 NN_Div                   */
/*                                            */
/**********************************************/

OpDiv::OpDiv() :
	NN_Operator("div"),
	_val(0.f)
{

}

void OpDiv::run(NN_Stream& st, const NN_List<GpuTensor<nn_type>>& input, NN_List<GpuTensor<nn_type>>& output) {
	GpuTensor<nn_type>& m_output = output[0].val();

	switch (_status)
	{
	case 1:
	{
		const GpuTensor<nn_type>& a_input = input[0].val();

		div_tensor(a_input, _val, m_output);
	}
	break;
	case 2:
	{
		const GpuTensor<nn_type>& a_input = input[0].val();

		inv_tensor(a_input, m_output);
		mul_tensor(m_output, _val, m_output);
	}
	break;
	case 3:
	{
		const GpuTensor<nn_type>& a_input = input[0].val();
		const GpuTensor<nn_type>& b_input = input[1].val();
		
		div_tensor(a_input, b_input, m_output);
	}
	break;
	default:
		break;
	}
}

void OpDiv::set_const_value(nn_type val, int status) {
	_val = val;
	_status = status;
}

nn_type OpDiv::get_val() {
	return _val;
}