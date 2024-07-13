Introduction
============
This library source has developed a deep learning framework for c++ users.

It's not complete yet, but it was developed based on keras framework.

This library's features are simplicity and fast speed.

If you use this library and want develop, you must be installed at least cuda v11 and have a intel tbb library and visula leak detector library. 

Structure of classes
===================
![class structure](https://github.com/aax1221xaa/nn_core/assets/135483148/3b122b34-f480-4c24-b993-5ea6ad689412)

Example
==========
```cpp
#include "cpp_source/nn_core.h"
#include "cpp_source/mnist.h"

int main(){
  NN_Manager nn;

  Layer_t x_input = nn.input(NN_Shape({1, 28, 28}), -1, "input", NULL);

  Layer_t x = nn(NN_Conv2D(32, {5, 5}, {1, 1}, Pad::VALID, "conv_1"))(x_input);
  x = nn(NN_ReLU("relu_1"))(x);
  x = nn(NN_Maxpool2D({ 2, 2 }, { 2, 2 }, Pad::VALID, "maxpool_1"))(x);
		
  x = nn(NN_Conv2D(64, { 5, 5 }, { 1, 1 }, Pad::VALID, "conv_2"))(x);
  x = nn(NN_ReLU("relu_2"))(x);
  x = nn(NN_Maxpool2D({ 2, 2 }, { 2, 2 }, Pad::VALID, "maxpool_2"))(x);

  x = nn(NN_Flatten("flat"))(x);

  x = nn(NN_Dense(256, "dense_1"))(x);
  x = nn(NN_Sigmoid("sigmoid_3"))(x);
  x = nn(NN_Dense(128, "dense_2"))(x);
  x = nn(NN_Sigmoid("sigmoid_4"))(x);
  x = nn(NN_Dense(10, "dense_3"))(x);
  x = nn(NN_Softmax("softmax"))(x);

  Model model(nn, x_input, x, "model_1");

  MNIST mnist("E:\\data_set\\mnist");
  std::vector<DataSet<uchar, uchar>> samples = mnist.get_samples();
  DataSet<uchar, uchar>& train = samples[0];
  DataSet<uchar, uchar>& test = samples[1];

  Tensor<uchar> mx = Tensor<uchar>::expand_dims(test._x[0], 1);

  std::vector<Tensor<uchar>> _x = { mx };

  model.summary();
  model.load_weights("E:\\data_set\\mnist\\mnist.h5");
		
  NN_List<Tensor<nn_type>> result = model.predict(_x, 16, 1);
	
  for (NN_List<Tensor<nn_type>>& m_result : result) {
    for (NN_List<Tensor<nn_type>>& p_result : m_result) {
      std::cout << std::endl << p_result.val();
    }
  }
return 0;
}
```
