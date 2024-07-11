Introduction
============
This library source has developed a deep learning framework for c++ users.

It's not complete yet, but it was developed based on keras framework.

This library's features are simplicity and fast speed.

If you use this library and want develop, you must be installed at least cuda v11 and have a intel tbb library and visula leak detector library. 

Structure of classes
===================
![class structure](https://github.com/aax1221xaa/nn_core/assets/135483148/3b122b34-f480-4c24-b993-5ea6ad689412)

* NN_Check

  This class is to notify the host of error during operating framework.

* NN_Exception

  This exception class throw error message to user.

* Tensor

  This class manages data in multidimensional array in host.

  Can perform arithmetic operations, slice, etc.

* GpuTensor

  This class manage data in multidimensional array in GPU device.
