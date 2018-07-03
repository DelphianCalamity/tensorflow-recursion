#ifndef TENSORFLOW_KERNELS_FUNCTION_CONTROL_OPS_H_
#define TENSORFLOW_KERNELS_FUNCTION_CONTROL_OPS_H_

#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

// A call op has one input and one output. It creates or finds
// the child frame that is uniquely identified by the frame_name,
// and makes its input available to the child frame.
class CallOp : public OpKernel {
public:
    explicit CallOp(OpKernelConstruction* context) : OpKernel(context) {}
    void Compute(OpKernelContext* context) override;
    bool IsExpensive() override { return false; }
    ~CallOp() override {}

    TF_DISALLOW_COPY_AND_ASSIGN(CallOp);
};

// A Return op has one input and one output. It exits the current
// frame to its parent frame, and makes its input available to the
// parent frame only if it receives a tensor with a specific tag.
class ReturnOp : public OpKernel {
public:
    explicit ReturnOp(OpKernelConstruction* context) : OpKernel(context) {}
    void Compute(OpKernelContext* context) override;
    bool IsExpensive() override { return false; }
    ~ReturnOp() override {}

    TF_DISALLOW_COPY_AND_ASSIGN(ReturnOp);
};
}  // namespace tensorflow

#endif
