/* Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
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
