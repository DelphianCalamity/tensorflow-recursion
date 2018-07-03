#include "tensorflow/core/kernels/function_control_ops.h"

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {

void CallOp::Compute(OpKernelContext* context) {
  if (IsRefType(context->input_dtype(0))) {
    context->forward_ref_input_to_ref_output(0, 0);
  } else {
    context->set_output(0, context->input(0));
  }
}

REGISTER_KERNEL_BUILDER(Name("Call").Device(DEVICE_CPU), CallOp);
REGISTER_KERNEL_BUILDER(Name("RefCall").Device(DEVICE_CPU), CallOp);

#define REGISTER_GPU_KERNEL(type) \
  REGISTER_KERNEL_BUILDER(        \
      Name("Call").Device(DEVICE_GPU).TypeConstraint<type>("T"), CallOp)
#define REGISTER_GPU_REF_KERNEL(type) \
  REGISTER_KERNEL_BUILDER(            \
      Name("RefCall").Device(DEVICE_GPU).TypeConstraint<type>("T"), CallOp)

TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_GPU_KERNEL);
TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_GPU_REF_KERNEL);
REGISTER_GPU_KERNEL(bool);
REGISTER_GPU_REF_KERNEL(bool);

#undef REGISTER_GPU_KERNEL
#undef REGISTER_GPU_REF_KERNEL

#ifdef TENSORFLOW_USE_SYCL
#define REGISTER_SYCL_KERNEL(type)  \
  REGISTER_KERNEL_BUILDER(          \
      Name("Call").Device(DEVICE_SYCL).TypeConstraint<type>("T"), CallOp)
REGISTER_SYCL_KERNEL(bool);
TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_SYCL_KERNEL);

#define REGISTER_SYCL_REF_KERNEL(type)  \
  REGISTER_KERNEL_BUILDER(              \
      Name("RefCall").Device(DEVICE_SYCL).TypeConstraint<type>("T"), CallOp)
REGISTER_SYCL_REF_KERNEL(bool);
TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_SYCL_REF_KERNEL);

#undef REGISTER_SYCL_KERNEL
#undef REGISTER_SYCL_REF_KERNEL
#define REGISTER_SYCL_HOST_KERNEL(type)                   \
  REGISTER_KERNEL_BUILDER(Name("Call")                   \
                              .Device(DEVICE_SYCL)        \
                              .HostMemory("data")         \
                              .HostMemory("output")       \
                              .TypeConstraint<type>("T"), \
                          CallOp)

#define REGISTER_SYCL_HOST_REF_KERNEL(type)               \
  REGISTER_KERNEL_BUILDER(Name("RefCall")                \
                              .Device(DEVICE_SYCL)        \
                              .HostMemory("data")         \
                              .HostMemory("output")       \
                              .TypeConstraint<type>("T"), \
                          CallOp)

REGISTER_SYCL_HOST_KERNEL(int32);
REGISTER_SYCL_HOST_REF_KERNEL(int32);
REGISTER_SYCL_HOST_KERNEL(string);
REGISTER_SYCL_HOST_REF_KERNEL(string);
REGISTER_SYCL_HOST_KERNEL(ResourceHandle);

#undef REGISTER_SYCL_HOST_KERNEL
#undef REGISTER_SYCL_HOST_REF_KERNEL
#endif // TENSORFLOW_USE_SYCL

#define REGISTER_GPU_HOST_KERNEL(type)                    \
  REGISTER_KERNEL_BUILDER(Name("Call")                   \
                              .Device(DEVICE_GPU)         \
                              .HostMemory("data")         \
                              .HostMemory("output")       \
                              .TypeConstraint<type>("T"), \
                          CallOp)

#define REGISTER_GPU_HOST_REF_KERNEL(type)                \
  REGISTER_KERNEL_BUILDER(Name("RefCall")                \
                              .Device(DEVICE_GPU)         \
                              .HostMemory("data")         \
                              .HostMemory("output")       \
                              .TypeConstraint<type>("T"), \
                          CallOp)

REGISTER_GPU_HOST_KERNEL(int32);
REGISTER_GPU_HOST_REF_KERNEL(int32);
REGISTER_GPU_HOST_KERNEL(string);
REGISTER_GPU_HOST_REF_KERNEL(string);
REGISTER_GPU_HOST_KERNEL(ResourceHandle);

#undef REGISTER_GPU_HOST_KERNEL
#undef REGISTER_GPU_HOST_REF_KERNEL

void ReturnOp::Compute(OpKernelContext* context) {
  if (IsRefType(context->input_dtype(0))) {
    context->forward_ref_input_to_ref_output(0, 0);
  } else {
    context->set_output(0, context->input(0));
  }
}

REGISTER_KERNEL_BUILDER(Name("Return").Device(DEVICE_CPU), ReturnOp);
REGISTER_KERNEL_BUILDER(Name("RefReturn").Device(DEVICE_CPU), ReturnOp);

#define REGISTER_GPU_KERNEL(type) \
  REGISTER_KERNEL_BUILDER(        \
      Name("Return").Device(DEVICE_GPU).TypeConstraint<type>("T"), ReturnOp);
#define REGISTER_GPU_REF_KERNEL(type) \
  REGISTER_KERNEL_BUILDER(            \
      Name("RefReturn").Device(DEVICE_GPU).TypeConstraint<type>("T"), ReturnOp);

TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_GPU_KERNEL);
TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_GPU_REF_KERNEL);
REGISTER_GPU_KERNEL(bool);
REGISTER_GPU_REF_KERNEL(bool);

#undef REGISTER_GPU_KERNEL
#undef REGISTER_GPU_REF_KERNEL

#ifdef TENSORFLOW_USE_SYCL
    #define REGISTER_SYCL_KERNEL(type)                                           \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("Return").Device(DEVICE_SYCL).TypeConstraint<type>("T"), ReturnOp);   \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("RefReturn").Device(DEVICE_SYCL).TypeConstraint<type>("T"), ReturnOp);
REGISTER_SYCL_KERNEL(bool);
TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_SYCL_KERNEL);

#undef REGISTER_SYCL_KERNEL
#undef REGISTER_SYCL_REF_KERNEL

#define REGISTER_SYCL_HOST_KERNEL(type)                   \
  REGISTER_KERNEL_BUILDER(Name("Return")                    \
                              .Device(DEVICE_SYCL)        \
                              .HostMemory("data")         \
                              .HostMemory("output")       \
                              .TypeConstraint<type>("T"), \
                          ReturnOp);                        \
  REGISTER_KERNEL_BUILDER(Name("RefReturn")                 \
                              .Device(DEVICE_SYCL)        \
                              .HostMemory("data")         \
                              .HostMemory("output")       \
                              .TypeConstraint<type>("T"), \
                          ReturnOp)

REGISTER_SYCL_HOST_KERNEL(int32);
REGISTER_SYCL_HOST_KERNEL(string);
#undef REGISTER_SYCL_HOST_KERNEL
#endif // TENSORFLOW_USE_SYCL

#define REGISTER_GPU_HOST_KERNEL(type)                    \
  REGISTER_KERNEL_BUILDER(Name("Return")                    \
                              .Device(DEVICE_GPU)         \
                              .HostMemory("data")         \
                              .HostMemory("output")       \
                              .TypeConstraint<type>("T"), \
                          ReturnOp);                        \
  REGISTER_KERNEL_BUILDER(Name("RefReturn")                 \
                              .Device(DEVICE_GPU)         \
                              .HostMemory("data")         \
                              .HostMemory("output")       \
                              .TypeConstraint<type>("T"), \
                          ReturnOp)

REGISTER_GPU_HOST_KERNEL(int32);
REGISTER_GPU_HOST_KERNEL(string);

#undef REGISTER_GPU_HOST_KERNEL

}  // namespace tensorflow
