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
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

// --------------------------------------------------------------------------
REGISTER_OP("Call")
    .Input("data: T")
    .Output("output: T")
    .Attr("T: type")
    .Attr("frame_name: string")
    .Attr("is_constant: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->UnknownShape());

      // Handle resource shape / dtype, if present.
      auto* handle_data = c->input_handle_shapes_and_types(0);
      if (handle_data != nullptr) {
        c->set_output_handle_shapes_and_types(0, *handle_data);
      } else {
        // Otherwise, propagate shape if output is a constant.
        bool is_constant;
        TF_RETURN_IF_ERROR(c->GetAttr("is_constant", &is_constant));
        if (is_constant) {
         c->set_output(0, c->input(0));
        }
      }
      return Status::OK();
    })
    .Doc(R"Doc(
Creates (or finds) a child frame, and makes `data` available to the child frame.

This op is used together with `Return` to create recursive calls in the graph.
The unique `frame_name` is used by the `Executor` to identify frames.

data: The tensor to be made available to the child frame.
frame_name: The name of the child frame.
output: The same tensor as `data`.

Returns tensors with the same shapes and contents as the input
tensors.
    )Doc");

REGISTER_OP("RefCall")
    .Input("data: Ref(T)")
    .Output("output: Ref(T)")
    .Attr("T: type")
    .Attr("frame_name: string")
    .Attr("is_constant: bool = false")
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(R"Doc(
Creates (or finds) a child frame, and makes `data` available to the child frame.

This op is used together with `Return` to create recursive calls in the graph.
The unique `frame_name` is used by the `Executor` to identify frames.

data: The tensor to be made available to the child frame.
frame_name: The name of the child frame.
output: The same tensor as `data`.

Returns tensors with the same shapes and contents as the input
tensors.
    )Doc");

// --------------------------------------------------------------------------
REGISTER_OP("Return")
.Input("data: T")
.Output("output: T")
.Attr("T: type")
.Attr("frame_name: string")
.SetShapeFn(shape_inference::UnchangedShape)
.Doc(R"Doc(
Exits the current frame to its parent frame.
Exit makes its input `data` available to the parent frame.
data: The list of tensors to be made available to the parent frame.
output: The same list of tensors as `data`.
    )Doc");

REGISTER_OP("RefReturn")
.Input("data: Ref(T)")
.Output("output: Ref(T)")
.Attr("T: type")
.Attr("frame_name: string")
.SetShapeFn(shape_inference::UnchangedShape)
.Doc(R"Doc(
Exits the current frame to its parent frame.
Exit makes its input `data` available to the parent frame.
data: The tensors to be made available to the parent frame.
output: The same tensors as `data`.
    )Doc");

}  // namespace tensorflow
