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

#ifndef TENSORFLOW_GRAPPLER_OPTIMIZERS_FUNCTION_TRANSFORMATION_H_
#define TENSORFLOW_GRAPPLER_OPTIMIZERS_FUNCTION_TRANSFORMATION_H_

#include "tensorflow/core/grappler/optimizers/graph_optimizer.h"
#include "tensorflow/core/grappler/grappler_item.h"

namespace tensorflow {
namespace grappler {

typedef std::unordered_map<string, NodeDef*> ArgMergeMap;

typedef struct {
  ArgMergeMap argMergeMap;
  gtl::ArraySlice<string> fetch;
} FuncInfo;

// Replace function calling nodes with pairs of new 'Call/Return' operators
class FunctionTransformation : public GraphOptimizer {
public:
    FunctionTransformation() {}
    ~FunctionTransformation() override {}

    string name() const override { return "function_transformation"; };

    Status Optimize(Cluster* cluster, const GrapplerItem& item,
                    GraphDef* optimized_graph) override;

    void Feedback(Cluster* cluster, const GrapplerItem& item,
                  const GraphDef& optimized_graph, double result) override;
};

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_GRAPPLER_OPTIMIZERS_FUNCTION_TRANSFORMATION_H_
