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

#include "tensorflow/core/grappler/optimizers/function_transformation.h"
#include <set>
#include <iostream>
#include <unordered_map>
#include "tensorflow/core/util/event.pb.h"
#include "tensorflow/core/util/events_writer.h"

#include "tensorflow/core/graph/tensor_id.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/graph_def_util.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/grappler/utils/functions.h"

namespace tensorflow {
namespace grappler {
namespace {

typedef std::unordered_map<string, NodeDef*> ArgMergeMap;

typedef struct {
  ArgMergeMap argMergeMap;
  gtl::ArraySlice<string> fetch;
} FuncInfo;

// same with commit b691c0 (possibly)
class FunctionInliningContext {
  public:
    explicit FunctionInliningContext(const GrapplerItem& item)
            : library_(&item.graph.library()), functions_(InliningCandidates(item)) {}

    const FunctionDefLibrary& Library() const { return *library_; }

    bool HasInlinedFunctions() const { return !functions_.empty(); }

    // Find inlining candidate by name. Return nullptr if not found.
    const FunctionDef* FindInlinedFunction(const string& name) const {
      auto it = functions_.find(name);
      if (it != functions_.end()) {
        return it->second;
      } else {
        return nullptr;
      }
    }

  private:
    std::unordered_map<string, const FunctionDef*> InliningCandidates(const GrapplerItem& item) const {
      std::unordered_map<string, const FunctionDef*> functions;
      for (const FunctionDef& func : item.graph.library().function()) {
        // Don't inline functions marked as noinline
//                    if (func.attr().count("_noinline") != 0) {
//                      continue;
//                    }
        // Don't touch anything marked XLA to prevent XLA failures further down
        // the road.
        if (func.attr().count("_XlaCompile") > 0 &&
            func.attr().at("_XlaCompile").b()) {
          continue;
        }
        // Can't create IdentityN nodes with no input or output: skip these
        // functions for now.
        if (func.signature().input_arg_size() == 0 ||
            func.signature().output_arg_size() == 0) {
          continue;
        }
        functions[func.signature().name()] = &func;
      }
      return functions;
    }

    const FunctionDefLibrary* library_;
    std::unordered_map<string, const FunctionDef*> functions_;

    TF_DISALLOW_COPY_AND_ASSIGN(FunctionInliningContext);
};

// Copy input/output argument type to the type. Return error if argument
// type is not explicitly defined, and not specified in function attributes.
Status CopyArgType(const NodeDef& func_node,
                   const std::unordered_map<string, AttrValue>& func_attr,
                   const string& arg_kind, const OpDef::ArgDef& arg,
                   DataType* type) {
    if (arg.type() != DT_INVALID) {
      *type = arg.type();
    } else {
      auto it = func_attr.find(arg.type_attr());
      if (it == func_attr.end() || it->second.type() == DT_INVALID) {
        return errors::InvalidArgument(
                "Invalid ", arg_kind, " argument ", arg.name(), " for function ",
                func_node.op(), " instantiated by ", func_node.name());
      }
      *type = it->second.type();
    }
    return Status::OK();
}

// Copy input/output argument type to the type_list. Return error if argument
// type is not explicitly defined, and not specified in function attributes.
Status CopyArgType(const NodeDef& func_node,
                    const std::unordered_map<string, AttrValue>& func_attr,
                    const string& arg_kind, const OpDef::ArgDef& arg,
                    AttrValue::ListValue* type_list) {
    if (arg.type() != DT_INVALID) {
      type_list->add_type(arg.type());
    } else {
      auto it = func_attr.find(arg.type_attr());
      if (it == func_attr.end() || it->second.type() == DT_INVALID) {
        return errors::InvalidArgument(
                "Invalid ", arg_kind, " argument ", arg.name(), " for function ",
                func_node.op(), " instantiated by ", func_node.name());
      }
      type_list->add_type(it->second.type());
    }
    return Status::OK();
}

string ParseString(string input) {
    size_t pos = 0;
    std::string res = "";
    std::string delimiter = ":";

    if ((pos = input.find(delimiter)) != std::string::npos) {
      res = res + input.substr(0, pos);
      input.erase(0, pos + delimiter.length());
      res = res + "/Ret" + input;
    }
    else {
      res = input + "/Ret0";
    }
    return res;
}

Status GatherOutputs(const GrapplerItem& item, const FunctionInliningContext& ctx,
                     std::set<string> &foutputs) {
    for (const NodeDef& node : item.graph.node()) {
      const FunctionDef* func = ctx.FindInlinedFunction(node.op());
      if (func != nullptr) {      // If it's a function calling node
        for (int i = 0; i < func->signature().output_arg_size(); ++i) {
         // const OpDef::ArgDef &arg = func->signature().output_arg(i);
          foutputs.emplace(node.name());                   // Fac
          foutputs.emplace(strings::StrCat(node.name(), ":", i));      // Fac:i
          //foutputs.emplace(strings::StrCat(node.name(), ":", arg.name(), ":", i));      // Fac:outarg:i
        }
      }
    }
    return Status::OK();
}


Status CreateCycle(NodeDef& func_node, const FunctionDef& func, GraphDef* optimized_graph,
                   std::unordered_map<string, FuncInfo> &functions_in, int call_id, string device) {
    const std::unordered_map<string, AttrValue> func_attr(func_node.attr().begin(), func_node.attr().end());

    DataType type;
    ArgMergeMap& argmerge_map = functions_in[func_node.op()].argMergeMap;

    NodeDef *call;
    for (int i = 0; i < func.signature().input_arg_size(); ++i) {
      const OpDef::ArgDef &arg = func.signature().input_arg(i);

      // Create and add in graph a Call node for every input arg
      call = optimized_graph->add_node();
      call->set_name(strings::StrCat(func_node.name(), "/", "Call_", i));
      call->set_op("Call");
      call->set_device(device);
      call->add_input(func_node.input(i));
      TF_RETURN_IF_ERROR(CopyArgType(func_node, func_attr, "input", arg, &type));
      (*call->mutable_attr())["T"].set_type(type);
      (*call->mutable_attr())["frame_name"].set_s(func_node.op());
      (*call->mutable_attr())["call_id"].set_i(call_id);
      (*call->mutable_attr())["arg_id"].set_i(i);
      (*call->mutable_attr())["is_constant"].set_b(false);

      NodeDef* merge = argmerge_map[arg.name()];
      merge->add_input(call->name());
    }

    for (int i = 0; i < func.signature().output_arg_size(); ++i) {
      const OpDef::ArgDef &arg = func.signature().output_arg(i);

      NodeDef *ret = optimized_graph->add_node();
      ret->set_name(strings::StrCat(func_node.name(), "/", "Ret", i));
      ret->set_op("Return");
      ret->set_device(device);
      // Counting on the fact that op name will be the same as the name given initially to function
      ret->add_input(strings::StrCat(func_node.op(), "/", functions_in[func_node.op()].fetch[i]));
      TF_RETURN_IF_ERROR(CopyArgType(func_node, func_attr, "output", arg, &type));
      (*ret->mutable_attr())["T"].set_type(type);
      (*ret->mutable_attr())["frame_name"].set_s(func_node.op());
      (*ret->mutable_attr())["call_id"].set_i(call_id);
      (*ret->mutable_attr())["arg_id"].set_i(i);

      // Add a control input from Call to Returns
      *ret->add_input() = AsControlDependency(call->name());
    }
    return Status::OK();
}


Status InlineFunction(const NodeDef& func_node, const FunctionDef& func,
                      const FunctionInliningContext& ctx,
                      GraphDef* optimized_graph,
                      std::unordered_map<string, FuncInfo> &functions_in,
                      int& frame_name, string device) {

    int cpframe_name = frame_name;

    const std::unordered_map<string, AttrValue> func_attr(func_node.attr().begin(), func_node.attr().end());
    std::unique_ptr<GrapplerItem> item = GrapplerItemFromFunctionDef(func, func_attr, ctx.Library());

    if (!item) {
      return errors::InvalidArgument(
                "Failed to inline function ", func_node.op(),
                " instantiated by ", func_node.name());
    }

    std::set<string> foutputs;
    GatherOutputs(*item, ctx, foutputs);

    DataType type;
    std::unordered_map<string, int> input_nodes;
    functions_in[func_node.op()].fetch = item->fetch;
    ArgMergeMap& argmerge_map = functions_in[func_node.op()].argMergeMap;

    NodeDef* call;
    for (int i = 0; i < func.signature().input_arg_size(); ++i) {
      const OpDef::ArgDef& arg = func.signature().input_arg(i);

      input_nodes[arg.name()] = i;

      // Create and add in graph a Call node for every input arg
      call = optimized_graph->add_node();
      call->set_name(strings::StrCat(func_node.name(), "/", "Call_", i));
      call->set_op("Call");
      call->set_device(device);
      call->add_input(func_node.input(i));
      TF_RETURN_IF_ERROR(CopyArgType(func_node, func_attr, "input", arg, &type));
      (*call->mutable_attr())["T"].set_type(type);
      (*call->mutable_attr())["frame_name"].set_s(func_node.op());
      (*call->mutable_attr())["call_id"].set_i(frame_name);
      (*call->mutable_attr())["arg_id"].set_i(i);
      (*call->mutable_attr())["is_constant"].set_b(false);

      // Create and add a temporary merge node (IdentityN) for every input arg
      NodeDef* merge = optimized_graph->add_node();
      merge->set_name(strings::StrCat(func_node.name(), "/", "Merge_", i));
      merge->set_op("IdentityN");
      merge->set_device(device);
      merge->add_input(call->name());

      argmerge_map.emplace(arg.name(), merge);
    }

    for (NodeDef& func_body_node : *item->graph.mutable_node()) {
      // If the func body node is func's input argument
      if (input_nodes.find(func_body_node.name()) != input_nodes.end()) {
        CHECK_EQ(0, func_body_node.input_size());
        // Turn input placeholders into identity nodes
        if (IsPlaceholder(func_body_node)) {
          func_body_node.set_op("Identity");
        }
        // Connect merge with input arg
        func_body_node.add_input(argmerge_map[func_body_node.name()]->name());
      } else { // Else if not an input_arg_node
        // Update the input names if any.
        for (string& input : *func_body_node.mutable_input()) {

          // If it takes input from a function
          if (foutputs.find(input) != foutputs.end()) {
            input = ParseString(input);
          }
          input = AddPrefixToNodeName(input, /*prefix=*/func_node.name());
        }
        // If the node has no input, hook it up to the Merge nodes to ensure
        // it runs in the same frame as the other nodes of the function body.
        if (func_body_node.input_size() == 0) {
          for (auto it = argmerge_map.begin(); it != argmerge_map.end(); ++it) {
            *func_body_node.add_input() = AsControlDependency(it->second->name());
          }
        }
      }

      // Add the node name as a prefix to avoid collisions after inlining
      func_body_node.set_name(strings::StrCat(func_node.name(), "/", func_body_node.name()));

      // Make sure the node is placed
      string dvc = func_body_node.device();
      (dvc == "") ? (func_body_node.set_device(device)) : (func_body_node.set_device(dvc));

      // Check if a body node is itself a function
      const FunctionDef* func_body_node_func = ctx.FindInlinedFunction(func_body_node.op());

      // Node is yet another function
      if (func_body_node_func != nullptr) {

        // Check if that function has already been inlined
        auto it = functions_in.find(func_body_node.op());

        // Not already in => Inline it
        if (it == functions_in.end()) {
          FuncInfo func_info;
          functions_in.emplace(func_body_node.op(), func_info);
          InlineFunction(func_body_node, *func_body_node_func, ctx, optimized_graph, functions_in, ++frame_name, device);
          functions_in.erase(func_body_node.op());
        } else {
          // Already in -> Insert Enter/Exit ops end create cycle
          //  (recursion or mutually recursive functions)
          CreateCycle(func_body_node, *func_body_node_func, optimized_graph, functions_in, ++frame_name, device);
        }
      } else {
        // Move the node to the main graph
        optimized_graph->add_node()->Swap(&func_body_node);
      }
    }

    for (int i = 0; i < func.signature().output_arg_size(); ++i) {
      const OpDef::ArgDef &arg = func.signature().output_arg(i);

      NodeDef *ret = optimized_graph->add_node();
      ret->set_name(strings::StrCat(func_node.name(), "/", "Ret", i));
      ret->set_op("Return");
      ret->set_device(device);
      // If it takes input from a function
      string input = item->fetch[i];
      if (foutputs.find(input) != foutputs.end()) {
        input = ParseString(input);
      }

      ret->add_input(strings::StrCat(func_node.name(), "/", input));
      TF_RETURN_IF_ERROR(CopyArgType(func_node, func_attr, "output", arg, &type));
      (*ret->mutable_attr())["T"].set_type(type);
      (*ret->mutable_attr())["frame_name"].set_s(func_node.op());
      (*ret->mutable_attr())["call_id"].set_i(cpframe_name);
      (*ret->mutable_attr())["arg_id"].set_i(i);

      // Add a control input from Call to Returns
      *ret->add_input() = AsControlDependency(call->name());
    }

    int j=0;
    for (auto it = argmerge_map.begin(); it != argmerge_map.end(); ++it, ++j) {
        DataType type;
        NodeDef *new_merge, *merge = it->second;
        int i, size = merge->input_size();

        TF_RETURN_IF_ERROR(CopyArgType(func_node, func_attr,
                "input", func.signature().input_arg(j), &type));

        if (size <= 1) {
            merge->set_op("Identity");
            merge->set_device(device);
            (*merge->mutable_attr())["T"].set_type(type);
        } else {
            merge->set_op("Merge");
            merge->set_device(func_node.device());
            (*merge->mutable_attr())["T"].set_type(type);
            (*merge->mutable_attr())["N"].set_i(size);
        }
    }

    return Status::OK();
}

}  // namespace

Status FunctionTransformation::Optimize(Cluster* cluster, const GrapplerItem& item,
                                        GraphDef* optimized_graph) {
    FunctionInliningContext ctx(item);

    int frame_name = 0;
    std::set<string> foutputs;

    GatherOutputs(item, ctx, foutputs);

    //std::cout << foutputs.size() << '\n';
    //for( const auto& str : foutputs ) std::cout << str << '\n';

    // Nothing to do here.
    if (!ctx.HasInlinedFunctions()) {
      *optimized_graph = item.graph;
      return Status::OK();
    }

    std::unordered_map<string, FuncInfo> functions_in;

    // Copying node cause I need to make changes on it
    for (NodeDef node : item.graph.node()) {
      for (string& input : *node.mutable_input()) {
        // If it takes input from a function
        if (foutputs.find(input) != foutputs.end()) {
          input = ParseString(input);
        }
      }

      const FunctionDef* func = ctx.FindInlinedFunction(node.op());
      if (func != nullptr) {
        FuncInfo func_info;
        // All the special nodes of this function and its 'callee-functions' too,
        // will colocate in the same device (important for distributed)
        string device = node.device();
        functions_in.emplace(node.op(), func_info);
        InlineFunction(node, *func, ctx, optimized_graph, functions_in, frame_name, device);
        functions_in.erase(node.op());      // At this point functions_in will be empty

        // Check if the function node corresponded to some fetch_outputs
        // before transformation occurred
        NodeDef *idN;
        bool created = false;
        const std::unordered_map<string, AttrValue> func_attr(node.attr().begin(), node.attr().end());

        for (size_t i = 0; i < item.fetch.size(); ++i) {
          const string &t = item.fetch[i];
          // Parse t into node_name and output_index.
          TensorId id(ParseTensorName(t));

          if (node.name() == id.first) {

            if (created == false) {
              idN = optimized_graph->add_node();
              idN->set_op("IdentityN");
              idN->set_name(node.name());
              idN->set_device(device);

              AttrValue::ListValue* type_list = (*idN->mutable_attr())["T"].mutable_list();
              for (const OpDef::ArgDef& arg : func->signature().output_arg()) {
                TF_RETURN_IF_ERROR(CopyArgType(node, func_attr, "input", arg, type_list));
              }

              idN->add_input(strings::StrCat(node.name(), "/Ret", id.second));

              created = true;
            } else {
              idN->add_input(strings::StrCat(node.name(), "/Ret", id.second));
            }
          }
        }
      } else {
        *optimized_graph->add_node() = node;
      }
    }

    *optimized_graph->mutable_versions() = item.graph.versions();
    *optimized_graph->mutable_library() = item.graph.library();

    /******************************************************************************************************
    // Dumps optimized graph in a not so readable form
    // const GraphDef* tmp = optimized_graph;
    // printf("Summarize Optimized Graph\n %s\n", SummarizeGraphDef(*tmp).c_str());

    // Write an event, so that we can visualize this optimized graph in tensorboard
    EventsWriter writer("TRANSFORMATION");
    Event event;
    event.set_wall_time(1234);
    event.set_step(34);

    const size_t proto_size = optimized_graph->ByteSizeLong();
    void* buf = port::Malloc(proto_size);
    if (buf == nullptr) {
      return errors::ResourceExhausted(
                "Failed to allocate memory to serialize message of type '" ,
                optimized_graph->GetTypeName(), "' and size ", proto_size);
    }
    optimized_graph->SerializeToArray(buf, proto_size);
    const void* bf = buf;
    event.set_graph_def(bf, proto_size);
    writer.WriteEvent(event);
    ******************************************************************************************************/

    return Status::OK();
}

void FunctionTransformation::Feedback(Cluster* cluster, const GrapplerItem& item,
                                      const GraphDef& optimized_graph,
                                      double result) {
    // Nothing to do for FunctionTransformation.
}

}  // end namespace grappler
}  // end namespace tensorflow
