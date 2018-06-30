/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
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

//            std::cout << res << std::endl;

            return res;
          }

          Status GatherOutputs(std::set<string> &foutputs, const GrapplerItem& item,
                               const FunctionInliningContext& function_inlining_ctx) {

            for (const NodeDef& node : item.graph.node()) {

              const FunctionDef* func = function_inlining_ctx.FindInlinedFunction(node.op());
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
                                std::unordered_map<string, FuncInfo> &functions_in) {

//            printf("Recursion Detected\n");

            const std::unordered_map<string, AttrValue> func_attr(func_node.attr().begin(), func_node.attr().end());

            DataType type;
            ArgMergeMap& argmerge_map = functions_in[func_node.op()].argMergeMap;

            for (int i = 0; i < func.signature().input_arg_size(); ++i) {
              const OpDef::ArgDef &arg = func.signature().input_arg(i);

              // Create and add in graph a Call node for every input arg
              NodeDef *call = optimized_graph->add_node();
              call->set_name(strings::StrCat(func_node.name(), "/", "Call_", i));
              call->set_op("Call");
              call->set_device(func_node.device());
              call->add_input(func_node.input(i));
              TF_RETURN_IF_ERROR(CopyArgType(func_node, func_attr, "input", arg, &type));
              (*call->mutable_attr())["T"].set_type(type);
              (*call->mutable_attr())["frame_name"].set_s(strings::StrCat(func_node.name()));
              (*call->mutable_attr())["is_constant"].set_b(false);

              NodeDef* merge = argmerge_map[arg.name()];
              merge->add_input(call->name());
            }

            for (int i = 0; i < func.signature().output_arg_size(); ++i) {
              const OpDef::ArgDef &arg = func.signature().output_arg(i);

              NodeDef *ret = optimized_graph->add_node();
              ret->set_name(strings::StrCat(func_node.name(), "/", "Ret", i));
              ret->set_op("Return");
              ret->set_device(func_node.device());
              // Counting on the fact that op name will be the same as the name given initially to function
              ret->add_input(strings::StrCat(func_node.op(), "/", functions_in[func_node.op()].fetch[i]));
              TF_RETURN_IF_ERROR(CopyArgType(func_node, func_attr, "output", arg, &type));
              (*ret->mutable_attr())["T"].set_type(type);
              (*ret->mutable_attr())["frame_name"].set_s(strings::StrCat(func_node.name()));

            }

            return Status::OK();
          }


          Status InlineFunction(const NodeDef& func_node, const FunctionDef& func, const FunctionInliningContext& ctx,
                              GraphDef* optimized_graph, std::unordered_map<string, FuncInfo> &functions_in) {

            const std::unordered_map<string, AttrValue> func_attr(func_node.attr().begin(), func_node.attr().end());

            std::unique_ptr<GrapplerItem> item = GrapplerItemFromFunctionDef(func, func_attr, ctx.Library());
            if (!item) {
              return errors::InvalidArgument("Failed to inline function ", func_node.op(), " instantiated by ", func_node.name());
            }

            std::set<string> foutputs;
            GatherOutputs(foutputs, *item, ctx);

//std::cout << foutputs.size() << '\n';
//for( const auto& str : foutputs ) std::cout << str << '\n';

            DataType type;
            std::unordered_map<string, int> input_nodes;
            functions_in[func_node.op()].fetch = item->fetch;
            ArgMergeMap& argmerge_map = functions_in[func_node.op()].argMergeMap;

            for (int i = 0; i < func.signature().input_arg_size(); ++i) {
              const OpDef::ArgDef& arg = func.signature().input_arg(i);

              input_nodes[arg.name()] = i;

              // Create and add in graph a Call node for every input arg
              NodeDef* call = optimized_graph->add_node();
              call->set_name(strings::StrCat(func_node.name(), "/", "Call_", i));
              call->set_op("Call");
              call->set_device(func_node.device());
              call->add_input(func_node.input(i));
              TF_RETURN_IF_ERROR(CopyArgType(func_node, func_attr, "input", arg, &type));
              (*call->mutable_attr())["T"].set_type(type);
              (*call->mutable_attr())["frame_name"].set_s(strings::StrCat(func_node.name()));
              (*call->mutable_attr())["is_constant"].set_b(false);

              // Create and add a temporary merge node (IdentityN) for every input arg
              NodeDef* merge = optimized_graph->add_node();
              merge->set_name(strings::StrCat(func_node.name(), "/", "Merge_", i));
              merge->set_op("IdentityN");
              merge->set_device(func_node.device());
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
              }

              // Else if not an input_arg_node
              else {
                // Update the input names if any.
                for (string& input : *func_body_node.mutable_input()) {

                  // If it takes input from a function
                  if (foutputs.find(input) != foutputs.end()) {
                    input = ParseString(input);
                  }
                  input = AddPrefixToNodeName(input, /*prefix=*/func_node.name());
                }

                // If the node has no input, make hook it up to the Merge nodes to ensure
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
              func_body_node.set_device(func_node.device());

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
                  InlineFunction(func_body_node, *func_body_node_func, ctx, optimized_graph, functions_in);
                  functions_in.erase(func_body_node.op());
                }
                // Already in -> Insert Enter/Exit ops end create cycle
                //  (recursion or mutually recursive functions)
                else {
                  CreateCycle(func_body_node, *func_body_node_func, optimized_graph, functions_in);
                }
              }

              else {
                // Move the node to the main graph
                optimized_graph->add_node()->Swap(&func_body_node);
              }
            }

            for (int i = 0; i < func.signature().output_arg_size(); ++i) {
              const OpDef::ArgDef &arg = func.signature().output_arg(i);

              NodeDef *ret = optimized_graph->add_node();
              ret->set_name(strings::StrCat(func_node.name(), "/", "Ret", i));
              ret->set_op("Return");
              ret->set_device(func_node.device());
              // If it takes input from a function
              string input = item->fetch[i];
              if (foutputs.find(input) != foutputs.end()) {
                input = ParseString(input);
              }

              ret->add_input(strings::StrCat(func_node.name(), "/", input));
              TF_RETURN_IF_ERROR(CopyArgType(func_node, func_attr, "output", arg, &type));
              (*ret->mutable_attr())["T"].set_type(type);
              (*ret->mutable_attr())["frame_name"].set_s(strings::StrCat(func_node.name()));
            }

            // Break IdentityN Merges into multiple common Merge ops
            int j=0;
            for (auto it = argmerge_map.begin(); it != argmerge_map.end(); ++it, ++j) {

              DataType type;
              NodeDef *new_merge, *merge = it->second;
              int i, size = merge->input_size();
              TF_RETURN_IF_ERROR(CopyArgType(func_node, func_attr, "input", func.signature().input_arg(j), &type));

              // If there is only one call site
              if (size < 2) {
                merge->set_op("Identity");
                merge->set_device(func_node.device());
                (*merge->mutable_attr())["T"].set_type(type);
              }

              else {

                string name = merge->name();
                string in1 = merge->input(0), in2;

                for (i = 1; i < size-1; i++) {

                  in2 = merge->input(i);
                  new_merge = optimized_graph->add_node();

                  name = strings::StrCat(name, size - i - 1);
                  new_merge->set_name(name);
                  new_merge->set_op("Merge");
                  new_merge->set_device(func_node.device());
                  new_merge->add_input(in1);
                  new_merge->add_input(in2);
                  (*new_merge->mutable_attr())["T"].set_type(type);
                  (*new_merge->mutable_attr())["N"].set_i(2);

                  in1 = name;
                }

                // Modify initial Merge
                in2 = merge->input(i);
                merge->set_op("Merge");
                merge->set_device(func_node.device());
                merge->clear_input();
                merge->add_input(in1);
                merge->add_input(in2);
                (*merge->mutable_attr())["T"].set_type(type);
                (*merge->mutable_attr())["N"].set_i(2);
              }
            }

            return Status::OK();
          }
/*
            class FakeCPUDevice : public Device {
            public:
                FakeCPUDevice(Env* env, const DeviceAttributes& attr) : Device(env, attr) {}
                Status Sync() override { return Status::OK(); }
            };

            class SymbolicGradientEnv {
            public:
                SymbolicGradientEnv(int graph_version, const FunctionDefLibrary& library)
                        : graph_version_(graph_version), library_(library) {}

                FunctionLibraryDefinition* function_library() {
                  InitializeIfNeeded();
                  return fld_.get();
                }
                FunctionLibraryRuntime* function_library_runtime() {
                  InitializeIfNeeded();
                  return flr_;
                }

            private:
                // This initialization is expensive. Do it lazily to avoid paying for it
                // unless it's needed.
                void InitializeIfNeeded() {
                  if (flr_) {
                    return;
                  }
                  Env* env = Env::Default();
                  DeviceAttributes attr;
                  attr.set_name("/device:CPU:0");
                  attr.set_device_type("CPU");
                  FakeCPUDevice* dev = new FakeCPUDevice(env, attr);
                  std::vector<Device*> devices;
                  devices.push_back(dev);
                  dvc_mgr_.reset(new DeviceMgr(devices));
                  fld_.reset(new FunctionLibraryDefinition(OpRegistry::Global(), library_));
                  OptimizerOptions optimizer_opts;
                  optimizer_opts.set_do_function_inlining(true);
                  pflr_.reset(new ProcessFunctionLibraryRuntime(
                          dvc_mgr_.get(), env, graph_version_, fld_.get(), optimizer_opts));
                  flr_ = pflr_->GetFLR(dev->name());
                }

                const int graph_version_;
                const FunctionDefLibrary& library_;
                std::unique_ptr<DeviceMgr> dvc_mgr_;
                std::unique_ptr<FunctionLibraryDefinition> fld_;
                std::unique_ptr<ProcessFunctionLibraryRuntime> pflr_;
                FunctionLibraryRuntime* flr_ = nullptr;
            };

            Status InlineSymbolicGradient(const NodeDef& node, SymbolicGradientEnv* env,
                                          GraphDef* inlined_graph)
            {
              GraphDef graph_def;

              // Create a node to anchor the gradient inputs
              NodeDef* inlined_input = graph_def.add_node();
              inlined_input->set_name("FunctionInputs");
              inlined_input->set_op("IdentityN");
              AttrValue::ListValue* type_list =
                      (*inlined_input->mutable_attr())["T"].mutable_list();
              for (const auto& type : node.attr().at("Tin").list().type()) {
                type_list->add_type(static_cast<DataType>(type));
              }

              // Add the gradient node
              NodeDef* inlined = graph_def.add_node();
              *inlined = node;
              inlined->clear_input();
              for (int i = 0; i < node.attr().at("Tin").list().type_size(); ++i) {
                inlined->add_input(strings::StrCat(inlined_input->name(), ":", i));
              }

              // Create a node to anchor the gradient outputs
              NodeDef* inlined_output = graph_def.add_node();
              inlined_output->set_name("FunctionOutputs");
              inlined_output->set_op("IdentityN");
              type_list = (*inlined_output->mutable_attr())["T"].mutable_list();
              for (const auto& type : node.attr().at("Tout").list().type()) {
                type_list->add_type(static_cast<DataType>(type));
              }
              for (int i = 0; i < node.attr().at("Tout").list().type_size(); ++i) {
                inlined_output->add_input(strings::StrCat(inlined->name(), ":", i));
              }

              // Convert the graphdef to a graph
              GraphConstructorOptions graph_ctor_opts;
              graph_ctor_opts.allow_internal_ops = true;
              graph_ctor_opts.expect_device_spec = false;
              Graph graph(env->function_library());
              TF_RETURN_IF_ERROR(
                      ConvertGraphDefToGraph(graph_ctor_opts, graph_def, &graph));

              // Recursively inline the functions until there is nothing more to inline. We
              // should at least expand one function.
              int counter = 0;
              while (counter < 50 &&
                     ExpandInlineFunctions(env->function_library_runtime(), &graph)) {
                ++counter;
              }

              GraphDef inlined_graph_def;
              graph.ToGraphDef(&inlined_graph_def);

              // Add the default values of attributes to the nodes that have been inlined.
              TF_RETURN_IF_ERROR(AddDefaultAttrsToGraphDef(&inlined_graph_def, *graph.op_registry(), 0, true));

              // Add the inlined nodes to the graph
              for (NodeDef& inlined_node : *inlined_graph_def.mutable_node()) {
                if (inlined_node.name() == "FunctionOutputs") {
                  inlined_node.set_name(node.name());
                  for (int i = 0; i < inlined_node.input_size(); ++i) {
                    inlined_node.set_input(
                            i, AddPrefixToNodeName(inlined_node.input(i), node.name()));
                  }
                } else if (inlined_node.name() == "FunctionInputs") {
                  inlined_node.set_name(
                          AddPrefixToNodeName(inlined_node.name(), node.name()));
                  inlined_node.clear_input();
                  for (int i = 0; i < node.input_size(); ++i) {
                    inlined_node.add_input(node.input(i));
                  }
                } else {
                  inlined_node.set_name(
                          AddPrefixToNodeName(inlined_node.name(), node.name()));
                  for (int i = 0; i < inlined_node.input_size(); ++i) {
                    inlined_node.set_input(
                            i, AddPrefixToNodeName(inlined_node.input(i), node.name()));
                  }
                  // If the node has no input, hook it up to the function input node to make
                  // sure it runs in the same frame as the other nodes of the function body.
                  if (inlined_node.input_size() == 0) {
                    *inlined_node.add_input() = AsControlDependency(
                            AddPrefixToNodeName("FunctionInputs", node.name()));
                  }
                }
                inlined_node.set_device(node.device());
                inlined_graph->add_node()->Swap(&inlined_node);
              }

              return Status::OK();
            }
*/
        }  // namespace


        Status FunctionTransformation::Optimize(Cluster* cluster, const GrapplerItem& item,
                                                GraphDef* optimized_graph) {

          FunctionInliningContext function_inlining_ctx(item);

          std::set<string> foutputs;
          GatherOutputs(foutputs, item, function_inlining_ctx);

//std::cout << foutputs.size() << '\n';
//for( const auto& str : foutputs ) std::cout << str << '\n';

          // Nothing to do here.
          if (!function_inlining_ctx.HasInlinedFunctions()) {
            *optimized_graph = item.graph;
            return Status::OK();
          }

//          SymbolicGradientEnv env(item.graph.versions().producer(),item.graph.library());

          std::unordered_map<string, FuncInfo> functions_in;

          // Copying node cause I need to make changes on it
          for (NodeDef node : item.graph.node()) {
//            if (node.op() == "SymbolicGradient") {
//              TF_RETURN_IF_ERROR(InlineSymbolicGradient(node, &env, optimized_graph));
//              continue;
//            }

            for (string& input : *node.mutable_input()) {

              // If it takes input from a function
              if (foutputs.find(input) != foutputs.end()) {
                input = ParseString(input);
              }
            }

            const FunctionDef* func = function_inlining_ctx.FindInlinedFunction(node.op());
            if (func != nullptr) {
              FuncInfo func_info;
              functions_in.emplace(node.op(), func_info);
              InlineFunction(node, *func, function_inlining_ctx, optimized_graph, functions_in);
              functions_in.erase(node.op());      // At this point functions_in will be empty
            }
            else {
              *optimized_graph->add_node() = node;
            }
          }

          *optimized_graph->mutable_versions() = item.graph.versions();
          *optimized_graph->mutable_library() = item.graph.library();


          /******************************************************************************************************/
          // Dumps optimized graph in a not so readable form
          const GraphDef* tmp = optimized_graph;
          printf("Summarize Optimized Graph\n %s\n", SummarizeGraphDef(*tmp).c_str());

          // Write an event, so that we can visualize this optimized graph in tensorboard
          EventsWriter writer("INLINE");
          Event event;
          event.set_wall_time(1234);
          event.set_step(34);

          const size_t proto_size = optimized_graph->ByteSizeLong();
          void* buf = port::Malloc(proto_size);
          if (buf == nullptr) {
            return tensorflow::errors::ResourceExhausted("Failed to allocate memory to serialize message of type '"
                                                         ,optimized_graph->GetTypeName(), "' and size ", proto_size);
          }
          optimized_graph->SerializeToArray(buf, proto_size);
          const void* bf = buf;
          event.set_graph_def(bf, proto_size);
          writer.WriteEvent(event);
          /******************************************************************************************************/

          return Status::OK();
        }

        void FunctionTransformation::Feedback(Cluster* cluster, const GrapplerItem& item,
                                         const GraphDef& optimized_graph,
                                         double result) {
          // Nothing to do for FunctionOptimizer.
        }

    }  // end namespace grappler
}  // end namespace tensorflow
