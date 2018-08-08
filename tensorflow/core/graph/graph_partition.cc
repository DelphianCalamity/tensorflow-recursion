/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/graph/graph_partition.h"

#include <deque>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "tensorflow/core/framework/memory_types.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/control_flow.h"
#include "tensorflow/core/graph/costmodel.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/tensor_id.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {

namespace {

inline bool IsMerge(const NodeDef& node_def) {
  return node_def.op() == "Merge" || node_def.op() == "RefMerge";
}

inline bool IsNextIteration(const NodeDef& node_def) {
  return node_def.op() == "NextIteration" ||
         node_def.op() == "RefNextIteration";
}

struct DupRecvKey {
  int src_node_id;           // Edge's src node id
  int src_output_slot;       // Edge's src node output slot
  GraphDef* dst_graph;       // Edge's dst node is in this subgraph
  bool recv_output_on_host;  // The output of recv is on host
};

struct DupRecvKeyHash {
  size_t operator()(const DupRecvKey& k) const {
    size_t h = Hash64(reinterpret_cast<const char*>(&k.src_node_id),
                      sizeof(k.src_node_id), k.src_output_slot);
    h = Hash64(reinterpret_cast<const char*>(&k.dst_graph), sizeof(k.dst_graph),
               h);
    h = Hash64(reinterpret_cast<const char*>(&k.recv_output_on_host),
               sizeof(k.recv_output_on_host), h);
    return h;
  }
};

struct DupRecvKeyEq {
  bool operator()(const DupRecvKey& x, const DupRecvKey& y) const {
    return (x.src_node_id == y.src_node_id) &&
           (x.src_output_slot == y.src_output_slot) &&
           (x.dst_graph == y.dst_graph) &&
           (x.recv_output_on_host == y.recv_output_on_host);
  }
};

// struct used to store the recvs, so that start times can be properly updated
struct RecvInfo {
  NodeDef* recv;
  NodeDef* real_recv;
  int64 start_time;
};

typedef std::unordered_map<DupRecvKey, RecvInfo, DupRecvKeyHash, DupRecvKeyEq>
    DupRecvTable;

struct PairIntHash {
 public:
  std::size_t operator()(const std::pair<int, int>& x) const {
    return std::hash<int>()(x.first) ^ std::hash<int>()(x.second);
  }
};
// A map used to store memory types for the inputs/outputs of every node.
// The key is a pair of ints consisting of a node id and input/output index.
typedef std::unordered_map<std::pair<int, int>, MemoryType, PairIntHash>
    MemoryTypeMap;

// We collect the following information about the graph before performing
// graph partitioning.
struct GraphInfo {
  std::vector<DeviceType> device_types;
  MemoryTypeMap input_types;
  MemoryTypeMap output_types;
  std::vector<ControlFlowInfo> cf_info;
};

DataType EdgeType(const Edge* e) {
  if (e->IsControlEdge()) {
    return DT_FLOAT;
  } else {
    return e->dst()->input_type(e->dst_input());
  }
}

// Return true iff we need to add a same device send/recv for 'edge'.
bool NeedSameDeviceSendRecv(const Edge* edge, const GraphInfo& info) {
  if (edge->IsControlEdge()) {
    return false;
  }

  Node* src = edge->src();
  Node* dst = edge->dst();
  if (src->assigned_device_name() == dst->assigned_device_name()) {
    int src_port = edge->src_output();
    int dst_port = edge->dst_input();
    if (info.device_types[src->id()] != DEVICE_CPU) {
      auto src_it = info.output_types.find({src->id(), src_port});
      DCHECK(src_it != info.output_types.end());
      auto dst_it = info.input_types.find({dst->id(), dst_port});
      DCHECK(dst_it != info.input_types.end());
      return src_it->second != dst_it->second;
    }
  }
  return false;
}

// Return true iff (dst, dst_input) is specified on host memory.
bool IsDstInputOnHost(const Edge* edge, const GraphInfo& info) {
  Node* dst = edge->dst();
  int dst_port = edge->dst_input();
  if (info.device_types[dst->id()] != DEVICE_CPU) {
    if (edge->IsControlEdge()) return false;
    auto dst_it = info.input_types.find({dst->id(), dst_port});
    DCHECK(dst_it != info.input_types.end());
    return dst_it->second == HOST_MEMORY;
  }
  return true;
}

// Add an input to dst that comes from the "src_slot" output of the
// node named by "src_name".
void AddInput(NodeDef* dst, StringPiece src_name, int src_slot) {
  if (src_slot == Graph::kControlSlot) {
    dst->add_input(strings::StrCat("^", src_name));
  } else if (src_slot == 0) {
    dst->add_input(src_name.data(), src_name.size());
  } else {
    dst->add_input(strings::StrCat(src_name, ":", src_slot));
  }
}

// Add a control edge from each input to each recv.
void AddReadControl(const std::vector<NodeDef*>& recvs,
                    const std::vector<string>& inputs) {
  for (NodeDef* recv : recvs) {
    for (const string& input : inputs) {
      recv->add_input(strings::StrCat("^", input));
    }
  }
}

void SetSendRecvAttrs(const PartitionOptions& opts, const Edge* edge,
                      NodeDefBuilder* builder) {
  builder->Attr("tensor_name",
                strings::StrCat("edge_", edge->id(), "_", edge->src()->name()));
  builder->Attr("send_device", edge->src()->assigned_device_name());
  builder->Attr("send_device_incarnation",
                static_cast<int64>(
                    opts.get_incarnation(edge->src()->assigned_device_name())));
  builder->Attr("recv_device", edge->dst()->assigned_device_name());
  builder->Attr("client_terminated", false);
}

NodeDef* AddSend(const PartitionOptions& opts, const GraphInfo& g_info,
                 GraphDef* gdef, const Edge* edge,
                 NodeDefBuilder::NodeOut send_from, int64 start_time,
                 Status* status) {
  const DataType dtype = send_from.data_type;
  const DataType cast_dtype = opts.should_cast ? opts.should_cast(edge) : dtype;
  const Node* src = edge->src();
  const int src_port = edge->src_output();

  // host_memory = true iff we need to use HostSend/HostCast.
  bool host_memory = false;
  if (!edge->IsControlEdge()) {
    auto src_it = g_info.output_types.find({src->id(), src_port});
    DCHECK(src_it != g_info.output_types.end());
    host_memory = (src_it->second == HOST_MEMORY);
  }

  // Add a cast node that casts dtype to cast_dtype.
  // NOTE(yuanbyu): Only cast for cross-device send/recv.
  if (dtype != cast_dtype && !NeedSameDeviceSendRecv(edge, g_info)) {
    const string cast_op = (host_memory) ? "_HostCast" : "Cast";
    NodeDefBuilder cast_builder(opts.new_name(src->name()), cast_op);
    cast_builder.Device(src->assigned_device_name()).Input(send_from);
    if (opts.scheduling_for_recvs) {
      cast_builder.Attr("_start_time", start_time);
    }
    cast_builder.Attr("DstT", cast_dtype);
    NodeDef* cast = gdef->add_node();
    *status = cast_builder.Finalize(cast);
    if (!status->ok()) return nullptr;

    // Connect the Send op to the cast.
    send_from.Reset(cast->name(), 0, cast_dtype);
  }

  // Add the send node.
  const string send_op = (host_memory) ? "_HostSend" : "_Send";
  NodeDefBuilder send_builder(opts.new_name(src->name()), send_op);
  SetSendRecvAttrs(opts, edge, &send_builder);
  send_builder.Device(src->assigned_device_name()).Input(send_from);
  if (opts.scheduling_for_recvs) {
    send_builder.Attr("_start_time", start_time);
  }
  NodeDef* send = gdef->add_node();
  *status = send_builder.Finalize(send);
  return send;
}

NodeDef* AddRecv(const PartitionOptions& opts, const GraphInfo& g_info,
                 GraphDef* gdef, const Edge* edge, NodeDef** real_recv,
                 Status* status) {
  const DataType dtype = EdgeType(edge);
  const Node* src = edge->src();
  const Node* dst = edge->dst();
  const int dst_port = edge->dst_input();
  DataType cast_dtype = dtype;

  // NOTE(yuanbyu): Only cast for cross-device send/recv.
  if (opts.should_cast && !NeedSameDeviceSendRecv(edge, g_info)) {
    cast_dtype = opts.should_cast(edge);
  }

  // host_memory = true iff we need to use HostRecv/HostCast.
  bool host_memory = false;
  if (!edge->IsControlEdge()) {
    auto dst_it = g_info.input_types.find({dst->id(), dst_port});
    DCHECK(dst_it != g_info.input_types.end());
    host_memory = (dst_it->second == HOST_MEMORY);
  }

  // Add the recv node.
  const string recv_op = (host_memory) ? "_HostRecv" : "_Recv";
  NodeDefBuilder recv_builder(opts.new_name(src->name()), recv_op);
  SetSendRecvAttrs(opts, edge, &recv_builder);
  recv_builder.Device(dst->assigned_device_name())
      .Attr("tensor_type", cast_dtype);
  NodeDef* recv = gdef->add_node();
  *status = recv_builder.Finalize(recv);
  if (!status->ok()) return nullptr;
  *real_recv = recv;

  // Add the cast node (from cast_dtype to dtype) or an Identity node.
  if (dtype != cast_dtype) {
    const string cast_op = (host_memory) ? "_HostCast" : "Cast";
    NodeDefBuilder cast_builder(opts.new_name(src->name()), cast_op);
    cast_builder.Attr("DstT", dtype);
    cast_builder.Device(dst->assigned_device_name())
        .Input(recv->name(), 0, cast_dtype);
    NodeDef* cast = gdef->add_node();
    *status = cast_builder.Finalize(cast);
    if (!status->ok()) return nullptr;
    return cast;
  } else if (edge->IsControlEdge()) {
    // An Identity is only needed for control edges.
    NodeDefBuilder id_builder(opts.new_name(src->name()), "Identity");
    id_builder.Device(dst->assigned_device_name())
        .Input(recv->name(), 0, cast_dtype);
    NodeDef* id = gdef->add_node();
    *status = id_builder.Finalize(id);
    if (!status->ok()) return nullptr;
    return id;
  } else {
    return recv;
  }
}

NodeDef* AddDummyConst(const PartitionOptions& opts, GraphDef* gdef,
                       const Edge* edge, Status* status) {
  const Node* src = edge->src();
  Tensor tensor(DT_FLOAT, TensorShape({0}));
  NodeDef* result = gdef->add_node();
  *status = NodeDefBuilder(opts.new_name(src->name()), "Const")
                .Device(src->assigned_device_name())
                .Attr("dtype", DT_FLOAT)
                .Attr("value", tensor)
                .Finalize(result);
  return result;
}

// A dummy node for scheduling.
NodeDef* AddControlTrigger(const PartitionOptions& opts, GraphDef* gdef,
                           const string& assigned_device_name, int64 epoch,
                           int64 starttime, Status* status) {
  NodeDef* result = gdef->add_node();
  *status = NodeDefBuilder(opts.new_name(strings::StrCat("synch_", epoch)),
                           "ControlTrigger")
                .Device(assigned_device_name)
                .Attr("_start_time", starttime)
                .Finalize(result);
  return result;
}

// Optimize colocation for control flow nodes. For cond, we want the
// switch nodes to colocate with its data input. This is particularly
// needed for conditional reading of a remote variable. It may also
// reduce the number of devices involved in a loop.
// TODO(yuanbyu): In this case, we don't respect the requested device in
// the GraphDef for these nodes. Ideally, the placer would enforce the
// colocation to render this unnecessary.
void OptimizeControlFlowColocation(Graph* graph) {
  auto visit = [](Node* node) {
    if (IsSwitch(node)) {
      for (const Edge* in_edge : node->in_edges()) {
        if (in_edge->dst_input() == 0) {
          // Colocate with the data input.
          node->set_assigned_device_name(
              in_edge->src()->assigned_device_name());
          return;
        }
      }
    } else if (IsExit(node)) {
      for (const Edge* in_edge : node->in_edges()) {
        if (!in_edge->IsControlEdge()) {
          // Colocate with upstream node.
          node->set_assigned_device_name(
              in_edge->src()->assigned_device_name());
          return;
        }
      }
    } else {
      if ((IsEnter(node) && !IsRefType(node->input_type(0))) ||
          IsNextIteration(node)) {
        const Edge* data_edge = nullptr;
        for (const Edge* out_edge : node->out_edges()) {
          if (!out_edge->IsControlEdge()) {
            data_edge = out_edge;
            break;
          }
        }
        // Colocate with the first downstream data node.
        if (data_edge) {
          node->set_assigned_device_name(
              data_edge->dst()->assigned_device_name());
        }
      }
    }
  };
  DFS(*graph, visit, {});
}

string ControlLoopName(const string& name) {
  return strings::StrCat("_cloop", name);
}

bool IsControlLoop(const Node* node) {
  const string& name = node->name();
  return StringPiece(name).starts_with("_cloop");
}

// An enter node for control flow.
Node* AddControlEnter(Graph* g, const string& node_name,
                      const string& device_name, const string& frame_name,
                      const int parallel_iterations, Status* status) {
  NodeBuilder node_builder(node_name, "Enter", g->op_registry());
  node_builder.Input({"dummy", 0, DT_FLOAT});
  node_builder.Attr("frame_name", frame_name);
  node_builder.Attr("parallel_iterations", parallel_iterations);
  Node* res_node;
  *status = node_builder.Finalize(g, &res_node);
  if (!status->ok()) return nullptr;
  res_node->set_assigned_device_name(device_name);
  return res_node;
}

// A merge node for control flow.
Node* AddControlMerge(const string& in_name1, const string& in_name2, Graph* g,
                      const string& node_name, const string& device_name,
                      Status* status) {
  NodeBuilder node_builder(node_name, "Merge", g->op_registry());
  node_builder.Input({{in_name1, 0, DT_FLOAT}, {in_name2, 0, DT_FLOAT}});
  Node* res_node;
  *status = node_builder.Finalize(g, &res_node);
  if (!status->ok()) return nullptr;
  res_node->set_assigned_device_name(device_name);
  return res_node;
}

// A switch node for control flow.
Node* AddControlSwitch(NodeBuilder::NodeOut input1, NodeBuilder::NodeOut input2,
                       const string& device_name,
                       const GraphDefBuilder::Options& bopts) {
  Node* res_node =
      ops::BinaryOp("Switch", std::move(input1), std::move(input2), bopts);
  if (bopts.HaveError()) return nullptr;
  res_node->set_assigned_device_name(device_name);
  return res_node;
}

// A next_iteration node for control flow.
Node* AddControlNext(NodeBuilder::NodeOut input, const string& device_name,
                     const GraphDefBuilder::Options& bopts) {
  Node* res_node = ops::UnaryOp("NextIteration", std::move(input), bopts);
  if (bopts.HaveError()) return nullptr;
  res_node->set_assigned_device_name(device_name);
  return res_node;
}

Node* EmptyConst(const GraphDefBuilder::Options& options) {
  if (options.HaveError()) return nullptr;
  NodeBuilder node_builder(options.GetNameForOp("Const"), "Const",
                           options.op_registry());
  const DataType dt = DataTypeToEnum<float>::v();
  TensorProto proto;
  proto.set_dtype(dt);
  TensorShape empty_shape({0});
  empty_shape.AsProto(proto.mutable_tensor_shape());
  node_builder.Attr("dtype", dt).Attr("value", proto);
  return options.FinalizeBuilder(&node_builder);
}

// A dummy const node for control flow.
Node* AddControlConst(const string& device_name,
                      const GraphDefBuilder::Options& bopts) {
  Node* res_node = EmptyConst(bopts);
  if (bopts.HaveError()) return nullptr;
  res_node->set_assigned_device_name(device_name);
  return res_node;
}

// A synthetic loop, made up of dummy nodes. It performs control-flow actions
// on behalf of a leader on a different device.
struct ControlLoop {
  Node* enter = nullptr;
  Node* merge = nullptr;
  Node* switch_node = nullptr;
};

// Add the control flow info of a new node added during partitioning.
// The new node has the same control flow info as src.
void AddControlFlowInfo(const Node* node, const Node* src,
                        std::vector<ControlFlowInfo>* cf_info) {
  int id = node->id();
  if (static_cast<size_t>(id) >= cf_info->size()) {
    cf_info->resize(id + 1);
  }
  const ControlFlowInfo& src_info = (*cf_info)[src->id()];
  ControlFlowInfo* info = &(*cf_info)[id];
  info->frame = src_info.frame;
  info->parent_frame = src_info.parent_frame;
  info->frame_name = src_info.frame_name;
}

// Constructs a control loop. Returns a struct containing the newly created
// enter, merge, and switch nodes. The enter and merge nodes are used in the
// recursive construction of control loops for nested frames (loops). The
// switch node will be connected to the LoopCond node. The merge node will
// be connected to all the recvs of the same frame by control edges when
// the actual partitioning happens.
Status AddControlLoop(const PartitionOptions& opts, Graph* g, const Node* src,
                      const Edge* edge, Node* loop_cond,
                      std::vector<ControlFlowInfo>* cf_info,
                      ControlLoop* loop) {
  Status status;
  GraphDefBuilder::Options bopts(g, &status);
  const ControlFlowInfo& src_info = (*cf_info)[src->id()];
  const string& device_name = edge->dst()->assigned_device_name();
  const string& frame_name = src_info.frame_name;
  int parallel_iterations;
  status = GetNodeAttr(src_info.frame->attrs(), "parallel_iterations",
                       &parallel_iterations);
  if (!status.ok()) return status;

  // The names of the nodes to be added.
  const string& enter_name =
      ControlLoopName(opts.new_name(edge->dst()->name()));
  const string& merge_name =
      ControlLoopName(opts.new_name(edge->dst()->name()));
  const string& switch_name =
      ControlLoopName(opts.new_name(edge->dst()->name()));
  const string& next_name = ControlLoopName(opts.new_name(edge->dst()->name()));

  // Add the nodes to the graph g.
  Node* enter = AddControlEnter(g, enter_name, device_name, frame_name,
                                parallel_iterations, &status);
  if (!status.ok()) return status;
  Node* merge = AddControlMerge(enter_name, next_name, g, merge_name,
                                device_name, &status);
  if (!status.ok()) return status;
  Node* switch_node = AddControlSwitch(merge, loop_cond, device_name,
                                       bopts.WithName(switch_name));
  if (!status.ok()) return status;
  Node* next =
      AddControlNext({switch_node, 1}, device_name, bopts.WithName(next_name));
  if (!status.ok()) return status;

  // Add control flow info for these new nodes:
  AddControlFlowInfo(enter, src, cf_info);
  AddControlFlowInfo(merge, src, cf_info);
  AddControlFlowInfo(switch_node, src, cf_info);
  AddControlFlowInfo(next, src, cf_info);

  // Add input edges for the newly created merge node:
  g->AddEdge(enter, 0, merge, 0);
  g->AddEdge(next, 0, merge, 1);

  loop->enter = enter;
  loop->merge = merge;
  loop->switch_node = switch_node;
  return Status::OK();
}

// Build memory and device type info for every node in the graph.
// TODO(yuanbyu): It might be simpler if we convert MemoryType to
// DeviceType for the inputs/outputs of each node.
Status BuildMemoryDeviceInfo(const Graph& g, GraphInfo* info) {
  MemoryTypeVector input_memory_types;
  MemoryTypeVector output_memory_types;

  info->device_types.resize(g.num_node_ids(), DEVICE_CPU);
  for (const Node* node : g.op_nodes()) {
    DeviceNameUtils::ParsedName parsed;
    if (!DeviceNameUtils::ParseFullName(node->assigned_device_name(),
                                        &parsed)) {
      return errors::Internal("Malformed assigned device '",
                              node->assigned_device_name(), "'");
    }

    TF_RETURN_IF_ERROR(MemoryTypesForNode(
        g.op_registry(), DeviceType(parsed.type), node->def(),
        &input_memory_types, &output_memory_types));

    int node_id = node->id();
    info->device_types[node_id] = DeviceType(parsed.type);
    for (size_t i = 0; i < input_memory_types.size(); ++i) {
      info->input_types[{node_id, i}] = input_memory_types[i];
    }
    for (size_t i = 0; i < output_memory_types.size(); ++i) {
      info->output_types[{node_id, i}] = output_memory_types[i];
    }
  }
  return Status::OK();
}

const Node* InputFrame(const Node* node,
                       const std::vector<ControlFlowInfo>& cf_info) {
  // An input is in the same frame as the node except for Enter nodes.
  // The input of Enter is in the parent frame of the Enter node.
  if (!node->IsEnter()) {
    return node;
  }
  return cf_info[node->id()].parent_frame;
}

const Node* OutputFrame(const Node* node,
                        const std::vector<ControlFlowInfo>& cf_info) {
  // An output is in the same frame as the node except for Exit nodes.
  // The output of Exit is in the parent frame of the Exit node.
  if (!node->IsExit()) {
    return node;
  }
  return cf_info[node->id()].parent_frame;
}

// Each participating device needs to decide a) if there is a next iteration,
// and b) if the loop terminates. We take the approach to encode this control
// flow logic in the dataflow graph. There are at least two possible encodings.
// In a completely decentralized encoding, the participants communicate peer
// to peer. The other encoding uses a frame leader (the participant who owns
// the pivot termination predicate) to broadcast the termination condition to
// all the participants. For now we take the latter because it is simpler.
//
// TODO(yuanbyu): The correctness of this construction is rather subtle. I got
// it wrong many times so it would be nice to write a proof to be sure.
Status AddControlFlow(const PartitionOptions& opts, Graph* g,
                      GraphInfo* g_info) {
  Status status;
  GraphDefBuilder::Options bopts(g, &status);
  std::vector<ControlFlowInfo>& cf_info = g_info->cf_info;

  // Build the control flow info for every node.
  status = BuildControlFlowInfo(g, &cf_info);
  if (!status.ok()) return status;

  OptimizeControlFlowColocation(g);

  // The map from frames to their LoopCond nodes.
  std::unordered_map<string, Node*> frame_cond_map;
  int num_node_ids = g->num_node_ids();
  for (int i = 0; i < num_node_ids; ++i) {
    Node* node = g->FindNodeId(i);
    if (node == nullptr) continue;

    if (IsLoopCond(node)) {
      const string& frame_name = cf_info[node->id()].frame_name;
      DCHECK(!frame_name.empty());
      frame_cond_map[frame_name] = node;
    }
  }

  // Add all control loops for cross-device frames.
  // A control loop is added only when there is a cross-device edge in a
  // non-root frame. Nothing is added if there is no loops. We also don't
  // add anything for a frame that is completely local to a device. For
  // nested loops, we stack the control loops together by connecting
  // the merge of the outer loop to the enter of the inner loop.
  //
  // A map from <frame_name, device_name> to ControlLoop.
  std::unordered_map<string, ControlLoop> control_loops;
  int num_edge_ids = g->num_edge_ids();
  for (int i = 0; i < num_edge_ids; ++i) {
    const Edge* edge = g->FindEdgeId(i);
    if (edge == nullptr) continue;

    const Node* src = edge->src();
    const Node* dst = edge->dst();
    // Skip Sink/Source nodes.
    if (!src->IsOp() || !dst->IsOp()) continue;

    const string& src_device = src->assigned_device_name();
    const string& dst_device = dst->assigned_device_name();
    // Skip local edges.
    if (src_device == dst_device) continue;

    const Node* src_frame = OutputFrame(src, cf_info);
    const Node* dst_frame = InputFrame(dst, cf_info);
    const string& src_frame_name = cf_info[src_frame->id()].frame_name;
    const string& dst_frame_name = cf_info[dst_frame->id()].frame_name;
    // Skip if src and dst are not in the same frame.
    if (src_frame_name.empty() || src_frame_name != dst_frame_name) {
      continue;
    }

    // Add the control loop. Start by adding the control loop for the
    // current frame if needed, and recursively adding the control loop
    // for its outer frame when nested.
    ControlLoop child_loop;
    while (true) {
      const string& curr_frame_name = cf_info[src_frame->id()].frame_name;
      if (curr_frame_name.empty()) {
        // We have reached the root frame.
        if (child_loop.merge != nullptr) {
          const string& node_name = opts.new_name(edge->dst()->name());
          const string& device_name = edge->dst()->assigned_device_name();
          Node* const_node =
              AddControlConst(device_name, bopts.WithName(node_name));
          if (!status.ok()) return status;
          AddControlFlowInfo(const_node, src_frame, &cf_info);
          g->AddEdge(const_node, 0, child_loop.enter, 0);
        }
        break;
      }

      const string& cl_key = strings::StrCat(curr_frame_name, "$$", dst_device);
      auto it = control_loops.find(cl_key);
      if (it != control_loops.end()) {
        if (child_loop.enter != nullptr) {
          g->AddEdge(it->second.merge, 0, child_loop.enter, 0);
        }
        break;
      }

      // Get the frame's LoopCond.
      auto cond_it = frame_cond_map.find(curr_frame_name);
      if (cond_it == frame_cond_map.end()) {
        return errors::InvalidArgument(
            "A cross-device loop must have a pivot predicate: ",
            curr_frame_name);
      }
      Node* loop_cond = cond_it->second;

      // Add the control loop.
      ControlLoop curr_loop;
      status = AddControlLoop(opts, g, src_frame, edge, loop_cond, &cf_info,
                              &curr_loop);
      if (!status.ok()) return status;
      control_loops[cl_key] = curr_loop;

      if (child_loop.enter != nullptr) {
        // Connect the merge of the outer loop to the enter of the inner.
        g->AddEdge(curr_loop.merge, 0, child_loop.enter, 0);
      }
      src_frame = cf_info[src_frame->id()].parent_frame;
      child_loop = curr_loop;
    }
  }

  // For a cross-device edge, on the dst device, add a control edge
  // from the merge node of the control loop to dst. If a send/recv is
  // introduced for this edge in future partitioning, we delete this
  // control edge and add a new control edge from the merge to the recv.
  num_edge_ids = g->num_edge_ids();
  for (int i = 0; i < num_edge_ids; ++i) {
    const Edge* edge = g->FindEdgeId(i);
    if (edge == nullptr) continue;

    const Node* src = edge->src();
    Node* dst = edge->dst();
    // Skip Sink/Source nodes.
    if (!src->IsOp() || !dst->IsOp()) continue;

    const string& src_device = src->assigned_device_name();
    const string& dst_device = dst->assigned_device_name();
    if (src_device != dst_device) {
      const Node* src_frame = OutputFrame(src, cf_info);
      const Node* dst_frame = InputFrame(dst, cf_info);
      const string& src_frame_name = cf_info[src_frame->id()].frame_name;
      const string& dst_frame_name = cf_info[dst_frame->id()].frame_name;
      if (!src_frame_name.empty() && src_frame_name == dst_frame_name) {
        const string& cl_key =
            strings::StrCat(dst_frame_name, "$$", dst_device);
        ControlLoop loop = control_loops[cl_key];
        DCHECK(loop.enter != nullptr);
        g->AddControlEdge(loop.merge, dst);
      }
    }
  }
  return Status::OK();
}

struct PriorityTopoSortNode {
  PriorityTopoSortNode(const NodeDef* n, int64 st) : node(n), start_time(st) {}

  const NodeDef* node;
  int64 start_time;
};

struct PriorityTopoSortNodeGreater {
  bool operator()(const PriorityTopoSortNode& left,
                  const PriorityTopoSortNode& right) {
    return left.start_time > right.start_time;
  }
};

}  // namespace

// Returns in <nodes> the nodes that should participate in epoch-based recv
// scheduling, along with their times; <nodes> is ordered by increasing
// start_time. Returns in <node_to_start_time_out> the timing for all nodes,
// even those not in <nodes>.
//
// Comparing to sorting on the node's start time only, this also processes the
// nodes in dependency order, and updates start times to ensure a node's
// start_time > the start time for all dependencies.
//
// Note that graph_partition_test.cc accesses this function for testing, even
// though it's not declared in the header.
Status TopologicalSortNodesWithTimePriority(
    const GraphDef* gdef, std::vector<std::pair<const NodeDef*, int64>>* nodes,
    std::unordered_map<const NodeDef*, int64>* node_to_start_time_out) {
  // Queue of nodes to process; lowest start time is returned first.
  std::priority_queue<PriorityTopoSortNode, std::vector<PriorityTopoSortNode>,
                      PriorityTopoSortNodeGreater>
      q;
  std::unordered_map<const NodeDef*, int64> node_to_start_time;
  auto enqueue = [&q, &node_to_start_time](const NodeDef* node) {
    const int64 start_time = node_to_start_time[node];
    q.emplace(node, start_time);
  };

  // Build initial structures, initial contents of queue.
  std::unordered_map<string, std::vector<const NodeDef*>> node_to_output_nodes;
  std::unordered_map<const NodeDef*, int> inputs_needed;
  for (int n = 0; n < gdef->node_size(); ++n) {
    const NodeDef* ndef = &gdef->node(n);
    for (int i = 0; i < ndef->input_size(); ++i) {
      node_to_output_nodes[ParseTensorName(ndef->input(i)).first.ToString()]
          .push_back(ndef);
    }
    int64 start_time;
    TF_RETURN_IF_ERROR(GetNodeAttr(*ndef, "_start_time", &start_time));
    node_to_start_time[ndef] = start_time;
    inputs_needed[ndef] = ndef->input_size();
    if (ndef->input_size() == 0) {
      enqueue(ndef);
    }
  }

  // Determine which merge nodes are parts of loops; these
  // need to happen in the traversal after all non-NextIteration inputs
  // are run.
  for (int n = 0; n < gdef->node_size(); ++n) {
    const NodeDef* ndef = &gdef->node(n);
    if (IsNextIteration(*ndef)) {
      for (const NodeDef* n : node_to_output_nodes[ndef->name()]) {
        if (IsMerge(*n)) {
          // n is a merge that is part of a loop structure.
          // It doesn't need to wait for this NextIteration loop
          // when doing the traversal.
          --inputs_needed[n];
        }
      }
    }
  }

  // Traverse.
  std::vector<std::pair<const NodeDef*, int64>> start_times;
  start_times.reserve(gdef->node_size());
  while (!q.empty()) {
    PriorityTopoSortNode cur = q.top();
    q.pop();

    start_times.emplace_back(cur.node, cur.start_time);

    for (const NodeDef* n : node_to_output_nodes[cur.node->name()]) {
      auto& output_start_time = node_to_start_time[n];
      if (output_start_time <= cur.start_time) {
        output_start_time = cur.start_time + 1;
      }
      if (--inputs_needed[n] == 0) {
        enqueue(n);
      }
    }
  }

  // Done.
  nodes->swap(start_times);
  node_to_start_time_out->swap(node_to_start_time);
  return Status::OK();
}

Status AddControlEdges(const PartitionOptions& opts,
                       std::unordered_map<string, GraphDef>* partitions) {
  Status status;
  // TODO(yuanbyu): Very naive for now. To be improved.
  const int num_epochs = 100;
  const int prefetch = 6;

  for (auto& part : *partitions) {
    GraphDef* gdef = &part.second;
    std::vector<std::pair<const NodeDef*, int64>> start_times;
    std::unordered_map<const NodeDef*, int64> node_to_start_time;
    status = TopologicalSortNodesWithTimePriority(gdef, &start_times,
                                                  &node_to_start_time);
    if (!status.ok()) {
      return status;
    }

    // Add a dummy node for every epoch, and add a control edge from the
    // "last" node in the preceding epoch to the dummy node.
    string device_name = gdef->node(0).device();
    int64 makespan = start_times.back().second;
    int64 resolution = (makespan / num_epochs) + 1;

    int i = 0;
    int j = 0;
    std::vector<NodeDef*> dummys;
    while (i < num_epochs && static_cast<size_t>(j) < start_times.size()) {
      if (i * resolution > start_times[j].second) {
        j++;
      } else {
        NodeDef* dummy = AddControlTrigger(opts, gdef, device_name, i,
                                           i * resolution, &status);
        if (!status.ok()) {
          return status;
        }
        dummys.push_back(dummy);
        if (j > 0) {
          string src_name = start_times[j - 1].first->name();
          AddInput(dummy, src_name, Graph::kControlSlot);
        }
        i++;
      }
    }

    // Finally, add the control edges to recvs.
    for (int n = 0; n < gdef->node_size(); ++n) {
      NodeDef* ndef = gdef->mutable_node(n);
      if (ndef->op() == "_Recv") {
        const int64 start_time = node_to_start_time[ndef];
        const int recv_epoch = start_time / resolution;
        if (recv_epoch >= prefetch) {
          NodeDef* dummy = dummys[recv_epoch - prefetch];
          AddInput(ndef, dummy->name(), Graph::kControlSlot);
        }
      }
    }
  }
  return Status::OK();
}

// If 'ndef' is a Send or Recv, fills its attr send_device_incarnation
// if possible.
void SetIncarnation(const PartitionOptions& opts, NodeDef* ndef) {
  StringPiece op(ndef->op());
  if (op != "_Send" && op != "_Recv") {
    // Not related to send/recv.
    return;
  }
  string send_device;
  if (!GetNodeAttr(*ndef, "send_device", &send_device).ok()) {
    // No known send_device. The runtime will detect it later.
    return;
  }
  int64 incarnation = PartitionOptions::kIllegalIncarnation;
  if (!GetNodeAttr(*ndef, "send_device_incarnation", &incarnation).ok() ||
      (incarnation == PartitionOptions::kIllegalIncarnation)) {
    incarnation = opts.get_incarnation(send_device);
    SetAttrValue(incarnation,
                 &((*ndef->mutable_attr())["send_device_incarnation"]));
  }
}

// Sets attribute send_device_incarnation of all Send/Recv nodes in
// 'gdef', if possible.
void SetIncarnation(const PartitionOptions& opts, GraphDef* gdef) {
  for (NodeDef& ndef : *gdef->mutable_node()) {
    SetIncarnation(opts, &ndef);
  }
  for (FunctionDef& fdef : *gdef->mutable_library()->mutable_function()) {
    for (NodeDef& ndef : *fdef.mutable_node_def()) {
      SetIncarnation(opts, &ndef);
    }
  }
}


/**************************************************************************************************/

typedef struct state_machine_node {
  Node* node;
  std::vector<string> inputs;
} StateMachineNode;

typedef struct state_machine_graph {
  std::unordered_map<string, StateMachineNode*> nodes;
  std::set<string> partitions;
  Node* merge;
} StateMachineGraph;

typedef struct state_machine {
  // A map from frame_names to  StateMachineGraphs representing a general dynamic
  // state machine that we update every time a function gets called, and helps us
  // gradually build the state machines of the partitions
  std::unordered_map<string, StateMachineGraph*> state_machine_graphs;
  // state_machine_parents is the 'spine' of the graph,
  // containing only control flow nodes
  std::vector<Node*> state_machine_parents;
  
  std::unordered_map<string, GraphDef*> partitions_state_machines;
  std::unordered_map<Node*, Node*> switches_info;
  string leader_partition;

  // Maps device names to smaller strings .
  std::unordered_map<string, string> device_names_map;

} StateMachine;

typedef struct func_info {
  // A map from <frame_name> to the num of function's arguments
  std::unordered_map<string, int> funcInputs;
  // Εach vector<Node*> below operates as a barrier,
  // we don't call CallingFunction(..) before we gather
  // all function's arguments/calls first
  std::unordered_map<int, std::vector<Node*>*> funcCalls;
} FuncInfo;

// Adds root nodes into ready_nodes queue and sets ready_inputs appropriately
void PreprocessGraph(std::unordered_map<const Node*, int> &ready_inputs, Graph* g,
                     std::deque<Node*> &ready_nodes) {

  std::unordered_map<const Node*, std::set<int>> returning_nodes;

  for (Node* node : g->nodes()) {

    if (node->in_edges().empty()) {
      ready_nodes.push_back(node);
    }
    bool recursion_merge = 0;
    if (IsMerge(node)) {
      ready_inputs[node] = 0;
      for (const Edge* in_edge : node->in_edges()) {

        Node* in = in_edge->src();
        // if (IsNextIteration(*output_map.GetNode(input))) {
        //   ready_inputs[node]++;
        // }
        if (IsCall(in)) {
          ready_inputs[node]++;
          recursion_merge = 1;
        }
      }
      if (recursion_merge) {
        ready_inputs[node]--;
        recursion_merge = 0;
      }

    } else if (IsReturn(node)) {

      for (const Edge* in_edge : node->in_edges()) {
        Node* in = in_edge->src();

        if (!in_edge->IsControlEdge()) {
          int call_id;
          GetNodeAttr(node->attrs(), "call_id", &call_id);
          returning_nodes[in].emplace(call_id);
        }
      }
      ready_inputs[node] = 0;

    } else {
      ready_inputs[node] = 0;
    }
  }

  for (const auto& retnode : returning_nodes) {
    if (retnode.second.size() > 1) {
      // Detected Cycle
      ready_inputs[retnode.first]++;
    }
  }
}

string GetDeviceMappedName(StateMachine &state_machine, string device_name) {

  string pname;
  std::unordered_map<string, string>& device_map = state_machine.device_names_map;

  auto it = device_map.find(device_name);
  if (it == device_map.end()) {
    pname = strings::StrCat("_p", device_map.size()+1);
    device_map[device_name] = pname;
  } else pname = it;

  return pname;
}

bool IsCallSuccessor(Node* node) {

  for (const Edge* in_edge : node->in_edges()) {
    Node* src = in_edge->src();
    if (IsCall(src) && !in_edge->IsControlEdge())
      return true;
  }
  return false;
}

std::vector<Node*>* GetOrCreateCalls(int call_id, std::unordered_map<int,std::vector<Node*>*> &funcCalls) {
  //todo : deallocate
    auto slot = &funcCalls[call_id];
    if (*slot == nullptr)
      *slot = new std::vector<Node*>;
    return *slot;
}

GraphDef* GetOrCreateGraphDefs(string partition, StateMachine& state_machine) {

  //todo : deallocate
  auto slot = &state_machine.partitions_state_machines[partition];
  if (*slot == nullptr)
    *slot = new GraphDef;
  return *slot;
}

// For one if-else construction there are more than one Switch nodes guarding all the inputs
// that are needed inside the branches but live outside of them. We need to collect all the Switch
// nodes that correspond to one if-else construction and treat them as one in the state machines
// switches_info: Every switch node maps to the original switch that we "ll take into account
void CollectSwitches(Graph* g, std::unordered_map<Node*, Node*>& switches_info) {

  // <predicate_node, switch_node>
  std::unordered_map<Node*, Node*> pred_switch;

  for (Node *node : g->nodes()) {

    if (IsSwitch(node)) {

      for (const Edge *in_edge : node->in_edges()) {

        int port = in_edge->dst_input();

        // A sloppy way to determine if this is the predicate input
        if (!in_edge->IsControlEdge() && port == 1) {

          Node *predicate = in_edge->src();

          while (IsIdentity(predicate)) {
            for (const Edge *inEdge : predicate->in_edges()) {
              if (!inEdge->IsControlEdge()) {
                predicate = inEdge->src();
                break;
              }
            }
          }

          // We 've got the real predicate
          Node *switchNode;
          if (pred_switch.find(predicate) == pred_switch.end()) {
            // Original switch
            pred_switch[predicate] = node;
            switchNode = node;
          } else {
            // "Synonym" switch
            switchNode = pred_switch[predicate];
          }

          switches_info[node] = switchNode;

          break;
        }
      }
      printf("Switch : %s -> %s\n", node->name().c_str(), switches_info[node]->name().c_str());
    }
  }

  printf("\n\n\n");
}

NodeDef* FindNodeInGraphDef(GraphDef& graphDef, string node_name) {

  for (NodeDef& nodeDef : *graphDef.mutable_node()) {
    if (nodeDef.name() == node_name)
      return &nodeDef;
  }
  return nullptr;
}

void ConnectMergeToNode(GraphDef& graphDef, string merge_name, string node_name, string partition_name) {

  // Every nodedef's name inside device's state machine, will have <partition_name> as suffix
  // so we can safely infer Merge's name and add is as control input to the node
  string merge_input = strings::StrCat(merge_name, "_", partition_name, "^");

  NodeDef* node = FindNodeInGraphDef(graphDef, node_name);
  *node->add_input() = merge_input;
}

void AddPartitionStateMachine(StateMachineGraph* sm_graph, GraphDef& main_graphDef, string partition) {

}


void AddNodeToStateMachine(StateMachine& state_machine, string frame, Node* node, bool cycle) {

  StateMachineGraph* smg = state_machine.state_machine_graphs[frame];
  StateMachineNode* smn = new StateMachineNode;

  smn->node = node;

  string input;
  Node* parent = state_machine.state_machine_parents[node->id()];
  // If parent is null add a dummy constant as input
  (parent != nullptr) ? input = parent->name() : input = strings::StrCat("Dummy_", node->name());
  smn->inputs.push_back(input);

  smg->nodes[node->name()] = smn;

  // If cycle is true, node is a recursive call, that needs to be added as
  // input to the corresponding Merge node
  if (cycle == true) {
    // We traverse graph the way topological sort does, so we will never
    // meet a recursive call node before its corresponding Merge
    StateMachineNode* merge = smg->nodes[smg->merge->name()];
    merge->inputs.push_back(node->name());
  }
}

void CallingFunction(Graph* graph, GraphDef& main_graphDef, StateMachine& state_machine, FuncInfo& funcInfo,
                    string function_frame_name, int function_call_id,
                    std::unordered_map<const Node*, int>& ready_inputs,
                    std::deque<Node*>& prev_ready_nodes) {

  Node *merge, *call;
  std::deque<Node*> ready_nodes;

  std::vector<Node*>* calls = funcInfo.funcCalls[function_call_id];
  for (int i=0; i < calls->size(); ++i) {
    ready_nodes.push_back((*calls)[i]);
  }
  call = (*calls)[0];

  // We add only one Call node for all possible function's args in the state machine
  AddNodeToStateMachine(state_machine, function_frame_name, call, false);

  std::vector<Node*>& state_machine_parents = state_machine.state_machine_parents;
  StateMachineGraph* sm_graph = state_machine.state_machine_graphs[function_frame_name];

  // Call's successor (the non control output) will be either
  // a Merge node (in case of recursion) or an Identity node.
  // Either way we add that successor to the state machine, too. (todo: optimize)
  // Same as above, we add only one Merge node instead of one per function's arg
  for (const Edge* out_edge : call->out_edges()) {
    if (!out_edge->IsControlEdge()) {
      merge = out_edge->dst();
      state_machine_parents[merge->id()] = call;
      AddNodeToStateMachine(state_machine, function_frame_name, merge, false);
      sm_graph->merge = merge;
      break;
    }
  }

  while (!ready_nodes.empty()) {

    Node* ready_node = ready_nodes.front();
    ready_nodes.pop_front();

    printf("Pop From Queue: %s\n", ready_node->name().c_str());
    Node* parent = state_machine_parents[ready_node->id()];

    // The ops below need to update the parent
    if (IsCall(ready_node)) {
      parent = call;
    } else if (IsCallSuccessor(ready_node)) {
      parent = merge;
    } else if (IsSwitch(ready_node)) {
      Node *sw = state_machine.switches_info[ready_node];
      if (sw == ready_node)
        AddNodeToStateMachine(state_machine, function_frame_name, ready_node, false);
      parent = sw;
    } else if (IsMerge(ready_node)) {
      // Control Flow (regular) Merge has a corresponding Switch node
      // Parent gets the value of that switch node's parent
      parent = state_machine_parents[parent->id()];
    } else if (IsReturn(ready_node)) {
      // Return needs to propagate its corresponding Call's parent to all its successors
      for (const Edge* in_edge : ready_node->in_edges()) {
        if (in_edge->IsControlEdge()) {
          Node* call_node = in_edge->src();
          parent = state_machine_parents[call_node->id()];
          break;
        }
      }
      int call_id;
      GetNodeAttr(ready_node->attrs(), "call_id", &call_id);
      // If not a 'recursive' return
      if (call_id == function_call_id) {
        // Add the successors of Return node to prev_ready_nodes queue
        prev_ready_nodes.push_back(ready_node);
        // Set the parent value of the only actual output of return
        for (const Edge* out_edge : ready_node->out_edges()) {
          Node* out = out_edge->dst();
          state_machine_parents[out->id()] = parent;
          break;
        }
        continue;
      }
    }

    // Process ready_node's outputs
    for (const Edge* out_edge : ready_node->out_edges()) {
      Node* out = out_edge->dst();

      ready_inputs[out]++;

      // For a cross-device edge, on the dst device, add a control edge
      // from the merge node of the state machine to dst. If a send/recv is
      // introduced for this edge in future partitioning, we delete this
      // control edge and add a new control edge from the merge to the recv.
      const string& src_device = ready_node->assigned_device_name();
      const string& dst_device = out->assigned_device_name();
      if (src_device != dst_device) {
        ConnectMergeToNode(main_graphDef, merge->name(), out->name(), dst_device);
      }

      if (ready_inputs[out] == out->in_edges().size()) {

        printf("        Out Node: %s\n", out->name().c_str());
        // Set node's parent
        state_machine_parents[out->id()] = parent;

        std::unordered_map<string, StateMachineGraph*>& sm_graphs = state_machine.state_machine_graphs;

        if (IsCall(out)) {


          string frame_name;
          GetNodeAttr(out->attrs(), "frame_name", &frame_name);
          int call_id;
          GetNodeAttr(out->attrs(), "call_id", &call_id);

          std::vector<Node*>* calls = GetOrCreateCalls(call_id, funcInfo.funcCalls);
          calls->push_back(out);

          printf("        Call Node: %s, FrameName: %s, Call_Id: %d\n", out->name().c_str(), frame_name.c_str(), call_id);
          if (funcInfo.funcInputs[frame_name] == calls->size()) {

            // We gathered all function's inputs
            auto it = sm_graphs.find(frame_name);
            if (it == sm_graphs.end()) {
              printf("        Gathered all Calls:        CallingFunction recursively!!\n\n");
              StateMachineGraph partial_state_machine;
              sm_graphs.emplace(frame_name, &partial_state_machine);
              CallingFunction(graph, main_graphDef, state_machine, funcInfo, frame_name, call_id, ready_inputs, ready_nodes);
              //todo : deallocate
              sm_graphs.erase(frame_name);
              printf("\nGathered all Returns:      EXIT!!\n\n");

            }

            else {

              printf("        Gathered all Calls:        Recursive Calls\n");

              // Recursive Call (either to the same function or another one (mutual recursion)
              AddNodeToStateMachine(state_machine, frame_name, (*calls)[0], true);
              // Add the recursive call nodes to ready_nodes
              for (int i=0; i < calls->size(); ++i) {
                ready_nodes.push_back((*calls)[i]);
              }
              // call = (*calls)[0];
              StateMachineGraph* smg = sm_graphs[frame_name];
              for (string device_name : smg->partitions) {
                for (auto& it : sm_graphs)
                  it.second->partitions.emplace(dst_device);
              }
            } // end of recursive call
          } // end of -gathered all calls
        }

        // Not a call
        else {
          printf("        Partition: '%s' to all SMGs: \n", dst_device.c_str());


          string dst_device = out->assigned_device_name();
          // Add partition to all StateMachineGraphs' partitions
          for (auto& it : sm_graphs) {
            it.second->partitions.emplace(dst_device);
            printf("                    Graph: %s\n", it.first.c_str());
          }
          ready_nodes.push_back(out);
        }

      } // end of all inputs ready

    } // end of for all edges

  } // end of while ready_nodes

  // Exiting function : We add this level's state machine, to all
  // the partitions that need it - Only at this point we know that this
  // level's state machine will be complete

/*******************************************************************************/
  //print
  for (const auto& it: state_machine.state_machine_graphs) {
    string frame = it.first;
    StateMachineGraph* sm_graph = it.second;
    printf("\n\nsm_graph :'%s'\n", frame.c_str());
    for(const auto& itt : sm_graph->nodes) {
      string node_name = itt.first;
      StateMachineNode* sm_node = itt.second;
      printf("Node: %s, \nInputs: \n", sm_node->node->name().c_str());
      for(const auto& input : sm_node->inputs) {
        printf("        %s\n", input.c_str());
      }
    }
  }
/*******************************************************************************/

  for (const auto& device_name: sm_graph->partitions) {
    // Leader Partition already has its state machine
    if (device_name != state_machine.leader_partition)
      AddPartitionStateMachine(sm_graph, main_graphDef, device_name);
  }
}

Status AddFunctionStateMachines(const PartitionOptions& opts,
                            Graph* g, GraphInfo* g_info) {

  Status status;
  GraphDefBuilder::Options bopts(g, &status);

  FuncInfo funcInfo;
  int nodes_num = g->num_node_ids();

  const FunctionDefLibrary& fdef = opts.flib_def->ToProto();
  for (const FunctionDef& func : fdef.function()) {

      int num_inputs = func.signature().input_arg_size();

      string name = func.signature().name();
      funcInfo.funcInputs[name] = num_inputs;
      printf("func sign= %s, inputs : %d\n", name.c_str(), num_inputs);
  }

  StateMachine state_machine;
  state_machine.state_machine_parents.resize(nodes_num);

  CollectSwitches(g, state_machine.switches_info);

  // Add all state machines for cross-device frames.
  // A state machine is added only when there is a cross-device edge in a
  // non-root frame.

  // Visit nodes the way topological sort does
  std::deque<Node*> ready_nodes;
  std::unordered_map<const Node*, int> ready_inputs;

  PreprocessGraph(ready_inputs, g, ready_nodes);

  // We convert graph to its equivalent graph_def, cause it's easier
  // to extend it with the GraphDef state machines of partitions
  // todo: traverse and manipulate the graph_def instead of graph
  GraphDef main_graphDef;
  g->ToGraphDef(&main_graphDef);

  while (!ready_nodes.empty()) {
    Node* ready_node = ready_nodes.front();
    ready_nodes.pop_front();
    printf("\nPop From Queue: %s\n\n", ready_node->name().c_str());

    for (const Edge* out_edge : ready_node->out_edges()) {
      Node* out = out_edge->dst();

      ready_inputs[out]++;

      if (ready_inputs[out] == out->in_edges().size()) {

        printf("        Out Node: %s\n", out->name().c_str());
        if (IsCall(out)) {
          string frame_name;
          GetNodeAttr(out->attrs(), "frame_name", &frame_name);
          int call_id;
          GetNodeAttr(out->attrs(), "call_id", &call_id);

          std::vector<Node*>* calls = GetOrCreateCalls(call_id, funcInfo.funcCalls);
          calls->push_back(out);

          printf("        Call Node: %s, FrameName: %s, Call_Id: %d\n", out->name().c_str(), frame_name.c_str(), call_id);

          if (funcInfo.funcInputs[frame_name] == calls->size()) {

            printf("        Gathered all Calls:        CallingFunction recursively!!\n\n");
            state_machine.leader_partition = out->assigned_device_name();
            // We gathered all function's inputs
            StateMachineGraph partial_state_machine;
            state_machine.state_machine_graphs.emplace(frame_name, &partial_state_machine);
            CallingFunction(g, main_graphDef, state_machine, funcInfo, frame_name, call_id, ready_inputs, ready_nodes);
            //todo : deallocate
            state_machine.state_machine_graphs.erase(frame_name);
            printf("        Gathered all Returns:      EXIT!!\n\n");

          }
        }
        else
          ready_nodes.push_back(out);
      }
    }
  }

  // todo: HERE: Extend main_graphdef with graph def state machines, then convert to Graph

  return Status::OK();
}



/**************************************************************************************************/


Status Partition(const PartitionOptions& opts, Graph* g,
                 std::unordered_map<string, GraphDef>* partitions) {
  Status status;
  partitions->clear();

  GraphInfo g_info;
  if (!opts.control_flow_added) {
    // Add the "code" for distributed execution of control flow. Code is
    // added only for the frames that are placed on multiple devices. The
    // new graph is an equivalent transformation of the original graph and
    // has the property that it can be subsequently partitioned arbitrarily
    // (down to the level of individual device) for distributed execution.
    printf("Trying to add state machines\n");
    status =  AddFunctionStateMachines(opts, g, &g_info);
    // status =  AddControlFlow(opts, g, &g_info);
    if (!status.ok()) return status;
  }
  printf("added state machines\n");
  exit(1);


  // At this point, all the graph mutations have been done. Build memory
  // and device type info for every node and edge in the graph.
  status = BuildMemoryDeviceInfo(*g, &g_info);
  if (!status.ok()) return status;

  string dstp;
  std::vector<const Edge*> inputs;
  DupRecvTable dup_recv(3);
  // For a node dst, 'ref_recvs' remembers the recvs introduced by a ref
  // edge to dst. 'ref_control_inputs' remembers the inputs by a non-ref
  // edge to dst. We will add a control edge for every pair in
  // (ref_recvs x ref_control_inputs).
  std::vector<NodeDef*> ref_recvs;
  std::vector<string> ref_control_inputs;

  int32 num_data = 0;
  int32 num_control = 0;
  for (const Node* dst : g->op_nodes()) {
    dstp = opts.node_to_loc(dst);
    GraphDef* dst_graph = &(*partitions)[dstp];
    NodeDef* dst_def = dst_graph->add_node();
    *dst_def = dst->def();
    dst_def->set_device(dst->assigned_device_name());
    dst_def->clear_input();  // Inputs are filled below
    if (opts.need_to_record_start_times) {
      int64 start_time;
      status = GetNodeAttr(*dst_def, "_start_time", &start_time);
      if (errors::IsNotFound(status)) {
        start_time = opts.start_times[dst->id()].value();
        AddNodeAttr("_start_time", start_time, dst_def);
      } else if (!status.ok()) {
        return status;
      }
    }

    // Arrange the incoming edges to dst so that input[i] holds the
    // input flowing into slot numbered i. Trailing entries in input[]
    // hold control edges.
    inputs.clear();
    inputs.resize(dst->num_inputs(), nullptr);
    ref_recvs.clear();
    ref_control_inputs.clear();
    const Edge* control_flow_edge = nullptr;
    int32 num_control_flow_edges = 0;
    int32 num_input_edges = 0;
    for (const Edge* edge : dst->in_edges()) {
      if (edge->IsControlEdge()) {
        if (IsMerge(edge->src()) && IsControlLoop(edge->src())) {
          // This is one of the control edges added for control flow. There
          // can be multiple such edges as the dest node may have multiple
          // remote inputs. We keep track of the number of such edges.
          control_flow_edge = edge;
          ++num_control_flow_edges;
        } else {
          inputs.push_back(edge);
        }
      } else {
        DCHECK(inputs[edge->dst_input()] == nullptr);
        inputs[edge->dst_input()] = edge;
        ++num_input_edges;
      }
    }

    if (num_input_edges != dst->num_inputs()) {
      return errors::InvalidArgument("Incomplete graph, missing ",
                                     (dst->num_inputs() - num_input_edges),
                                     " inputs for ", dst->name());
    }

    // Process in order so that all data edges are added as inputs to
    // dst in Edge::dst_input() order.
    for (const Edge* edge : inputs) {
      const Node* src = edge->src();
      if (!src->IsOp()) continue;  // Skip Sink/Source nodes.

      GraphDef* src_graph = &(*partitions)[opts.node_to_loc(src)];
      if (src_graph == dst_graph && !NeedSameDeviceSendRecv(edge, g_info)) {
        // Same partition and compatible memory types:
        AddInput(dst_def, src->name(), edge->src_output());
        if (edge->IsControlEdge() ||
            !IsRefType(src->output_type(edge->src_output()))) {
          ref_control_inputs.push_back(src->name());
        }
        continue;
      }

      int64 send_start_time = 0;
      int64 recv_start_time = 0;
      if (opts.scheduling_for_recvs) {
        status = GetNodeAttr(src->attrs(), "_start_time", &send_start_time);
        if (errors::IsNotFound(status) && opts.need_to_record_start_times) {
          send_start_time = opts.start_times[src->id()].value();
        } else if (!status.ok()) {
          return status;
        }

        status = GetNodeAttr(dst->attrs(), "_start_time", &recv_start_time);
        if (errors::IsNotFound(status) && opts.need_to_record_start_times) {
          recv_start_time = opts.start_times[dst->id()].value();
        } else if (!status.ok()) {
          return status;
        }
      }

      // Check whether there is already a send/recv pair transferring
      // the same tensor/control from the src to dst partition.
      const bool on_host = IsDstInputOnHost(edge, g_info);
      DupRecvKey key{src->id(), edge->src_output(), dst_graph, on_host};
      auto iter = dup_recv.find(key);
      if (iter != dup_recv.end()) {
        // We found one. Reuse the data/control transferred already.
        const string& recv_node_name = iter->second.recv->name();
        if (edge->IsControlEdge()) {
          AddInput(dst_def, recv_node_name, Graph::kControlSlot);
        } else {
          AddInput(dst_def, recv_node_name, 0);
        }
        ref_control_inputs.push_back(recv_node_name);

        // We want the start_time for the recv to be the smallest of the start
        // times of it's consumers. So we update this whenever we use a recv,
        // and write it out to the attribute at the end of the subroutine
        if (iter->second.start_time > recv_start_time) {
          iter->second.start_time = recv_start_time;
        }
        continue;
      }

      NodeDefBuilder::NodeOut send_from;
      if (edge->IsControlEdge()) {
        // Insert a dummy const node that will generate a tiny
        // data element to be sent from send to recv.
        VLOG(1) << "Send/Recv control: " << src->assigned_device_name() << "["
                << src->name() << "] -> " << dst->assigned_device_name() << "["
                << dst->name() << "]";
        NodeDef* dummy = AddDummyConst(opts, src_graph, edge, &status);
        if (!status.ok()) return status;
        // Set the start time for this dummy node.
        if (opts.scheduling_for_recvs) {
          AddNodeAttr("_start_time", send_start_time, dummy);
        }
        AddInput(dummy, src->name(), Graph::kControlSlot);
        send_from.Reset(dummy->name(), 0, DT_FLOAT);
      } else {
        send_from.Reset(src->name(), edge->src_output(), EdgeType(edge));
      }

      // Need to split edge by placing matching send/recv nodes on
      // the src/dst sides of the edge.
      NodeDef* send = AddSend(opts, g_info, src_graph, edge, send_from,
                              send_start_time, &status);
      if (!status.ok()) return status;

      NodeDef* real_recv = nullptr;
      NodeDef* recv =
              AddRecv(opts, g_info, dst_graph, edge, &real_recv, &status);
      if (!status.ok()) return status;

      // Fix up the control flow edge.
      // NOTE(yuanbyu): 'real_recv' must be the real recv node.
      if (src_graph == dst_graph) {
        // For same device send/recv, add a control edge from send to recv.
        // This prevents the asynchronous recv kernel from being scheduled
        // before the data is available.
        AddInput(real_recv, send->name(), Graph::kControlSlot);
      } else if (control_flow_edge != nullptr) {
        // Redirect control edge to the real recv since this is not a same
        // device send/recv.
        --num_control_flow_edges;
        AddInput(real_recv, control_flow_edge->src()->name(),
                 Graph::kControlSlot);
      }

      if (!edge->IsControlEdge() &&
          IsRefType(src->output_type(edge->src_output()))) {
        AddNodeAttr("_start_time", recv_start_time, recv);
        if (real_recv != recv) {
          AddNodeAttr("_start_time", recv_start_time, real_recv);
        }
        // If src is of ref type and the edge is not a control edge, dst has
        // read semantics and therefore we must control the recv.
        ref_recvs.push_back(real_recv);
      } else {
        // Memorize the send/recv pair, only if this is not a "ref" edge.
        // NOTE(yuanbyu): Collapsing ref edges requires extreme care so
        // for now we don't do it.
        dup_recv[key] = {recv, real_recv, recv_start_time};
        ref_control_inputs.push_back(recv->name());
      }

      if (edge->IsControlEdge()) {
        ++num_control;
        AddInput(dst_def, recv->name(), Graph::kControlSlot);
      } else {
        ++num_data;
        AddInput(dst_def, recv->name(), 0);
      }
    }

    // Add control edges from 'ref_control_inputs' to 'ref_recvs'.
    // NOTE(yuanbyu): Adding these control edges should not introduce
    // deadlocks. 'dst' has implicit "read" nodes that, when we split
    // across devices, are made explicit; Retargettig the dependencies
    // to 'dst' to those nodes would not introduce cycles if there isn't
    // one before the transformation.
    // NOTE(yuanbyu): This may impact performance because it defers the
    // execution of recvs until all the other inputs become available.
    AddReadControl(ref_recvs, ref_control_inputs);

    // Add back the control edges for control flow that are not used.
    if (control_flow_edge != nullptr) {
      for (int i = 0; i < num_control_flow_edges; ++i) {
        AddInput(dst_def, control_flow_edge->src()->name(),
                 Graph::kControlSlot);
      }
    }
  }

  const FunctionLibraryDefinition* flib_def = opts.flib_def;
  if (flib_def == nullptr) {
    flib_def = &g->flib_def();
  }

  // Set versions, function library and send/recv incarnation.
  for (auto& it : *partitions) {
    GraphDef* gdef = &it.second;
    *gdef->mutable_versions() = g->versions();
    *gdef->mutable_library() = flib_def->ToProto();

    // Traverse the graph to fill every send/recv op's incarnation
    // information.
    SetIncarnation(opts, gdef);
  }

  // Set the start times for recvs at the very end.
  if (opts.scheduling_for_recvs) {
    for (auto& it : dup_recv) {
      AddNodeAttr("_start_time", it.second.start_time, it.second.recv);
      if (it.second.real_recv != it.second.recv) {
        AddNodeAttr("_start_time", it.second.start_time, it.second.real_recv);
      }
    }
  }

  VLOG(1) << "Added send/recv: controls=" << num_control
          << ", data=" << num_data;
  return Status::OK();
}

}  // namespace tensorflow
