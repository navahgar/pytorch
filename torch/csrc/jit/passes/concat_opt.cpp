#include <torch/csrc/jit/passes/concat_opt.h>

#include <torch/csrc/jit/jit_log.h>

namespace torch {
namespace jit {

namespace {

using node_list = std::vector<Node*>;
std::unordered_map<Node*, node_list> list_appends;
std::unordered_map<Node*, Node*> concated_lists;

void HandleCat(Node* node) {
  // std::cout << "*** cat: " << *node;
  auto list_node = node->input(0)->node();
  auto appends = list_appends[list_node];
  // for (auto n : appends) {
  //   std::cout << "---- " << *n;
  // }
  if (appends.empty()) {
    auto it = concated_lists.find(list_node);
    if (it != concated_lists.end()) {
      // If this list has already been concated, use that result
      // instead of concating again.
      GRAPH_UPDATE("Replacing\n", *node, "with\n", *it->second);
      node->replaceAllUsesWith(it->second);
    } else {
      concated_lists[list_node] = node;
    }
    // std::cout << "*** End of Cat1:\n";
    return;
  }

  for (auto it1 = appends.rbegin(); it1 != appends.rend(); ++it1) {
    auto curr_append = *it1;
    auto it2 = concated_lists.find(curr_append);
    if (it2 != concated_lists.end()) {
      auto prev_concat_result = it2->second;
      // std::vector<Value*> new_list_values = {prev_concat_result->output()};
      // --it1;
      // while (it1 != appends.rend()) {
      //   std::cout << "getting from append list: " << **it1;
      //   new_list_values.push_back((*it1)->output());
      //   std::cout << "after pushing into vector\n";
      //   --it1;
      //   std::cout << "end = " << (it1 == appends.rend()) << std::endl;
      //   std::cout << "end of iteration\n";
      // }
      std::vector<Value*> new_list_values;
      for (auto it3 = appends.rbegin(); it3 != appends.rend(); ++it3) {
        if (it3 == it1)
          break;
        new_list_values.push_back((*it3)->input(1));
      }
      new_list_values.push_back(prev_concat_result->output());
      std::reverse(new_list_values.begin(), new_list_values.end());
      // for (auto k : new_list_values) {
      //   std::cout << "-- new_list_value: " << *k->node();
      // }
      // std::cout << "After getting from append list\n";
      auto new_list_node =
          node->owningGraph()->createList(TensorType::get(), new_list_values);
      GRAPH_UPDATE("Inserting\n", *new_list_node, "before\n", *node);
      new_list_node->insertBefore(node);
      GRAPH_UPDATE(
          "Replacing input 0 of\n",
          *node,
          "with\n",
          *new_list_node->output()->node());
      node->replaceInput(0, new_list_node->output());
      break;
    }
  }
  concated_lists[appends.back()] = node;
  // std::cout << "*** End of Cat2:\n";
}

void OptimizeBlock(Block* block) {
  // auto handle_cat = [](Node* node) {
  //   std::cout << "***** cat: " << *node << std::endl;
  //   std::cout << "*****--- input0: " << node->input(0)->node() << std::endl;
  //   std::cout << "*****--- input0: " << *node->input(0)->node() << std::endl;
  //   // std::cout << "*****--- input0 size = " <<
  //   node->input(0)->node()->inputs().size() << std::endl;
  // };
  for (auto node : block->nodes()) {
    switch (node->kind()) {
      case aten::cat:
        HandleCat(node);
        break;
      case aten::append: {
        // std::cout << "*** append: " << *node;
        // std::cout << "****-- input0: " << node->input(0)->node() <<
        // std::endl;
        auto list_node = node->input(0)->node();
        if (!list_appends.count(list_node)) {
          throw std::runtime_error(
              "List append found without a list construct");
        }
        list_appends[list_node].push_back(node);
        break;
      }
      case prim::ListConstruct:
        // std::cout << "*** ListConstruct: " << *node;
        // std::cout << "****-- result: " << node << std::endl;
        if (list_appends.count(node)) {
          throw std::runtime_error("Unexpected list construct");
        }
        list_appends[node] = {};
        break;
    }
    if (node->kind() == aten::append) {
      // std::cout << "*** append: " << *node << std::endl;
      // std::cout << "****-- input0: " << node->input(0)->node() << std::endl;
      // std::cout << "****-- input0: " << *node->input(0)->node() << std::endl;
    }
    if (!node->blocks().empty()) {
      for (auto block : node->blocks()) {
        OptimizeBlock(block);
      }
    }
  }
}

} // namespace

void OptimizeConcat(const std::shared_ptr<Graph>& graph) {
  GRAPH_DUMP("Before ConcatOpt", graph);
  OptimizeBlock(graph->block());
  GRAPH_DUMP("After ConcatOpt", graph);
}

} // namespace jit
} // namespace torch
