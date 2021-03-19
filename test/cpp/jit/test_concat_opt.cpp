#include <gtest/gtest.h>

#include <torch/csrc/jit/ir/irparser.h>
#include <torch/csrc/jit/passes/concat_opt.h>
#include <torch/csrc/jit/testing/file_check.h>

namespace torch {
namespace jit {

TEST(OptimizeConcatTest, Simple) {
  auto graph = std::make_shared<Graph>();

  const std::string input =
      R"IR(
        graph(%0: Tensor,
              %1: Tensor,
              %2: Tensor,
              %3: Tensor,
              %4: Tensor):
          %5 : int = prim::Constant[value=1]()
          %features : Tensor[] = prim::ListConstruct(%0)
          %concat.1 : Tensor = aten::cat(%features, %5)
          %6 : Tensor[] = aten::append(%features, %1)
          %concat.2 : Tensor = aten::cat(%features, %5)
          %7 : Tensor[] = aten::append(%features, %2)
          %concat.3 : Tensor = aten::cat(%features, %5)
          %8 : Tensor[] = aten::append(%features, %3)
          %concat.4 : Tensor = aten::cat(%features, %5)
          %9 : Tensor[] = aten::append(%features, %4)
          %concat.5 : Tensor = aten::cat(%features, %5)
          %res : Tensor[] = prim::ListConstruct(%concat.1, %concat.2, %concat.3, %concat.4, %concat.5)
          return (%res)
      )IR";
  parseIR(input, graph.get());
  OptimizeConcat(graph);
}

} // namespace jit
} // namespace torch
