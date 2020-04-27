//
// Created by gs1010 on 27/04/20.
//

#ifndef MLPROJECT_SRC_OPTIMIZER_GRADIENTDESCENT_H_
#define MLPROJECT_SRC_OPTIMIZER_GRADIENTDESCENT_H_
#include "../network/network.h"
class GradientDescent : public Optimizer {
 public:
  ~GradientDescent() override = default;
  void OptimizeBackward(Network *currNet, const arma::mat &&partialDerivativeOutput) override;

};

#endif //MLPROJECT_SRC_OPTIMIZER_GRADIENTDESCENT_H_
