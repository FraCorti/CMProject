//
// Created by gs1010 on 27/04/20.
//

#ifndef CMPROJECT_SRC_OPTIMIZER_GRADIENTDESCENT_H_
#define CMPROJECT_SRC_OPTIMIZER_GRADIENTDESCENT_H_

#include "optimizer.h"
class GradientDescent : public Optimizer {
 public:
  ~GradientDescent() override = default;
  void OptimizeBackward(Network *currNet, const arma::mat &&partialDerivativeOutput) override;

};

#endif //CMPROJECT_SRC_OPTIMIZER_GRADIENTDESCENT_H_
