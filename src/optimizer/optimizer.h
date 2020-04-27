//
// Created by gs1010 on 27/04/20.
//

#ifndef MLPROJECT_SRC_OPTIMIZER_OPTIMIZER_H_
#define MLPROJECT_SRC_OPTIMIZER_OPTIMIZER_H_

#include "../network/network.h"
class Optimizer {
 public:
  virtual ~Optimizer() = default;
  virtual void OptimizeBackward(Network *currNet, const arma::mat &&partialDerivativeOutput) = 0;
};

#endif //MLPROJECT_SRC_OPTIMIZER_OPTIMIZER_H_
