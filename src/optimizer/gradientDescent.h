//
// Created by gs1010 on 27/04/20.
//

#ifndef MLPROJECT_SRC_OPTIMIZER_GRADIENTDESCENT_H_
#define MLPROJECT_SRC_OPTIMIZER_GRADIENTDESCENT_H_

#include "optimizer.h"

class GradientDescent : public Optimizer {
 public:
  ~GradientDescent() override = default;
  void OptimizeBackward(Network *currNet,
                        const arma::mat &&partialDerivativeOutput,
                        const double momentum = 0.0) override;
  void OptimizeUpdateWeight(Network *network, const double learningRate,
                            const double weightDecay, const double momentum) override;

};

#endif //MLPROJECT_SRC_OPTIMIZER_GRADIENTDESCENT_H_
