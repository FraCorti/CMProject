//
// Created by gs1010 on 27/04/20.
//

#ifndef MLPROJECT_SRC_OPTIMIZER_LBFGS_H_
#define MLPROJECT_SRC_OPTIMIZER_LBFGS_H_

#include "../network/network.h"
class LBFGS : public Optimizer {
 private:
  std::vector<int> pastCurvature; //TODO: oggetto (o struct?) di size k per memorizzare le scorse iterate
  double lineSearch(Network *currNet);
 public:
  ~LBFGS() override = default;
  void OptimizeBackward(Network *currNet, const arma::mat &&partialDerivativeOutput) override;
};

#endif //MLPROJECT_SRC_OPTIMIZER_LBFGS_H_
