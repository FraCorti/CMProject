//
// Created by gs1010 on 10/05/20.
//

#ifndef CMPROJECT_SRC_OPTIMIZER_PROXIMALBUNDLEMETHOD_H_
#define CMPROJECT_SRC_OPTIMIZER_PROXIMALBUNDLEMETHOD_H_

#include <deque>
#include "optimizer.h"
#include <gurobi_c++.h>

class ProximalBundleMethod : public Optimizer {
  std::vector<std::deque<std::tuple<arma::mat, arma::mat, arma::mat>>>
      B;
  size_t storageSize;
  void vectorizeParameters(Network *currNet, arma::Col<double> &&columnParameters);
  void vectorizeGradients(Network *currNet, arma::Col<double> &&columnGradients);
  void computeGradient(Network *network, const arma::mat &&partialDerivativeOutput);
  void unvectorizeParameters(Network *currNet, arma::mat &&updatedParameters);
 public:
  ~ProximalBundleMethod() override = default;
  ProximalBundleMethod(const int nLayer);
  void OptimizeBackward(Network *currNet, const arma::mat &&partialDerivativeOutput) override;
  void OptimizeUpdateWeight(Network *network, const double learningRate,
                            const double weightDecay, const double momentum) override;

};

#endif //CMPROJECT_SRC_OPTIMIZER_PROXIMALBUNDLEMETHOD_H_
