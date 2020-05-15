//
// Created by gs1010 on 10/05/20.
//

#ifndef CMPROJECT_SRC_OPTIMIZER_PROXIMALBUNDLEMETHOD_H_
#define CMPROJECT_SRC_OPTIMIZER_PROXIMALBUNDLEMETHOD_H_

#include <deque>
#include "optimizer.h"
#include <gurobi_c++.h>

class ProximalBundleMethod : public Optimizer {
  arma::Col<double> columnParameters;
  arma::Col<double> columnGradients;
  arma::mat subgradients;
  arma::mat F;
  arma::mat S;

  void vectorizeParameters(Network *currNet, arma::Col<double> &&columnParameters);
  void vectorizeGradients(Network *currNet, arma::Col<double> &&columnGradients);
  void computeGradient(Network *network, const arma::mat &&partialDerivativeOutput);
  void unvectorizeParameters(Network *currNet, arma::Col<double> &&updatedParameters);
 public:
  ~ProximalBundleMethod() override = default;
  ProximalBundleMethod() = default;
  void OptimizeBackward(Network *currNet, const arma::mat &&partialDerivativeOutput) override;
  void OptimizeUpdateWeight(Network *network, const double learningRate,
                            const double weightDecay, const double momentum) override;

};

#endif //CMPROJECT_SRC_OPTIMIZER_PROXIMALBUNDLEMETHOD_H_
