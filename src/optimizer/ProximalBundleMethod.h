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
  arma::Col<double> theta;
  arma::mat subgradients;
  arma::mat fc;
  double a;
  arma::mat F;
  arma::mat S;
  bool singletonInit = true; // singleton for init method
  void init(Network *currNet, const arma::mat &&partialDerivativeOutput);
  double lineSearchL(Network *currNet, double v, arma::Col<double> &&c, const arma::Col<double> &&d);
  double lineSearchR(Network *currNet,
                     double v,
                     const arma::Col<double> &&c,
                     const arma::Col<double> &&d,
                     const double stepL);
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