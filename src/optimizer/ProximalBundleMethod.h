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
  arma::Col<double> currentSubgradient;
  arma::Col<double> theta;
  arma::mat subgradients;
  arma::mat fc;
  double a;
  arma::mat F;
  arma::mat S;
  bool singletonInit = true; // singleton for init method
  GRBEnv env;
  double mu;
  double gamma;
  double r_max;
  double accuracy_tolerance;
  double tL_tolerance;
  double mR;
  double mL;
  void init(Network *currNet, const arma::mat &&partialDerivativeOutput);
  double lineSearchL(Network *currNet, double v, arma::Col<double> &&c, const arma::Col<double> &&d);
  double lineSearchR(Network *currNet,
                     double v,
                     const arma::Col<double> &&c,
                     const arma::Col<double> &&d,
                     const double stepL);
  void optimize(arma::Col<double> &&updatedParameters,
                const arma::mat &&constraintCoeff, const arma::mat &&beta,
                const arma::mat &&firstGradeCoeff,
                const arma::mat &&secondGradeCoeff, double &v);
  void vectorizeParameters(Network *currNet, arma::Col<double> &&columnParameters);
  void vectorizeGradients(Network *currNet, arma::Col<double> &&columnGradients);
  void computeGradient(Network *network, const arma::mat &&partialDerivativeOutput);
  void unvectorizeParameters(Network *currNet, arma::Col<double> &&updatedParameters);
  void setupSolverParameters(arma::mat &&alpha, arma::mat &&beta, arma::mat &&constraintCoeff,
                             arma::mat &&secondGradeCoeff, arma::mat &&firstGradeCoeff);
 public:
  ~ProximalBundleMethod() override = default;
  ProximalBundleMethod();
  void OptimizeBackward(Network *currNet,
                        const arma::mat &&partialDerivativeOutput,
                        const double momentum = 0.0) override;
  void OptimizeUpdateWeight(Network *network, const double learningRate,
                            const double weightDecay, const double momentum) override;

  void SetParameters(double mu_ = 1,
                     double gamma_ = 0.5,
                     double r_max_ = 1,
                     double accuracy_tolerance_ = 0.001,
                     double mR_ = 0.99,
                     double mL_ = 0.01,
                     double tL_tolerance_ = 0.5);

};

#endif //CMPROJECT_SRC_OPTIMIZER_PROXIMALBUNDLEMETHOD_H_