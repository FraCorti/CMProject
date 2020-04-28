//
// Created by gs1010 on 27/04/20.
//

#ifndef MLPROJECT_SRC_OPTIMIZER_LBFGS_H_
#define MLPROJECT_SRC_OPTIMIZER_LBFGS_H_

#include <deque>
#include "optimizer.h"

class LBFGS : public Optimizer {
 private:
  //! The pair is defined as: {s_{i},y_{i}} where s_{i}= w_{k+1} - w_{k} and y_{k}= \nabla f_{k+1} - \nabla f_{k}.
  //! deque chose because we need to iterate forward and backward.
  //! The most recently curvature information are pushed in the front on the queue.
  //! The pop operation is done on the tail of the queue.
  std::deque<std::pair<arma::mat, arma::mat>>
      pastCurvature; //TODO: oggetto (o struct?) di size k per memorizzare le scorse iterate
  int storageSize;
  double lineSearch(Network *currNet);
  void computeLayersDirections(std::vector<Layer> &net);
  inline void searchDirection(arma::mat &&approxInvHessian, arma::mat &&q, arma::mat &&currentLayerDirection);
 public:
  ~LBFGS() override = default;
  void OptimizeBackward(Network *currNet, const arma::mat &&partialDerivativeOutput) override;

};

#endif //MLPROJECT_SRC_OPTIMIZER_LBFGS_H_
