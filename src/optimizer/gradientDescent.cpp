//
// Created by gs1010 on 27/04/20.
//

#include "gradientDescent.h"
void GradientDescent::OptimizeBackward(Network *currNet, const arma::mat &&partialDerivativeOutput) {
  std::vector<Layer> &net = currNet->getNet();
  auto currentLayer = net.rbegin();
  currentLayer->OutputLayerGradient(std::move(partialDerivativeOutput));   //TODO: move to optimizer(currentLayer,)
  arma::mat currentGradientWeight;
  currentLayer->RetroPropagationError(std::move(currentGradientWeight));
  currentLayer++;
  // Iterate from the precedent Layer of the tail to the head
  for (; currentLayer != net.rend(); currentLayer++) {
    currentLayer->Gradient(std::move(currentGradientWeight));
    currentLayer->RetroPropagationError(std::move(currentGradientWeight));
  }
}
