//
// Created by gs1010 on 10/05/20.
//

#include "ProximalBundleMethod.h"
void ProximalBundleMethod::OptimizeBackward(Network *currNet, const arma::mat &&partialDerivativeOutput) {

  std::vector<Layer> &net = currNet->GetNet();
  auto currentLayer = net.rbegin();
  currentLayer->OutputLayerGradient(std::move(partialDerivativeOutput));
  arma::mat currentGradientWeight;
  currentLayer->RetroPropagationError(std::move(currentGradientWeight));
  currentLayer++;
  // Iterate from the precedent Layer of the tail to the head
  for (; currentLayer != net.rend(); currentLayer++) {
    currentLayer->Gradient(std::move(currentGradientWeight));
    currentLayer->RetroPropagationError(std::move(currentGradientWeight));
  }

  

  // Risolutore master problem
  // Evaluate with new weight
  // add weight, function and gradient of x^*


}
void ProximalBundleMethod::OptimizeUpdateWeight(Network *network,
                                                const double learningRate,
                                                const double weightDecay,
                                                const double momentum) {

}
ProximalBundleMethod::ProximalBundleMethod(const int nLayer) : B(nLayer), storageSize(25) {

}
