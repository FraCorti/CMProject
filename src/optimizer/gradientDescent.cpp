//
// Created by gs1010 on 27/04/20.
//

#include "gradientDescent.h"

/**   Iterate from last layer to first layer, during the iteration the gradient is computed (stored inside layer)
 *    and retropagated using RetropagationError().
 *
 *    @param currNet Pointer to the neural network object to be optimized
 *    @param partialDerivativeOutput Partial derivative of curreNet output layer
 * */
void GradientDescent::OptimizeBackward(Network *currNet, const arma::mat &&partialDerivativeOutput) {
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
}
