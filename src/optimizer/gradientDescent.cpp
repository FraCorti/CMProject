//
// Created by gs1010 on 27/04/20.
//

#include "gradientDescent.h"
#include "../regularizer/regularizer.h"

/**   Iterate from last layer to first layer, during the iteration the gradient is computed (stored inside layer)
 *    and retropagated using RetropagationError().
 *
 *    @param currNet Pointer to the neural network object to be optimized
 *    @param partialDerivativeOutput Partial derivative of curreNet output layer
 * */
void GradientDescent::OptimizeBackward(Network *currNet, const arma::mat &&partialDerivativeOutput,
                                       const double momentum) {
  std::vector<Layer> &net = currNet->GetNet();
  const double nesterovMomentum = momentum * currNet->GetNesterov();
  auto currentLayer = net.rbegin();
  currentLayer->OutputLayerGradient(std::move(partialDerivativeOutput));
  currentLayer->SetDirection(std::move(currentLayer->GetGradientWeight()));
  arma::mat currentGradientWeight;
  currentLayer->RetroPropagationError(std::move(currentGradientWeight), nesterovMomentum);
  currentLayer++;
  // Iterate from the precedent Layer of the tail to the head
  for (; currentLayer != net.rend(); currentLayer++) {
    currentLayer->Gradient(std::move(currentGradientWeight));
    currentLayer->SetDirection(std::move(currentLayer->GetGradientWeight()));
    currentLayer->RetroPropagationError(std::move(currentGradientWeight), nesterovMomentum);
  }
}
/***
 * Update the weight of the network
 * @param network
 * @param learningRate
 * @param weightDecay
 * @param momentum
 */
void GradientDescent::OptimizeUpdateWeight(Network *network,
                                           const double learningRate,
                                           const double weightDecay,
                                           const double momentum) {
  std::vector<Layer> &net = network->GetNet();
  for (Layer &currentLayer : net) {
    arma::mat regMat;
    (network->GetRegularizer())->ForWeight(&currentLayer, std::move(regMat));
    currentLayer.SetRegularizationMatrix(std::move(regMat));
    currentLayer.AdjustWeight(learningRate, weightDecay, momentum);
  }
}
