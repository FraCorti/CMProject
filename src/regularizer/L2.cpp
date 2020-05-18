//
// Created by gs1010 on 18/05/20.
//

#include "L2.h"
double L2::ForError(Network *network) const {
  double weightsSum = 0;

  for (Layer &currentLayer : network->GetNet()) {
    weightsSum += arma::accu(arma::pow(currentLayer.GetWeight(), 2));
  }
  return weightsSum;
}
void L2::ForWeight(Layer *layer, arma::mat &&regularizerTerm) const {

  regularizerTerm = 2 * layer->GetWeight();

}
