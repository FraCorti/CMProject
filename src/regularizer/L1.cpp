//
// Created by gs1010 on 18/05/20.
//

#include "L1.h"
double L1::ForError(Network *network) const {

  double weightsSum = 0;

  for (Layer &currentLayer : network->GetNet()) {
    weightsSum += arma::accu(arma::abs(currentLayer.GetWeight()));
  }

  return weightsSum;
}
void L1::ForWeight(Layer *layer, arma::mat &&regularizerTerm) const {
  regularizerTerm = arma::sign(layer->GetWeight());
}
