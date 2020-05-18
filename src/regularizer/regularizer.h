//
// Created by gs1010 on 18/05/20.
//

#ifndef CMPROJECT_SRC_REGULARIZER_REGULARIZER_H_
#define CMPROJECT_SRC_REGULARIZER_REGULARIZER_H_

#include "../network/network.h"

class Regularizer {
 public:
  virtual ~Regularizer() = default;
  virtual double ForError(Network *network) const = 0;
  virtual void ForWeight(Layer *layer, arma::mat &&regularizerTerm) const = 0;
};

#endif //CMPROJECT_SRC_REGULARIZER_REGULARIZER_H_
