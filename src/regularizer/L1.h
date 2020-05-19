//
// Created by gs1010 on 18/05/20.
//

#ifndef CMPROJECT_SRC_REGULARIZER_L1_H_
#define CMPROJECT_SRC_REGULARIZER_L1_H_

#include "regularizer.h"
class L1 : public Regularizer {
 public:
  ~L1() override = default;
  double ForError(Network *network) const override;
  void ForWeight(Layer *layer, arma::mat &&regularizerTerm) const override;
};

#endif //CMPROJECT_SRC_REGULARIZER_L1_H_
