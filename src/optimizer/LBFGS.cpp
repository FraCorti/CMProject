//
// Created by gs1010 on 27/04/20.
//

#include "LBFGS.h"
/***
 * Iterate over the network from last layer to first. The gradient of all the layer is computed and
 * stored, then the error is "retropagated" throgh the layer using RetroPropagationError().
 *
 * @param currNetwork Current network considered
 * @param partialDerivativeOutput Partial derivative of the output layer
 */
void LBFGS::OptimizeBackward(Network *currNetwork, const arma::mat &&partialDerivativeOutput) {
  std::vector<Layer> &net = currNetwork->GetNet();
  auto currentLayer = net.rbegin();
  currentLayer->OutputLayerGradient(std::move(partialDerivativeOutput));
  arma::mat currentGradientWeight;
  currentLayer->RetroPropagationError(std::move(currentGradientWeight));
  currentLayer++;
  size_t indexLayer = 0;
  // Iterate from the precedent Layer of the tail to the head
  for (; currentLayer != net.rend(); currentLayer++) {

    //! Add "old" gradient (\nabla f_{k-1})
    arma::mat oldGradient = currentLayer->GetGradientWeight();
    currentLayer->Gradient(std::move(currentGradientWeight));

    //! Save previous y_{k-1}= \nabla f_{k} - \nabla f_{k-1}
    if (pastCurvatureLayer.size()) {
      pastCurvatureLayer[indexLayer].begin()->second = currentLayer->GetGradientWeight() - oldGradient;
      pastCurvatureLayer[indexLayer].begin()->second.reshape(oldGradient.n_elem, 1);
    }

    currentLayer->RetroPropagationError(std::move(currentGradientWeight));
    indexLayer++;
  }
  for (indexLayer = 0; indexLayer < net.size(); indexLayer++) {
    computeLayersDirections(net, indexLayer);
  }
}
/***
 *
 * @param currNet
 * @return
 */
double LBFGS::lineSearch(Network *currNet) {
  return 0;
}

/***
 * Following Algorithm 7.4 in chapter 7 of Numerical Optimization Jorge Nocedal Stephen J. Wright
 * @param net Current network to be considered
 */
void LBFGS::computeLayersDirections(std::vector<Layer> &net, const size_t indexLayer) {
  for (Layer &currentLayer : net) {
    const arma::uword sizeGradient = currentLayer.GetGradientWeight().n_elem;
    //Compute initial hessian
    //! H_{k}^{0} = \gamma_{k}*I
    //! where \gamma_{k} = \frac{s^T_{k-1}y_{k-1}}{y^T_{k-1}y_{k-1}}
    arma::mat approxInvHessian(sizeGradient, 1, arma::fill::ones);
    //! Skip first gamma computation
    if (pastCurvatureLayer[indexLayer].size()) {
      arma::mat
          gamma = (pastCurvatureLayer[indexLayer].front().first.t() * pastCurvatureLayer[indexLayer].front().second)
          / (pastCurvatureLayer[indexLayer].front().second.t() * pastCurvatureLayer[indexLayer].front().second);
      approxInvHessian = gamma * approxInvHessian;
    }
    arma::mat q = currentLayer.GetGradientWeight();

    //! reshape weight matrix to column vector
    q.reshape(q.n_elem, 1);

    arma::mat currentLayerDirection;
    searchDirection(std::move(approxInvHessian), std::move(q), std::move(currentLayerDirection), indexLayer);

    //! reshape direction column vector to original matrix form
    currentLayerDirection.reshape(currentLayer.GetOutSize(), currentLayer.GetInSize());
    currentLayer.SetDirection(std::move(currentLayerDirection));

    //! reshape weight column vector to original matrix form
    q.reshape(currentLayer.GetOutSize(), currentLayer.GetInSize());
  }

}
/***
 *
 * @param approxInvHessian
 * @param q
 * @param r
 */
void LBFGS::searchDirection(arma::mat &&approxInvHessian, arma::mat &&q, arma::mat &&r, const size_t indexLayer) {
  //! Saving alpha and rho results to avoid redundant computation
  std::vector<arma::mat> alpha;
  alpha.reserve(pastCurvatureLayer.size());
  std::vector<arma::mat> rho;
  rho.reserve(pastCurvatureLayer.size());
  auto currentReverseRho = rho.rbegin();
  auto currentReverseAlpha = alpha.rbegin();

  //! Iterate from head to tail  of the current[indexLayer] layer( from the earliest curvature information to the latest)
  for (std::pair<arma::mat, arma::mat> &currentPastCurvature : pastCurvatureLayer[indexLayer]) {
    //! \rho= \frac{1}{y_{k}^{T}*s_{k}}
    *currentReverseRho = 1 / (currentPastCurvature.second.t() * currentPastCurvature.first);
    *currentReverseAlpha = *currentReverseRho * currentPastCurvature.first.t() * q;
    q = q - *currentReverseAlpha * currentPastCurvature.second;
    currentReverseAlpha++;
    currentReverseRho++;
  }
  r = approxInvHessian * q;
  arma::mat beta;
  auto currentRho = rho.begin();
  auto currentAlpha = alpha.begin();
  //! Iterate from tail to head ( from the latest curvature information to the earliest)
  for (auto currentPastCurvature = pastCurvatureLayer[indexLayer].rbegin();
       currentPastCurvature != pastCurvatureLayer[indexLayer].rend();
       ++currentPastCurvature) {
    beta = *currentRho * currentPastCurvature->second.t() * r;
    r = r + currentPastCurvature->first * (*currentAlpha - beta);
    currentAlpha++;
    currentRho++;
  }
}
/***
 *|
 * @param currNetwork
 */
void LBFGS::OptimizeUpdateWeight(Network *currNetwork, const double learningRate,
                                 const double weightDecay, const double momentum) {
  std::vector<Layer> &net = currNetwork->GetNet();
  size_t indexLayer = 0;
  for (Layer &currentLayer : net) {
    arma::mat oldWeight = currentLayer.GetWeight();
    currentLayer.AdjustWeight(learningRate, weightDecay, momentum);
    if (pastCurvatureLayer.size() == storageSize) {
      pastCurvatureLayer.pop_back();
    }
    //! Save s_{k}= w_{k+1} - w_{k}
    pastCurvatureLayer[indexLayer].push_front(std::make_pair(currentLayer.GetGradientWeight() - oldWeight,
                                                             arma::mat(0)));
    pastCurvatureLayer[indexLayer].front().first.reshape(oldWeight.n_elem, 1);
    indexLayer++;
  }
}
LBFGS::LBFGS(const int nLayer)
    : storageSize(5),
      pastCurvatureLayer(nLayer, std::deque<std::pair<arma::mat, arma::mat>>(5, std::pair<arma::mat, arma::mat>())) {}
