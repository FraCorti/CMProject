//
// Created by gs1010 on 27/04/20.
//

#include "LBFGS.h"
#include "../regularizer/regularizer.h"
/***
 * Iterate over the network from last layer to first. The gradient of all the layer is computed and
 * stored, then the error is "retropagated" throgh the layer using RetroPropagationError().
 *
 * @param currNetwork Current network considered
 * @param partialDerivativeOutput Partial derivative of the output layer
 */
void LBFGS::OptimizeBackward(Network *currNetwork, const arma::mat &&partialDerivativeOutput, const double momentum) {
  std::vector<Layer> &net = currNetwork->GetNet();
  auto currentLayer = net.rbegin();

  //! Add "old" gradient (\nabla f_{k-1})
  arma::mat oldGradient = currentLayer->GetGradientWeight();

  currentLayer->OutputLayerGradient(std::move(partialDerivativeOutput));
  size_t indexLayer = net.size() - 1;
  arma::mat currentGradientWeight;

  //! Save previous y_{k-1}= \nabla f_{k} - \nabla f_{k-1}
  if (!pastCurvatureLayer[indexLayer].empty()) {
    pastCurvatureLayer[indexLayer].begin()->second = currentLayer->GetGradientWeight() - oldGradient;
    pastCurvatureLayer[indexLayer].begin()->second.reshape(oldGradient.n_elem, 1);
    secantEquationCondition(indexLayer);
  }
  indexLayer--;
  currentLayer->RetroPropagationError(std::move(currentGradientWeight));
  currentLayer++;
  // Iterate from the precedent Layer of the tail to the head
  for (; currentLayer != net.rend(); currentLayer++) {

    //! Add "old" gradient (\nabla f_{k-1})
    arma::mat oldGradient = currentLayer->GetGradientWeight();
    //! Compute "new" gradient (\nabla f_{k})
    currentLayer->Gradient(std::move(currentGradientWeight));

    //! Save previous y_{k-1}= \nabla f_{k} - \nabla f_{k-1}
    if (!pastCurvatureLayer[indexLayer].empty()) {
      pastCurvatureLayer[indexLayer].begin()->second = currentLayer->GetGradientWeight() - oldGradient;
      pastCurvatureLayer[indexLayer].begin()->second.reshape(oldGradient.n_elem, 1);
      secantEquationCondition(indexLayer);
    }

    currentLayer->RetroPropagationError(std::move(currentGradientWeight));
    indexLayer--;
  }

  computeLayersDirections(net);
}

/** Compute dot product between the gradients store inside the layers \phi'
 *
 * @param currNetwork Current network to be considered
 * @param initialSearchDirectionDotGradient Summation of all the layers dot products result
 */
double LBFGS::computeDirectionDescent(Network *currNetwork) {
  double searchDirectionDotGradient = 0;
  std::vector<Layer> &net = currNetwork->GetNet();
  //! Sum of the \nabla f(w_{k}) * p_{k} where k is the current iteration
  for (auto &currentLayer : net) {
    // We can use currentLayer.GetDirection because in LineSearchEvaluate we don't update the direction
    searchDirectionDotGradient += arma::dot(currentLayer.GetGradientWeight(), currentLayer.GetDirection());
  }
  return searchDirectionDotGradient;
}

/**
 *
 * @param currNet
 * @return Step size that satisfies A-W conditions
 */
double LBFGS::lineSearch(Network *currNetwork, const double weightDecay, const double momentum) {

  double alpha_0 = 0;
  double alpha_max = this->alpha_max;

  // Update the current alpha with a value between currentAlpha and alpha_max
  std::uniform_real_distribution<double> unif(alpha_0, alpha_max);
  // Set the seed to have repeatable executions
  double currentAlpha = unif(re);

  //! Compute \phi'(0) and check the descent direction of the network
  double initialSearchDirectionDotGradient = computeDirectionDescent(currNetwork);
  // Check descent direction
  if (initialSearchDirectionDotGradient > 0.0) {
    std::cout << "L-BFGS line search direction is not a descent direction "
              << "(terminating)!" << std::endl;
    return false;
  }

  // Deep copy of the current network
  Network lineSearchNetworkAlpha0(*currNetwork);

  //! Current error of the network, \phi(0)
  double phi0 = lineSearchNetworkAlpha0.LineSearchEvaluate(0, weightDecay, momentum);

  double previousAlpha = alpha_0;
  double phiPreviousAlpha = std::numeric_limits<double>::max();
  for (int i = 0; i < maxStep; i++) {

    //! Compute \phi(\alpha_{i})
    Network lineSearchNetworkAlphaI(*currNetwork);
    double phiCurrentAlpha = lineSearchNetworkAlphaI.LineSearchEvaluate(currentAlpha, weightDecay, momentum);

    if ((phiCurrentAlpha > phi0 + c1 * currentAlpha * initialSearchDirectionDotGradient)
        || (i && phiCurrentAlpha >= phiPreviousAlpha)) {
      return zoom(currNetwork,
                  weightDecay,
                  momentum,
                  previousAlpha,
                  currentAlpha,
                  phi0,
                  initialSearchDirectionDotGradient);
    }

    //! Compute \phi'(\alpha_{i})
    double currentSearchDirectionDotGradient = computeDirectionDescent(&lineSearchNetworkAlphaI);

    if (std::abs(currentSearchDirectionDotGradient) <= c2 * initialSearchDirectionDotGradient) {
      return currentAlpha;
    }

    if (currentSearchDirectionDotGradient >= 0) {
      return zoom(currNetwork,
                  weightDecay,
                  momentum,
                  currentAlpha,
                  previousAlpha,
                  phi0,
                  initialSearchDirectionDotGradient);
    }
    //! Saving \phi(\alpha_{i-1})
    phiPreviousAlpha = phiCurrentAlpha;
    previousAlpha = currentAlpha;
    std::uniform_real_distribution<double> unif(currentAlpha, alpha_max);
    currentAlpha = unif(re);
  }
  return currentAlpha;
}

/**
 *
 * @param currNetwork
 * @param weightDecay
 * @param momentum
 * @param alphaLow
 * @param alphaHi
 * @param phi0 Current \phi(0)
 * @param initialSearchDirectionDotGradient Current \phi'(0)
 * @return Step size of the current direction
 */
double LBFGS::zoom(Network *currNetwork,
                   const double weightDecay,
                   const double momentum,
                   double alphaLow,
                   double alphaHi,
                   const double phi0,
                   const double initialSearchDirectionDotGradient) {
  int i = 0;
  double alphaJ = 1; //  = alphaHi

  // Set the seed to have repeatable executions

  //! limit number of iteration to obtain a step length in a finite time
  while (i < 100) {

    //! Compute \phi(\alpha_{j})
    Network lineSearchNetworkAlphaJ(*currNetwork);
    double phiCurrentAlphaJ = lineSearchNetworkAlphaJ.LineSearchEvaluate(alphaJ, weightDecay, momentum);

    //! Compute \phi(\alpha_{lo})
    Network lineSearchNetworkAlphaLow(*currNetwork);
    double phiCurrentAlphaLow = lineSearchNetworkAlphaLow.LineSearchEvaluate(alphaLow, weightDecay, momentum);
    double currentSearchDirectionDotGradientAlphaLow = computeDirectionDescent(&lineSearchNetworkAlphaLow);


    //! Compute \alpha_{hi}
    Network lineSearchNetworkAlphaHi(*currNetwork);
    double phiCurrentAlphaHi = lineSearchNetworkAlphaHi.LineSearchEvaluate(alphaHi, weightDecay, momentum);
    double currentSearchDirectionDotGradientAlphaHi = computeDirectionDescent(&lineSearchNetworkAlphaHi);

    //quadraticInterpolation
    if (phiCurrentAlphaJ > phi0 + c1 * alphaJ * initialSearchDirectionDotGradient) {
      alphaJ = quadraticInterpolation(alphaLow,
                                      phiCurrentAlphaLow,
                                      currentSearchDirectionDotGradientAlphaLow,
                                      alphaHi,
                                      phiCurrentAlphaHi);
      Network lineSearchNetworkAlphaJ(*currNetwork);
      phiCurrentAlphaJ = lineSearchNetworkAlphaJ.LineSearchEvaluate(alphaJ, weightDecay, momentum);
    }
    //cubicInterpolation
    if (phiCurrentAlphaJ > phi0 + c1 * alphaJ * initialSearchDirectionDotGradient) {
      double alphaCubicInter = cubicInterpolation(alphaLow,
                                                  phiCurrentAlphaLow,
                                                  currentSearchDirectionDotGradientAlphaLow,
                                                  alphaHi,
                                                  phiCurrentAlphaHi,
                                                  currentSearchDirectionDotGradientAlphaHi);
      if (alphaCubicInter > 0 && alphaCubicInter <= 1) {
        alphaJ = alphaCubicInter;
        Network lineSearchNetworkAlphaJ(*currNetwork);
        phiCurrentAlphaJ = lineSearchNetworkAlphaJ.LineSearchEvaluate(alphaJ, weightDecay, momentum);
      }
    }
    // Bisection interpolation if quadratic goes wrong
    if (0 == alphaJ) {
      alphaJ = alphaLow + (alphaHi - alphaLow) / 2;
      Network lineSearchNetworkAlphaJ(*currNetwork);
      phiCurrentAlphaJ = lineSearchNetworkAlphaJ.LineSearchEvaluate(alphaJ, weightDecay, momentum);
    }

    if (phiCurrentAlphaJ > phi0 + c1 * alphaJ * initialSearchDirectionDotGradient
        || phiCurrentAlphaJ >= phiCurrentAlphaLow) {
      alphaHi = alphaJ;
    } else {

      //! Compute \phi'(\alpha_{j})
      double currentSearchDirectionDotGradient = computeDirectionDescent(&lineSearchNetworkAlphaJ);

      if (std::abs(currentSearchDirectionDotGradient) <= -c2 * initialSearchDirectionDotGradient) {
        return alphaJ;
      }

      if (currentSearchDirectionDotGradient * (alphaHi - alphaLow) >= 0) {
        alphaHi = alphaLow;
      }
      alphaLow = alphaJ;
    }

    i++;
  }
  return alphaJ;
}

/***
 * Following Algorithm 7.4 in chapter 7 of Numerical Optimization Jorge Nocedal Stephen J. Wright
 * @param net Current network to be considered
 */
void LBFGS::computeLayersDirections(std::vector<Layer> &net) {
  size_t indexLayer = 0;
  for (Layer &currentLayer : net) {
    const arma::uword sizeGradient = currentLayer.GetGradientWeight().n_elem;
    //Compute initial hessian
    //! H_{k}^{0} = \gamma_{k}*I
    //! where \gamma_{k} = \frac{s^T_{k-1}y_{k-1}}{y^T_{k-1}y_{k-1}}
    arma::mat approxInvHessian(sizeGradient, 1, arma::fill::ones);
    //! Skip first gamma computation
    if (!pastCurvatureLayer[indexLayer].empty()) {
      double
          gamma = arma::as_scalar(
          (pastCurvatureLayer[indexLayer].front().first.t() * pastCurvatureLayer[indexLayer].front().second)
              / (pastCurvatureLayer[indexLayer].front().second.t() * pastCurvatureLayer[indexLayer].front().second));
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

    indexLayer++;
  }

}
/***
 * Compute the direction as H_{k}\nabla f_{k} and save the result of it inside r. Algorithm 7.4 of "Numerical optimization" J. Nocedal
 * @param approxInvHessian
 * @param q the gradient of the current layer, is modified following the Algorithm 7.4
 * @param r descent direction of the function, is passed to avoid copy
 * @param indexLayer index of the current layer in the "pastCurvatureLayer" vector
 */
void LBFGS::searchDirection(arma::mat &&approxInvHessian, arma::mat &&q, arma::mat &&r, const size_t indexLayer) {
  //! Saving alpha and rho results to avoid redundant computation
  std::vector<double> alpha(pastCurvatureLayer[indexLayer].size());
  std::vector<double> rho(pastCurvatureLayer[indexLayer].size());
  auto currentReverseRho = rho.rbegin();
  auto currentReverseAlpha = alpha.rbegin();

  //! Iterate from head to tail  of the current[indexLayer] layer( from the earliest curvature information to the latest)
  for (std::pair<arma::mat, arma::mat> &currentPastCurvature : pastCurvatureLayer[indexLayer]) {
    //! \rho= \frac{1}{y_{k}^{T}*s_{k}}
    *currentReverseRho = 1 / arma::dot(currentPastCurvature.second.t(), currentPastCurvature.first);
    *currentReverseAlpha = arma::as_scalar(*currentReverseRho * currentPastCurvature.first.t() * q);
    q = q - *currentReverseAlpha * currentPastCurvature.second;
    currentReverseAlpha++;
    currentReverseRho++;
  }
  //! Hadamard product because we have two vectors column
  r = approxInvHessian % q;
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
 * Update the weight of the network and add the pair {s_{k} , 0 } for all the layer.
 * The second parameter of the pair y_{k} is added when we compute the next gradient
 * @param currNetwork
 */
void LBFGS::OptimizeUpdateWeight(Network *currNetwork, const double learningRate,
                                 const double weightDecay, const double momentum) {
  double stepSize = lineSearch(currNetwork, weightDecay, momentum);
  std::vector<Layer> &net = currNetwork->GetNet();
  size_t indexLayer = 0;
  for (Layer &currentLayer : net) {
    arma::mat oldWeight = currentLayer.GetWeight();
    arma::mat regMat;
    (currNetwork->GetRegularizer())->ForWeight(&currentLayer, std::move(regMat));
    currentLayer.SetRegularizationMatrix(std::move(regMat));
    currentLayer.AdjustWeight(stepSize, weightDecay, momentum);
    if (pastCurvatureLayer.size() == storageSize) {
      pastCurvatureLayer.pop_back();
    }

    //! Save s_{k}= w_{k+1} - w_{k}
    pastCurvatureLayer[indexLayer].push_front(std::pair<arma::mat, arma::mat>(
        currentLayer.GetWeight() - oldWeight, arma::mat(0, 0, arma::fill::zeros)));

    pastCurvatureLayer[indexLayer].front().first.reshape(oldWeight.n_elem, 1);
    indexLayer++;
  }
}
LBFGS::LBFGS(const int nLayer, const int storageSize, const int seed)
    : pastCurvatureLayer(nLayer), storageSize(storageSize), alpha_max(1), maxStep(100), c1(0.001), c2(0.9) {
  if (seed) {
    re.seed(seed);
  } else {
    std::random_device device;
    re.seed(device());
  }
}

/***
 * Check if secant equation (s_{k}^T y_{k}>0) is satisfied.
 * @param indexLayer index of the current layer in the "pastCurvatureLayer" vector
 */
void LBFGS::secantEquationCondition(const size_t indexLayer) {
  if (arma::dot(pastCurvatureLayer[indexLayer].begin()->first.t(), pastCurvatureLayer[indexLayer].begin()->second)
      <= 0) {
    throw Exception("Unsatisfied secant equation \n");
  }
}
double LBFGS::lineSearchBacktracking(Network *currNetwork, const double weightDecay, const double momentum) {
  double alphaT = 1;
  double rho = 0.54;
  double c = 0.0001;

  std::vector<Layer> &net = currNetwork->GetNet();
  double initialSearchDirectionDotGradient = 0;
  //! Sum of the \nabla f(w_{k}) * p_{k} where k is the current iteration
  for (auto &currentLayer : net) {
    initialSearchDirectionDotGradient += arma::dot(currentLayer.GetGradientWeight(), currentLayer.GetDirection());
  }

  Network lineSearchNetworkAlpha0(*currNetwork);
  double f0 = lineSearchNetworkAlpha0.LineSearchEvaluate(0, weightDecay, momentum);

  Network lineSearchNetworkAlphaT(*currNetwork);
  double fT = lineSearchNetworkAlphaT.LineSearchEvaluate(alphaT, weightDecay, momentum);

  int i;
  while (fT <= f0 + c * alphaT * initialSearchDirectionDotGradient && i < 10000) {
    alphaT *= rho;

    Network lineSearchNetworkAlphaT(*currNetwork);
    fT = lineSearchNetworkAlphaT.LineSearchEvaluate(alphaT, weightDecay, momentum);
    i++;
  }

  return alphaT;
}
double LBFGS::quadraticInterpolation(double alphaLo,
                                     double phiAlphaLo,
                                     double searchDirectionDotGradientAlphaLo,
                                     double alphaHi,
                                     double phiAlphaHi) {
  //! Update the current alpha with the minimum value of the quadratic
  //! quadraticInterpolation between \phi(0) \phi'(0) and \phi(\alpha_{0})


  return -(searchDirectionDotGradientAlphaLo * std::pow(alphaHi, 2))
      / (2 * (phiAlphaHi - phiAlphaLo - searchDirectionDotGradientAlphaLo * alphaHi));

}
double LBFGS::cubicInterpolation(double alphaLo,
                                 double phiAlphaLo,
                                 double searchDirectionDotGradientAlphaLo,
                                 double alphaHi,
                                 double phiAlphaHi,
                                 double searchDirectionDotGradientAlphaHi) {
  double d1 = searchDirectionDotGradientAlphaLo + searchDirectionDotGradientAlphaHi - 3 * (
      phiAlphaLo - phiAlphaHi) / (alphaLo - alphaHi);
  double d2 = (std::signbit(alphaHi - alphaLo) ? -1 : 1) *
      std::sqrt(std::pow(d1, 2) - searchDirectionDotGradientAlphaLo * searchDirectionDotGradientAlphaHi);

  return alphaHi
      - (alphaHi - alphaLo) * ((searchDirectionDotGradientAlphaHi + d2 - d1) / searchDirectionDotGradientAlphaHi
          - searchDirectionDotGradientAlphaLo + 2 * d2);
}
void LBFGS::SetParameters(int storageSize_, int maxStep_, double alpha_max_, double c1_, double c2_) {
  if (alpha_max_ > 1 || alpha_max_ < 0) {
    throw Exception("alpha_max must be between 0 and 1 \n");
  }
  if (c1_ > 1 || c1_ < 0) {
    throw Exception("c1 must be between 0 and 1 \n");
  }
  if (c2_ > 1 || c2_ < 0) {
    throw Exception("c2 must be between 0 and 1 \n");
  }

  storageSize = storageSize_;
  maxStep = maxStep_;
  alpha_max = alpha_max_;
  c1 = c1_;
  c2 = c2_;
}
