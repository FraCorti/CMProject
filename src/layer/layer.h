//
// Created by checco on 26/11/19.
//

#ifndef MLPROJECT_SRC_LAYER_H_
#define MLPROJECT_SRC_LAYER_H_

#include "armadillo"
#include "../activationFunction/activationFunction.h"

class Layer {
 private:
  int inSize;
  int outSize;
  //! weights pesi del layer corrente poer ogni nodo al suo interno
  arma::mat weight;
  //! bias del layer corrente
  arma::mat bias;
  //! locally-instored delta weight object
  arma::mat deltaWeight;
  //! locally-instored delta bias object
  arma::mat deltaBias;
  //! gradiente dei Weight del layer
  arma::mat gradientWeight;
  //! gradiente del layer
  arma::mat gradient;
  //! parametri di input del layer
  arma::mat inputParameter;
  //! parametri di output del layer
  arma::mat outputParameter;
  //! ActivationFunction utilizzata nel layer
  ActivationFunction *activationFunction = nullptr;
  //! Direction of update of the weight
  arma::mat direction;
  arma::mat regularizationMatrix;

 public:
  const arma::mat &GetDeltaBias() const;
  /**
   * Create the Linear layer object using the specified number of units.
   *
   * @param inSize The number of input units.
   * @param outSize The number of output units.
   */
  Layer(const int inSize, const int outSize, const std::string activationFunctionString);

  /**
   * Initialize the layer parameter.
   */
  void Initialize();

  /**
   *
   * @param input
   * @param output
   */
  void Activate(const arma::mat &input, arma::mat &&output);

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  void Forward(const arma::mat &&input, arma::mat &&output, const double nesterovMomentum = 0.0);

  /**
 * Line Search feed forward pass of a neural network, evaluating the function
 * f(x) by propagating the activity forward through f.
 *
 * @param input Input data used for evaluating the specified function.
 * @param output Resulting output activation.
 */
  void LineSearchForward(const arma::mat &&input,
                         arma::mat &&output,
                         const double stepSize,
                         const double weightDecay,
                         const double nesterovMomentum = 0.0);

  /**
  * Ordinary feed backward pass of a neural network, calculating the function
  * f(x) by propagating x backwards trough f. Using the results from the feed
  * forward pass.
  *
  * @param input The propagated input activation.
  * @param gy The backpropagated error.
  * @param g The calculated gradient.
  */
  void Backward(const arma::mat &&input, arma::mat &&gy, arma::mat &&g);

  /**
   * Calculate the gradient using the output delta and the input activation.
   *
   * @param input The input parameter used for calculating the gradient.
   *
   * @param partialDerivativeOutput The calculated error.
   * @param gradient The calculated gradient.
   */
  void OutputLayerGradient(const arma::mat &&partialDerivativeOutput);
  void Gradient(const arma::mat &&summationGradientWeight);
  void RetroPropagationError(arma::mat &&retroPropagatedError, const double nesterovMomentum = 0.0);
  void SaveOutputParameter(const arma::mat &input);
  void SaveInputParameter(const arma::mat &input);
  void AdjustWeight(const double learningRate, const double weightDecay, const double momentum);
  void Clear();
  void PrintWeight();
  [[nodiscard]] const arma::mat &GetWeight() const;
  [[nodiscard]] const arma::mat &GetBias() const;
  [[nodiscard]] const arma::mat &GetDelta() const;
  [[nodiscard]] const arma::mat &GetGradientWeight() const;
  [[nodiscard]] const arma::mat GetGradientBias() const;
  [[nodiscard]] const arma::mat &GetInputParameter() const;
  [[nodiscard]] const arma::mat &GetOutputParameter() const;
  [[nodiscard]] int GetInSize() const;
  [[nodiscard]] int GetOutSize() const;
  [[nodiscard]] std::pair<int, int> GetWeightDimensions() const;
  [[nodiscard]] int GetBiasRow() const;
  void Init(const double upperBound, const double lowerBound, const int seed);
  void SetDirection(const arma::mat &&optimizerComputedDirection);
  void SetWeight(const arma::mat &&newWeight);
  void SetRegularizationMatrix(const arma::mat &&regularizationMatrix_);
  void SetBias(const arma::mat &&newBias);
  const arma::mat &GetDirection() const;
};

#endif //MLPROJECT_SRC_LAYER_H_

