//
// Created by gs1010 on 29/11/19.
//

#include "network.h"
#include "../optimizer/gradientDescent.h"
#include "../optimizer/LBFGS.h"
#include "../optimizer/ProximalBundleMethod.h"
#include "../regularizer/L2.h"
#include "../regularizer/L1.h"

void Network::Add(Layer &layer) {
  net.push_back(layer);
}

/**
 *  Initilialize the weight parameter of the layer stored inside the network
 *
 *  @param upperBound The maximum random value generated
 *  @param lowerBound The minimum random value generated
 *  @param seed The seed set by the user
 * */
void Network::Init(const double upperBound = 1, const double lowerBound = -1, const int seed) {
  //std::cout <<"Seed: "<< seed << std::endl;
  for (Layer &i : net) {
    i.Init(upperBound, lowerBound, seed);
  }
}

/**
 * Shuffle and then split the training set in training data and training labels. After they are
 * passed in the network.
 *
 * @param trainingSet Deep copy of the training set
 * @param epoch Number of total shuffling of the training set feed in the network
 * @param batchSize Number of the example feed in the network for each forward pass
 * @param learningRate Adjust weight parameter
 * */
double Network::Train(arma::mat validationSet, arma::mat validationLabelSet, arma::mat trainingSet,
                      int labelCol,
                      int epoch,
                      int batchSize,
                      double learningRate,
                      double weightDecay,
                      double momentum) {
  if (batchSize > trainingSet.n_rows) {
    batchSize = trainingSet.n_rows;
  }
  //Weighted learning rate
  learningRate = learningRate / batchSize;
  weightDecay = (weightDecay * batchSize) / trainingSet.n_rows;
  trainingSet = arma::shuffle(trainingSet);
  arma::mat currentError = arma::zeros(1, 1);
  arma::mat deltaError;
  arma::mat previousError;
  double thresholdStopCondition = 0.00001;
  bool stopCondition = false;
  double nDelta = 0.0;
  for (int currentEpoch = 1; currentEpoch <= epoch && !stopCondition; currentEpoch++) {

    // Split the data from the training set.
    arma::mat trainLabels = arma::mat(trainingSet.memptr() + (trainingSet.n_cols - labelCol) * trainingSet.n_rows,
                                      trainingSet.n_rows,
                                      labelCol,
                                      false,
                                      false);

    // Split the labels from the training set.
    arma::mat trainingData = arma::mat(trainingSet.memptr(),
                                       trainingSet.n_rows,
                                       trainingSet.n_cols - labelCol,
                                       false,
                                       false);

    arma::mat epochError = arma::zeros(1);
    try {
      train(std::move(trainingData),
            std::move(trainLabels),
            std::move(epochError),
            batchSize,
            learningRate,
            weightDecay,
            momentum);
    }
    catch (Exception &e) {
      std::cout << e.what() << std::endl;
      stopCondition = true;
    }

    // add of the stop condition on validation set error
    previousError = currentError;
    currentError = arma::zeros(1, 1);
    Test(std::move(validationSet), std::move(validationLabelSet), std::move(currentError));

    arma::mat errortemp = arma::join_rows(epochError, currentError);
    errortemp.print("");
    deltaError = previousError - currentError;
    if (deltaError.at(0, 0) < 0) {
      nDelta++;
    }
    if (currentError.has_nan() || (deltaError.at(0, 0) < thresholdStopCondition && deltaError.at(0, 0) > 0)) {
      stopCondition = true;
    }
    // shuffle the training set for the new epoch
    trainingSet = arma::shuffle(trainingSet);
  }
  return nDelta;
}

/**  Retrieve the batch and its correspondents label. Then the batch is forwarded, the error is computed
 *  and the weights are updated proportionally.
 *
 * */
void Network::train(const arma::mat &&trainingData,
                    const arma::mat &&trainLabels,
                    arma::mat &&epochError,
                    int batchSize,
                    double learningRate,
                    double weightDecay,
                    double momentum) {

  int start = 0;
  int end = batchSize - 1;
  arma::mat outputActivateBatch;
  arma::mat partialDerivativeOutput;

  int batchNumber = std::ceil(trainingData.n_rows / batchSize);
  for (int i = 1; i <= batchNumber; i++) {

    arma::mat inputBatch = (trainingData.submat(start, 0, end, trainingData.n_cols - 1)).t();
    forward(std::move(inputBatch), std::move(outputActivateBatch));

    arma::mat labelBatch = (trainLabels.submat(start, 0, end, trainLabels.n_cols - 1)).t();
    arma::mat currentBatchError;
    error(std::move(labelBatch),
          std::move(outputActivateBatch),
          std::move(partialDerivativeOutput),
          std::move(currentBatchError),
          weightDecay);

    epochError = epochError + currentBatchError;

    //! Saving error for optimizer
    batchError = &currentBatchError;

    //! Saving input data for line search
    input = &inputBatch;
    inputLabel = &labelBatch;
    backward(std::move(partialDerivativeOutput), momentum);

    start = end + 1;
    end = i < batchNumber ? batchSize * (i + 1) - 1 : trainingData.n_rows - 1;

    optimizer->OptimizeUpdateWeight(this, learningRate, weightDecay, momentum);
  }
  epochError = epochError / batchNumber;
}

/**
 *  Iterate the network computing the weight and the activation over all the layer.
 *
 *  @param batch Current batch to feed in the network
 *  @param outputActivate Output layer vector produced value after the feedward pass
 *  @param outputWeight
 * */
void Network::forward(arma::mat &&batch, arma::mat &&outputActivate) {
  arma::mat activateWeight = batch;
  arma::mat outputWeight;
  for (Layer &currentLayer : net) {
    currentLayer.SaveInputParameter(activateWeight);    // save the input activated vector of the previous layer
    currentLayer.Forward(std::move(activateWeight), std::move(outputWeight));
    currentLayer.SaveOutputParameter(outputWeight);   // save the vectors of the current layer for backpropagation
    currentLayer.Activate(outputWeight, std::move(activateWeight));
  }
  outputActivate = activateWeight;
}

/**
 *  Make the loss function injected object compute the error made by the network for the data passed in and
 *  the partial derivative vector of the output unit
 *
 *  @param outputActivateBatch  Output value produced by the network
 *  @param trainLabelsBatch  Correct value of the data passed in the network
 *  @param partialDerivativeOutput Error of the current predicted value produced by the network
 * */
void Network::error(const arma::mat &&trainLabelsBatch,
                    arma::mat &&outputActivateBatch,
                    arma::mat &&partialDerivativeOutput,
                    arma::mat &&currentBatchError,
                    double weightDecay) {
  lossFunction->Error(std::move(trainLabelsBatch), std::move(outputActivateBatch), std::move(currentBatchError));
  if (weightDecay > 0) {
    double weightsSum = regularizer->ForError(this);
    currentBatchError += (weightDecay * weightsSum);
  }

  lossFunction->ComputePartialDerivative(std::move(trainLabelsBatch),
                                         std::move(outputActivateBatch),
                                         std::move(partialDerivativeOutput));
}

/** Iterate over the network from last layer to first. The gradient of all the layer is computed and
 * stored, then the error is "retropagated" throgh the layer using RetroPropagationError().
 *
 *  @param partialDerivativeOutput Partial derivative of the output layer 
 * */
void Network::backward(const arma::mat &&partialDerivativeOutput, const double momentum) {
  optimizer->OptimizeBackward(this, std::move(partialDerivativeOutput), momentum);
}

/***/arma::mat currentNetError; // fd
void Network::updateWeight(double learningRate, double weightDecay, double momentum) {

  for (Layer &currentLayer : net) {
    arma::mat regMat;
    regularizer->ForWeight(&currentLayer, std::move(regMat));
    currentLayer.SetRegularizationMatrix(std::move(regMat));
    currentLayer.AdjustWeight(learningRate, weightDecay, momentum);
  }

}

void Network::Test(const arma::mat &&testData, const arma::mat &&testLabels, arma::mat &&currentBatchError) {
  arma::mat outputActivateBatch;
  arma::mat testDataCopied = testData;

  inference(std::move(testDataCopied),
            std::move(outputActivateBatch));

  errorTest(std::move(testLabels.t()), std::move(outputActivateBatch), std::move(currentBatchError));
  currentBatchError = arma::mean(currentBatchError);
}
/***/
void Network::inference(arma::mat &&inputData, arma::mat &&outputData) {
  arma::mat activateWeight = inputData.t();
  for (Layer &currentLayer : net) {
    currentLayer.Forward(std::move(activateWeight), std::move(outputData));

    currentLayer.Activate(outputData, std::move(activateWeight));
  }
  outputData = activateWeight;
}

/** Compute forward and retrieve error of the network
 *
 * @param outputError
 * @param regularization
 */
void Network::Evaluate(arma::mat &&outputError, const double regularization) {
  arma::mat activateWeight = input->t();
  arma::mat outputActivateBatch;
  arma::mat currentBatchError;
  arma::mat partialDerivativeOutput;

  forward(std::move(*input), std::move(outputActivateBatch));

  error(std::move(*inputLabel),
        std::move(outputActivateBatch),
        std::move(partialDerivativeOutput),
        std::move(outputError),
        regularization);
}

/***/
void Network::TestWithThreshold(const arma::mat &&testData, const arma::mat &&testLabels, double threshold) {
  arma::mat outputActivateBatch;
  arma::mat testDataCopied = testData;

  inference(std::move(testDataCopied),
            std::move(outputActivateBatch));

  outputActivateBatch = outputActivateBatch.t();

  arma::mat thresholdMatrix = arma::ones<arma::mat>(outputActivateBatch.n_rows, outputActivateBatch.n_cols) * threshold;
  arma::mat resultWithThreshold = arma::conv_to<arma::mat>::from(outputActivateBatch > thresholdMatrix);

  arma::mat conta = arma::conv_to<arma::mat>::from(find((resultWithThreshold - testLabels) == 0));
  double elementiTotali = resultWithThreshold.n_elem;
  double elementiGiusti = conta.n_elem;
  std::cout << "all " << elementiTotali << " conta " << elementiGiusti << " % "
            << (elementiGiusti / elementiTotali) * 100 << std::endl;

}
void Network::SetLossFunction(const std::string loss_function) {
  if (loss_function == "meanSquaredError") {
    lossFunction = new MeanSquaredError();
  } else {
    lossFunction = new MeanSquaredError();
  }
}

/**
 *  Make the loss function injected object compute the error made by the network for the data passed in
 *
 * */
void Network::errorTest(const arma::mat &&trainLabelsBatch,
                        arma::mat &&outputActivateBatch,
                        arma::mat &&currentBatchError) {
  lossFunction->Error(std::move(trainLabelsBatch), std::move(outputActivateBatch), std::move(currentBatchError));
}

/**
 * Clear the internal variable of the network (without delete the lossFunction)
 */
void Network::Clear() {
  for (Layer &currentLayer : net) {
    currentLayer.Clear();
  }
}

/** Return the aliases of the std::vector that contains the layers. This method is needed in
 *  the Optimize class (OptimizeBackward method)
 */
std::vector<Layer> &Network::GetNet() {
  return net;
}

/** Set the optimizer of the Neural network
 *
 */
void Network::SetOptimizer(const std::string optimizer_) {
  if (optimizer_
      == "gradientDescent") { //TODO: mettere funzione che fa diverntare tutti i caratteri piccoli (.down()??)
    optimizer = new GradientDescent();
  } else if (optimizer_ == "LBFGS") {
    optimizer = new LBFGS(net.size());
  } else if (optimizer_ == "proximalBundleMethod") {
    optimizer = new ProximalBundleMethod();
  } else {
    optimizer = new GradientDescent();
  }

}
/***
 *
 * @param learningRate
 * @param weightDecay
 * @param momentum
 */
double Network::LineSearchEvaluate(const double stepSize, const double weightDecay, const double momentum) {
  arma::mat activateWeight = input->t();
  arma::mat outputActivateBatch;
  arma::mat currentBatchError;
  arma::mat partialDerivativeOutput;

  updateWeight(stepSize, weightDecay, momentum);

  forward(std::move(*input), std::move(outputActivateBatch));

  error(std::move(*inputLabel),
        std::move(outputActivateBatch),
        std::move(partialDerivativeOutput),
        std::move(currentBatchError),
        weightDecay);

  auto currentLayer = net.rbegin();
  currentLayer->OutputLayerGradient(std::move(currentBatchError));
  //currentLayer->SetDirection(std::move(currentLayer->GetGradientWeight()));
  arma::mat currentGradientWeight;
  currentLayer->RetroPropagationError(std::move(currentGradientWeight), momentum);
  currentLayer++;
  // Iterate from the precedent Layer of the tail to the head
  for (; currentLayer != net.rend(); currentLayer++) {
    currentLayer->Gradient(std::move(currentGradientWeight));
    //currentLayer->SetDirection(std::move(currentLayer->GetGradientWeight()));
    currentLayer->RetroPropagationError(std::move(currentGradientWeight), momentum);
  }

  return arma::as_scalar(currentBatchError);
}
/** Retrieve the batch error of the network
 *
 * @param batchError_ Store the current batch error
 */
void Network::GetBatchError(arma::mat &&batchError_) {
  batchError_ = *batchError;
}
void Network::SetRegularizer(std::string regularizer_) {
  if (regularizer_
      == "L1") { //TODO: mettere funzione che fa diverntare tutti i caratteri piccoli (.down()??)
    regularizer = new L1();
  } else if (regularizer_ == "L2") {
    regularizer = new L2();
  } else {
    regularizer = new L2();
  }

}
const Regularizer *Network::GetRegularizer() {
  return regularizer;
}
void Network::SetNesterov(bool nesterov_) {
  nesterov = nesterov_;
}
bool Network::GetNesterov() {
  return nesterov;
}
