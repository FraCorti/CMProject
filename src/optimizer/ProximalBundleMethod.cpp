//
// Created by gs1010 on 10/05/20.
//

#include "ProximalBundleMethod.h"

/**
 *
 * @param currNet
 * @param partialDerivativeOutput
 */
void ProximalBundleMethod::OptimizeBackward(Network *currNet, const arma::mat &&partialDerivativeOutput) {

  //! Store the parameters (weight and bias) in a single column vector.
  vectorizeParameters(currNet, std::move(columnParameters));
  //columnParameters.print("Column parameters after move");
  computeGradient(currNet, std::move(partialDerivativeOutput));

  //! Store the gradients (weight and biases) in a single column vector
  vectorizeGradients(currNet, std::move(columnGradients));

  //! Store the transpose of the column gradients
  arma::rowvec subgradients = arma::rowvec(columnGradients.memptr(), columnGradients.n_elem, false);

// LINE search subgradients

  arma::mat fc;
  currNet->GetBatchError(std::move(fc));

  arma::mat f = fc - arma::dot(subgradients, columnParameters);

  // SET-UP input
  // CALL Gurobi


  /*try {

    // Create an environment
    GRBEnv env = GRBEnv(true);
    env.set("LogFile", "mip1.log");
    env.start();

    // Create an empty model
    GRBModel model = GRBModel(env);

    // Create variables
    GRBVar x = model.addVar(0.0, 1.0, 0.0, GRB_BINARY, "x");
    GRBVar y = model.addVar(0.0, 1.0, 0.0, GRB_BINARY, "y");
    GRBVar z = model.addVar(0.0, 1.0, 0.0, GRB_BINARY, "z");

    // Set objective: maximize x + y + 2 z
    model.setObjective(x + y + 2 * z, GRB_MAXIMIZE);

    // Add constraint: x + 2 y + 3 z <= 4
    model.addConstr(x + 2 * y + 3 * z <= 4, "c0");

    // Add constraint: x + y >= 1
    model.addConstr(x + y >= 1, "c1");

    // Optimize model
    model.optimize();

    std::cout << x.get(GRB_StringAttr_VarName) << " "
              << x.get(GRB_DoubleAttr_X) << std::endl;
    std::cout << y.get(GRB_StringAttr_VarName) << " "
              << y.get(GRB_DoubleAttr_X) << std::endl;
    std::cout << z.get(GRB_StringAttr_VarName) << " "
              << z.get(GRB_DoubleAttr_X) << std::endl;

    std::cout << "Obj: " << model.get(GRB_DoubleAttr_ObjVal) << std::endl;

  } catch (GRBException e) {
    std::cout << "Error code = " << e.getErrorCode() << std::endl;
    std::cout << e.getMessage() << std::endl;
  } catch (...) {
    std::cout << "Exception during optimization" << std::endl;
  }
 */

}

/** Compute the gradient for all layers and retropagate the error
 *
 * @param currNet
 * @param partialDerivativeOutput
 */
void ProximalBundleMethod::computeGradient(Network *currNet, const arma::mat &&partialDerivativeOutput) {
  std::vector<Layer> &net = currNet->GetNet();
  auto currentLayer = net.rbegin();
  currentLayer->OutputLayerGradient(std::move(partialDerivativeOutput));
  currentLayer->SetDirection(std::move(currentLayer->GetGradientWeight()));
  arma::mat currentGradientWeight;
  currentLayer->RetroPropagationError(std::move(currentGradientWeight));
  currentLayer++;
  // Iterate from the precedent Layer of the tail to the head
  for (; currentLayer != net.rend(); currentLayer++) {
    currentLayer->Gradient(std::move(currentGradientWeight));
    currentLayer->SetDirection(std::move(currentLayer->GetGradientWeight()));
    currentLayer->RetroPropagationError(std::move(currentGradientWeight));
  }
}

/** Call the unvectorizeParameters() function that store the updated parameters (weights & biases)
 *  in the Nework. In the end clear the class parameters for next iterate
 *
 * @param currNet
 * @param learningRate
 * @param weightDecay
 * @param momentum
 */

void ProximalBundleMethod::OptimizeUpdateWeight(Network *currNet,
                                                const double learningRate,
                                                const double weightDecay,
                                                const double momentum) {
  //! FAKE restore to test unvectorize (column parameters will be updated by the bundle)
  unvectorizeParameters(currNet, std::move(columnParameters));

  //! Empty class parameters column vector for next iterate
  columnParameters.clear();
  columnGradients.clear();
}

/** Return a column vector with all the weights and biases of the network saved in
 *
 * @param columnParameters Saved column vector
 * @param currNet
 */
void ProximalBundleMethod::vectorizeParameters(Network *currNet, arma::Col<double> &&columnParameters) {
  std::vector<Layer> &net = currNet->GetNet();

  for (Layer &currentLayer : net) {
    arma::mat weight = currentLayer.GetWeight();
    //weight.print("Current layer weight");
    arma::mat bias = currentLayer.GetBias();
    //bias.print("Current Layer bias");

    //! Concatenate weight and bias
    arma::Col<double> layersParameters =
        arma::join_cols(arma::Col(weight.memptr(), weight.n_elem, true),
                        arma::Col(bias.memptr(), bias.n_elem, true));

    columnParameters = arma::join_cols(columnParameters, layersParameters);
    //layersParameters.print("Layers parameters");
  }
  //columnParameters.print("Column parameters");
}

/** Store the updated weights and bias inside the layers of the neural network
 *
 * @param currNet
 * @param updatedParameters Column vector with updated weight and bias
 */
void ProximalBundleMethod::unvectorizeParameters(Network *currNet, arma::Col<double> &&updatedParameters) {
  std::vector<Layer> &net = currNet->GetNet();

  int index = 0;
  for (Layer &currentLayer : net) {

    //! Retrieve weight dimensions in pair<n_rows,n_cols> format
    std::pair<int, int> weightDim = currentLayer.GetWeightDimensions();
    int biasDim = currentLayer.GetBiasRow();

    currentLayer.SetWeight(std::move(arma::mat(updatedParameters.memptr() + index,
                                               weightDim.first,
                                               weightDim.second,
                                               true)));
    index += weightDim.first * weightDim.second;

    currentLayer.SetBias(std::move(arma::mat(updatedParameters.memptr() + index,
                                             biasDim,
                                             1,
                                             true)));
    index += biasDim;
  }
  for (Layer &currentLayer : net) {
    currentLayer.GetWeight().print("Updated weight");
    currentLayer.GetBias().print("Updated bias");
  }
}

/** Return a column vector with all the gradient of the weights and biases of the network stored i n
 *
 * @param currNet
 * @param columnGradients Column vector with concatenated gradient of the network
 */
void ProximalBundleMethod::vectorizeGradients(Network *currNet, arma::Col<double> &&columnGradients) {
  std::vector<Layer> &net = currNet->GetNet();

  for (Layer &currentLayer : net) {
    arma::mat gradientWeight = currentLayer.GetGradientWeight();
    arma::mat gradientBias = currentLayer.GetGradientBias();

    //! Concatenate weight and bias gradients
    arma::Col<double> layersParameters =
        arma::join_cols(arma::Col(gradientWeight.memptr(), gradientWeight.n_elem, true),
                        arma::Col(gradientBias.memptr(), gradientBias.n_elem, true));

    columnGradients = arma::join_cols(columnGradients, layersParameters);
    //layersParameters.print("Gradients parameters");
  }
  //columnGradients.print("Column gradient parameters");
}
