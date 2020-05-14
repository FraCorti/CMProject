//
// Created by gs1010 on 10/05/20.
//

#include "ProximalBundleMethod.h"

ProximalBundleMethod::ProximalBundleMethod(const int nLayer) : B(nLayer), storageSize(25) {

}

void ProximalBundleMethod::OptimizeBackward(Network *currNet, const arma::mat &&partialDerivativeOutput) {
  arma::Col<double> columnParameters;
  arma::Col<double> columnGradients;

  //! Store the parameters (weight and bias) in a single column vector
  vectorizeParameters(currNet, std::move(columnParameters));
  //columnParameters.print("Column parameters after move");

  computeGradient(currNet, std::move(partialDerivativeOutput));
  //! Store the gradients (weight and biases) in a single column vector
  vectorizeGradients(currNet, std::move(columnGradients));
  //! Store the transpose of the column gradients
  arma::rowvec subgradient = arma::rowvec(columnGradients.memptr(), columnGradients.n_elem, false);

  arma::mat fc;
  currNet->GetBatchError(std::move(fc));

  arma::mat f = fc - arma::dot(subgradient, columnParameters);

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

  // Risolutore master problem
  // Evaluate with new weight
  // add weight, function and gradient of x^*


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

void ProximalBundleMethod::OptimizeUpdateWeight(Network *network,
                                                const double learningRate,
                                                const double weightDecay,
                                                const double momentum) {

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

/** Store the updated weight and bias inside the neural network
 *
 * @param currNet
 * @param updatedParameters Column vector with updated weight and bias
 */
void ProximalBundleMethod::unvectorizeParameters(Network *currNet, arma::mat &&updatedParameters) {
  std::vector<Layer> &net = currNet->GetNet();

  for (Layer &currentLayer : net) {

  }
}

/** Return a column vector with all the gradient of the weights and biases of the network stored i n
 *
 * @param currNet
 * @param columnGradients Column vector with concatened gradient of the network
 */
void ProximalBundleMethod::vectorizeGradients(Network *currNet, arma::Col<double> &&columnGradients) {
  std::vector<Layer> &net = currNet->GetNet();

  for (Layer &currentLayer : net) {
    arma::mat gradientWeight = currentLayer.GetGradientWeight();
    arma::mat gradientBias = currentLayer.GetGradientBias();
    //! Concatenate weight and gradientBias
    arma::Col<double> layersParameters =
        arma::join_cols(arma::Col(gradientWeight.memptr(), gradientWeight.n_elem, true),
                        arma::Col(gradientBias.memptr(), gradientBias.n_elem, true));

    columnGradients = arma::join_cols(columnGradients, layersParameters);
    //layersParameters.print("Gradients parameters");
  }
  //columnGradients.print("Column gradient parameters");
}
