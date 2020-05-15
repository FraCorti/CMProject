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

  double gamma = 0.5;
  double mu = 1;
  //! Store the parameters (weight and bias) in a single column vector.
  vectorizeParameters(currNet, std::move(columnParameters));
  //columnParameters.print("Column parameters after move");
  computeGradient(currNet, std::move(partialDerivativeOutput));

  //! Store the gradients (weight and biases) in a single column vector
  vectorizeGradients(currNet, std::move(columnGradients));

  //! Store the transpose of the column gradients
  subgradients = arma::join_cols(subgradients, arma::rowvec(columnGradients.memptr(), columnGradients.n_elem, false));

// LINE search subgradients

  arma::mat fc;
  currNet->GetBatchError(std::move(fc));

  //! TODO: Dare un nome ad f
  F = arma::join_cols(F, fc - arma::dot(columnGradients, columnParameters));
  // subgrad locality measure
  S = arma::mat(1, 1, arma::fill::zeros);

  //! Ottimizzatore

  // Create an environment
  // TODO: Spostare nel costruttore?
  GRBEnv env = GRBEnv(true);
  env.set("LogFile", "mip1.log");
  env.start();

  // Create an empty model
  GRBModel model = GRBModel(env);

  // Create parameters
  //! Alpha value can be different, this is caused by approximation in the machine
  arma::mat alpha =
      arma::mat(subgradients.n_rows, 1, arma::fill::ones) * arma::as_scalar(fc) - subgradients * columnParameters - F;
  arma::mat beta = arma::max(arma::abs(alpha), gamma * arma::pow(S, 2));
  arma::mat constraintCoeff = arma::join_cols(-arma::ones(1, subgradients.n_cols), subgradients).t();
  arma::mat secondGradeCoeff = mu * arma::eye(columnParameters.n_elem + 1, columnParameters.n_elem + 1);
  secondGradeCoeff(0, 0) = 0;
  arma::mat firstGradeCoeff = arma::eye(columnParameters.n_elem + 1, 1);

  for (int i = 0; i < columnParameters.n_elem + 1; i++) {
    model.addVar(-GRB_INFINITY, GRB_INFINITY, 0.0, GRB_CONTINUOUS, "x_" + std::to_string(i));
  }

  //GRBQuadExpr obj = x * x + x * y + y * y + y * z + z * z + 2 * x;




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
  /*
  for (Layer &currentLayer : net) {
    currentLayer.GetWeight().print("Updated weight");
    currentLayer.GetBias().print("Updated bias");
  }
   */
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
