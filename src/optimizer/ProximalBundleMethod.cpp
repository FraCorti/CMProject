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
  double mu = 2;

  //! Set-up bundle method parameters
  if (singletonInit) {
    init(currNet, std::move(partialDerivativeOutput));
  }

  //! OPTIMIZER
  // Create an environment
  // TODO: Spostare nel costruttore?
  GRBEnv env = GRBEnv(true);
  env.set("LogFile", "mip1.log");
  env.start();

  // Create an empty model
  GRBModel model = GRBModel(env);

  //TODO: move in a method
  //! Alpha value can be different, this is caused by approximation in the machine
  arma::mat alpha =
      arma::mat(subgradients.n_rows, 1, arma::fill::ones) * arma::as_scalar(fc) - subgradients * columnParameters - F;
  // h == beta
  arma::mat beta = arma::max(arma::abs(alpha), gamma * arma::pow(S, 2));
  arma::mat constraintCoeff = arma::join_cols(-arma::ones(1, subgradients.n_cols), subgradients).t(); // G
  subgradients.print("subgradients");
  constraintCoeff.print("constraintCoeff");
  //arma::mat constraintCoeff = subgradients.t();
  // TODO: capire perchè +1
  arma::mat secondGradeCoeff = mu * arma::eye(columnParameters.n_elem + 1, columnParameters.n_elem + 1); // P
  secondGradeCoeff(0, 0) = 0;
  arma::mat firstGradeCoeff = arma::eye(columnParameters.n_elem + 1, 1); // q

  /* This example formulates and solves the following simple QP model:

     minimize    1/2 * x^{T}*P*x+q^{T}*x
     subject to  G*x <= h
                  A*x = b
   It solves it once as a continuous model, and once as an integer model.
*/

  // Variable declaration
  GRBVar x[columnParameters.n_elem + 1];
  for (int i = 0; i < columnParameters.n_elem + 1; i++) {
    x[i] = model.addVar(-GRB_INFINITY, GRB_INFINITY, 0.0, GRB_CONTINUOUS, "x_" + std::to_string(i));
  }

  model.update();
  // Constraint Declaration
  for (int i = 0; i < constraintCoeff.n_cols; i++) {
    GRBLinExpr LHS = 0;
    for (int j = 0; j < constraintCoeff.n_rows; j++) {
      LHS += constraintCoeff(j, i) * x[j];
    }
    std::cout << LHS << std::endl;
    if (!i) {
      model.addConstr(LHS, GRB_LESS_EQUAL, 0);
    } else {
      model.addConstr(LHS, GRB_LESS_EQUAL, beta(i - 1));
    }
  }

  // Set minimization of the model
  model.set(GRB_IntAttr_ModelSense, 1);

  // Set objective function
  GRBQuadExpr obj = 0;
  std::cout << obj << std::endl;

  for (int j = 0; j < columnParameters.n_elem + 1; j++) {
    obj += 0.5 * x[j] * secondGradeCoeff(j, j) * x[j] + firstGradeCoeff[j] * x[j];
  }
  std::cout << obj << std::endl;
  model.setObjective(obj);

  try {
    model.optimize();
  } catch (GRBException e) {
    std::cout << "Error code = " << e.getErrorCode() << std::endl;
    std::cout << e.getMessage() << std::endl;
  } catch (...) {
    std::cout << "Exception during optimization" << std::endl;
  }

  arma::Col<double> updatedParameters(columnParameters.n_elem);  // d

  double v = x[0].get(GRB_DoubleAttr_X);   // v
  for (int i = 1; i < columnParameters.n_elem + 1; i++) {
    std::cout << x[i].get(GRB_StringAttr_VarName) << " "
              << x[i].get(GRB_DoubleAttr_X) << std::endl;
    updatedParameters(i - 1) = x[i].get(GRB_DoubleAttr_X);
  }


  //! Store new weights and retrieve error
  arma::mat currentNetError; // fd
  unvectorizeParameters(currNet, std::move(theta));
  currNet->Evaluate(std::move(currentNetError), 0);

  currentNetError.print("currentNetError");

  //! Store current subgradients (weight and biases) in a single column vector
  currentSubgradient.clear();
  vectorizeGradients(currNet, std::move(currentSubgradient));  // g

  //! Store the error of the network
  arma::mat previousParameterError;  // fc
  unvectorizeParameters(currNet, std::move(columnParameters));
  currNet->Evaluate(std::move(previousParameterError), 0);

  //! Store the transpose of the column gradients
  subgradients =
      arma::join_cols(subgradients, currentSubgradient.t()); // G

  //!
  F = arma::join_cols(F, currentNetError - arma::dot(currentSubgradient, theta)); // F

  //! Line search
  double s_c;
  double s_d;
  double tL = lineSearchL(currNet, v, std::move(columnParameters), std::move(updatedParameters));
  double dNorm = arma::norm(updatedParameters);

  if (tL > 0.5) {  //TODO: pass this parameter (accuracy tollerance) in the constructor ?
    columnParameters += tL * updatedParameters;
    theta = columnParameters;
    s_c = tL * dNorm;
    s_d = 0;
  } else {
    double tR = lineSearchR(currNet, v, std::move(columnParameters), std::move(updatedParameters), tL);
    if (tL > 0) {
      columnParameters += tL * updatedParameters;
      theta = columnParameters + tR * updatedParameters;
      s_c = 0;
      s_d = (tR - tL) * dNorm;
    } else {
      theta = columnParameters + tR * updatedParameters;
      s_c = 0;
      s_d = tR * dNorm;
    }
  }

  S += s_c;
  S = arma::join_cols(S, arma::mat(1, 1, arma::fill::ones) * s_d);
  a = std::max(a + s_c, s_d);
  // TODO: continue

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
  columnParameters.print("columnParameters");
  unvectorizeParameters(currNet, std::move(columnParameters));

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
/**
 *
 * @param v First parameter obtained from the Bundle solver
 * @param c Column parameters of the network (weight and bias)
 * @param d Updated parameters obtained from the Bundle solver
 * @return Step parameter to update the weight
 */
double ProximalBundleMethod::lineSearchL(Network *currNet,
                                         double v,
                                         arma::Col<double> &&c,
                                         const arma::Col<double> &&d) {
  double tL = 0;
  double r = 1;
  while (r - tL > 0.1) { //TODO: pass this parameter (accuracy tollerance) in the constructor
    double m = (r + tL) / 2.0;

    arma::mat lNetError; // error returned from updated weight: columnParameters + tL * d
    unvectorizeParameters(currNet, std::move(c + tL * d));
    currNet->Evaluate(std::move(lNetError), 0);

    arma::mat netError;
    unvectorizeParameters(currNet, std::move(c));
    currNet->Evaluate(std::move(netError), 0);

    //TODO: pass mL in the costructor (0.1)
    if (arma::as_scalar(lNetError) <= arma::as_scalar(netError) + 0.1 * tL * v) {
      tL = m;
    } else {
      r = m;
    }
  }
  return tL;
}

/** Line search R taken from the paper
 *
 * @param v
 * @param c
 * @param d
 * @param tL
 * @return
 */
double ProximalBundleMethod::lineSearchR(Network *currNet,
                                         double v,
                                         const arma::Col<double> &&c,
                                         const arma::Col<double> &&d,
                                         const double tL) {
  double tR = tL;
  double r = 1;
  while (r - tR > 0.0001) { //TODO: pass this parameter 0.0001 in the constructor
    double m = (r + tR) / 2.0;
    arma::mat lNetError; // error returned from updated weight: columnParameters + tL * d
    unvectorizeParameters(currNet, std::move(c + tL * d));
    currNet->Evaluate(std::move(lNetError), 0);

    arma::mat rNetError;
    unvectorizeParameters(currNet, std::move(c + tR * d));
    currNet->Evaluate(std::move(rNetError), 0);

    arma::Col<double> currentColumnGradient;
    vectorizeGradients(currNet, std::move(currentColumnGradient));
    double alpha =
        std::abs(
            arma::as_scalar(lNetError) - arma::as_scalar(rNetError) - (tL - tR) * arma::dot(currentSubgradient, d));
    if (-alpha + arma::dot(currentSubgradient, d) >= 0.99 * v) { // TODO: pass 0.99 as parameter
      tR = m;
    } else {
      r = m;
    }
  }
  return tR;
}
/** Set up initial parameters for bundle method
 *
 */
void ProximalBundleMethod::init(Network *currNet, const arma::mat &&partialDerivativeOutput) {

  //! Store the parameters (weight and bias) in a single column vector.
  vectorizeParameters(currNet, std::move(columnParameters));  // columnParameters == c
  theta = arma::mat(columnParameters.memptr(), columnParameters.n_elem, true); // theta

  //! Compute and store the current subgradient
  computeGradient(currNet, std::move(partialDerivativeOutput));

  //! Store the subgradient (weight and biases) in a single column vector
  vectorizeGradients(currNet, std::move(currentSubgradient));

  //! Store the transpose of the column subgradients
  subgradients =
      arma::join_cols(subgradients, arma::rowvec(currentSubgradient.memptr(), currentSubgradient.n_elem, false)); // G

  //! Unflat the network weights and retrieve the error with Evaluate()
  unvectorizeParameters(currNet, std::move(columnParameters));
  currNet->Evaluate(std::move(fc), 0); // fc

  //! TODO: Dare un nome ad f
  F = arma::join_cols(F, fc - arma::dot(currentSubgradient, columnParameters));
  // subgrad locality measure
  S = arma::mat(1, 1, arma::fill::zeros);
  a = 0;

  //! Set singleton parameter to false
  singletonInit = false;
}
