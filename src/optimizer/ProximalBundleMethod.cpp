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

  gamma = 0.5;
  mu = 1;

  //! Set-up bundle method parameters
  if (singletonInit) {
    init(currNet, std::move(partialDerivativeOutput));
  }

  //! Alpha value can be different, this is caused by approximation in the machine
  arma::mat alpha;
  arma::mat beta; // h == beta
  arma::mat constraintCoeff; // G
  arma::mat secondGradeCoeff; // P
  arma::mat firstGradeCoeff; // q

  setupSolverParameters(std::move(alpha),
                        std::move(beta),
                        std::move(constraintCoeff),
                        std::move(secondGradeCoeff),
                        std::move(firstGradeCoeff));
  /**
     This example show the following simple QP model which
     has the same structure (simpler) of the problem formulate by the program and
     solved by Gurobi:

     minimize    1/2 * x^{T}*P*x+q^{T}*x
     subject to  G*x <= h

     It solves it once as a continuous model, and once as an integer model.
  */
  double v;
  arma::Col<double> updatedParameters(columnParameters.n_elem);  // d

  //! Call the optimizer
  optimize(std::move(updatedParameters),
           std::move(constraintCoeff),
           std::move(beta),
           std::move(firstGradeCoeff),
           std::move(secondGradeCoeff),
           v);

  //! Store new weights and retrieve error
  arma::mat currentNetError; // fd
  unvectorizeParameters(currNet, std::move(theta));
  currNet->Evaluate(std::move(currentNetError), 0);

  currentNetError.print("currentNetError");

  //! Store current subgradients (weight and biases) in a single column vector
  currentSubgradient.clear();
  computeGradient(currNet, std::move(partialDerivativeOutput));
  vectorizeGradients(currNet, std::move(currentSubgradient));  // g
  std::cout << "norm currentSubgradient " << (arma::norm(currentSubgradient)) << std::endl;
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

  std::cout << "v: " << v << std::endl;

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
    arma::mat bias = currentLayer.GetBias();

    //! Concatenate weight and bias
    arma::Col<double> layersParameters =
        arma::join_cols(arma::Col(weight.memptr(), weight.n_elem, true),
                        arma::Col(bias.memptr(), bias.n_elem, true));

    columnParameters = arma::join_cols(columnParameters, layersParameters);
  }
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
  }
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
  while (r - tL > 0.001) { //TODO: pass this parameter (accuracy tollerance) in the constructor
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
  currNet->Evaluate(std::move(fc), 0); // fc


  //! TODO: Dare un nome ad f
  F = arma::join_cols(F, fc - arma::dot(currentSubgradient, columnParameters));
  // subgrad locality measure
  S = arma::mat(1, 1, arma::fill::zeros);
  a = 0;

  //! Set singleton parameter to false
  singletonInit = false;
}

/** Setup parameters for the solver
 *
 */
void ProximalBundleMethod::setupSolverParameters(arma::mat &&alpha,
                                                 arma::mat &&beta,
                                                 arma::mat &&constraintCoeff,
                                                 arma::mat &&secondGradeCoeff,
                                                 arma::mat &&firstGradeCoeff) {


  //! Alpha value can be different, this is caused by approximation in the machine
  alpha =
      (arma::mat(subgradients.n_rows, 1, arma::fill::ones) * arma::as_scalar(fc))
          - (subgradients * columnParameters - F);
  // h == beta
  beta = arma::max(arma::abs(alpha), gamma * arma::pow(S, 2));
  constraintCoeff = arma::join_rows(-arma::ones(subgradients.n_rows, 1), subgradients).t(); // G
  secondGradeCoeff = mu * arma::eye(columnParameters.n_elem + 1, columnParameters.n_elem + 1); // P
  secondGradeCoeff(0, 0) = 0;
  firstGradeCoeff = arma::eye(columnParameters.n_elem + 1, 1); // q
}

/** Set up the solver and compute the quadratic program
 *
 * @param updatedParameters Updated parameters founded by the solver
 */
void ProximalBundleMethod::optimize(arma::Col<double> &&updatedParameters,
                                    const arma::mat &&constraintCoeff,
                                    const arma::mat &&beta,
                                    const arma::mat &&firstGradeCoeff,
                                    const arma::mat &&secondGradeCoeff,
                                    double &v) {
  // Create an empty model
  GRBModel model = GRBModel(env);
  //! Set to 1 if you wuold see the Gurobi output
  model.set(GRB_IntParam_LogToConsole, 0);
  // Variable declaration
  GRBVar x[columnParameters.n_elem + 1];
  x[0] = model.addVar(-GRB_INFINITY, GRB_INFINITY, 0.0, GRB_CONTINUOUS, "v");
  for (int i = 0; i < columnParameters.n_elem; i++) {
    x[i + 1] = model.addVar(-GRB_INFINITY, GRB_INFINITY, 0.0, GRB_CONTINUOUS, "w_" + std::to_string(i));
  }

  model.update();
  // Constraint Declaration
  for (int i = 0; i < constraintCoeff.n_cols; i++) {
    GRBLinExpr LHS = 0;
    for (int j = 0; j < constraintCoeff.n_rows; j++) {
      LHS += constraintCoeff(j, i) * x[j];
    }
    model.addConstr(LHS, GRB_LESS_EQUAL, beta(i));
  }


  // Set minimization of the model
  model.set(GRB_IntAttr_ModelSense, 1);

  // Set objective function
  GRBQuadExpr obj = 0;

  for (int j = 0; j < columnParameters.n_elem + 1; j++) {
    obj += 0.5 * x[j] * secondGradeCoeff(j, j) * x[j] + firstGradeCoeff[j] * x[j];
  }
  model.setObjective(obj);

  try {
    model.optimize();
  } catch (GRBException e) {
    std::cout << "Error code = " << e.getErrorCode() << std::endl;
    std::cout << e.getMessage() << std::endl;
  } catch (...) {
    std::cout << "Exception during optimization" << std::endl;
  }

  v = x[0].get(GRB_DoubleAttr_X);   // v

  for (int i = 1; i < columnParameters.n_elem + 1; i++) {
    updatedParameters(i - 1) = x[i].get(GRB_DoubleAttr_X);
  }
}
ProximalBundleMethod::ProximalBundleMethod() : env(true) {
  // Create an environment
  env.set("LogFile", "mip1.log");
  env.start();
}
