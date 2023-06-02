#include "BayesEst/BayesEstimator.h"

int main() {
    BayesianEstimator estimator(2.0, 1.0, 3.0, 2.0);

    estimator.addObservation(4.0);
    estimator.addObservation(5.0);

    double posteriorMean = estimator.getPosteriorMean();
    std::cout << "Posterior mean: " << posteriorMean << std::endl;

    return 0;
}
