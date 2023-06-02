#include <iostream>
#include <vector>

class BayesianEstimator {
private:
    std::vector<double> observations;
    double priorMean;
    double priorVariance;
    double likelihoodMean;
    double likelihoodVariance;

public:
    BayesianEstimator(double priorMean, double priorVariance, double likelihoodMean, double likelihoodVariance);

    void addObservation(double observation);
    double getPosteriorMean();
};

