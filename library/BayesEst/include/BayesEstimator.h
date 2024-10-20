#include <iostream>
#include <vector>

class BayesianEstimator {
public:
    BayesianEstimator();
    BayesianEstimator(double priorMean, double priorVariance, double likelihoodMean, double likelihoodVariance);

    void addObservation(double observation);
    double getPosteriorMean();

private:
    std::vector<double> observations;
    
    double priorMean{};
    double priorVariance{};
    double likelihoodMean{};
    double likelihoodVariance{};
};

