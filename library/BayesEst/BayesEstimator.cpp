#include "BayesEstimator.h"

BayesianEstimator::BayesianEstimator(double priorMean, double priorVariance, double likelihoodMean, double likelihoodVariance)
        : priorMean(priorMean), priorVariance(priorVariance), likelihoodMean(likelihoodMean), likelihoodVariance(likelihoodVariance) {
    }

void BayesianEstimator::addObservation(double observation) {
        observations.push_back(observation);
    }

double BayesianEstimator::getPosteriorMean() {
        double posteriorMean;
        double posteriorVariance;

        double sum = priorVariance * likelihoodMean;
        for (double observation : observations) {
            sum += likelihoodVariance * observation;
        }

        posteriorMean = (priorMean * likelihoodVariance + sum) / (priorVariance + observations.size() * likelihoodVariance);
        posteriorVariance = 1 / (1 / priorVariance + observations.size() / likelihoodVariance);

        return posteriorMean;
    }