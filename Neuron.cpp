/*
 * Neuron.cpp
 *
 *  Created on: Jun 22, 2016
 *      Author: trabucco
 */

#include "Neuron.h"

long long Neuron::n = 0;

Neuron::Neuron(int nConnections) {
	// TODO Auto-generated constructor stub
	activation = 0; activationPrime = 0;
	connections = nConnections;
	default_random_engine g(time(0) + (n++));
	normal_distribution<double> d(0, 1);
	weight = (double *)malloc(sizeof(double) * nConnections);
	impulse = (double *)calloc(nConnections, sizeof(double));
	for (int i = 0; i < connections; i++) {
		weight[i] = (d(g));
	}
}

Neuron::~Neuron() {
	// TODO Auto-generated destructor stub
}

double Neuron::sigmoid(double input) {
	return 1 / (1 + exp(-input));
}

double Neuron::sigmoidPrime(double input) {
	return sigmoid(input) * (1 - sigmoid(input));
}

double Neuron::activate(double input) {
	return tanh(input);
}

double Neuron::activatePrime(double input) {
	return (1 - (tanh(input) * tanh(input)));
}

double Neuron::forward(double *input) {
	double sum = 0;
	memcpy(impulse, input, (sizeof(double) * connections));

	// find the weighted sum of all input
	for (int i = 0; i < connections; i++) {
		//cout << weight[i] << " ";
		sum += input[i] * weight[i];
	}// cout << " sum : " << sum << " weights : " << weight.size() << endl;
	activation = activate(sum);
	activationPrime = activatePrime(sum);
	return activation;
}

double *Neuron::backward(double errorPrime, double learningRate) {
	double *weightedError;
	weightedError = (double *)malloc(sizeof(double) * connections);
	// update all weights
	for (int i = 0; i < connections; i++) {
		weightedError[i] = (errorPrime * weight[i] * activationPrime);
		weight[i] -= learningRate * errorPrime * impulse[i];
	}
	return weightedError;
}

