/*
 * TimeDelayNetwork.cpp
 *
 *  Created on: Jul 27, 2016
 *      Author: trabucco
 */

#include "TimeDelayNetwork.h"

TimeDelayNetwork::TimeDelayNetwork(int is, int fw, double l, double d) {
	// TODO Auto-generated constructor stub
	inputSize = is;
	frameWindow = fw;
	learningRate = l;
	decayRate = d;
	timeSteps = vector<double>((is * fw), 0.0);
}

TimeDelayNetwork::~TimeDelayNetwork() {
	// TODO Auto-generated destructor stub
}

int TimeDelayNetwork::getPreviousNeurons() {
	return (layers.size() == 0) ? (inputSize * frameWindow) : layers[layers.size() - 1].size();
}

void TimeDelayNetwork::addLayer(int size) {
	vector<Neuron> buffer;
	for (int i = 0; i < size; i++) {
		buffer.push_back(Neuron(getPreviousNeurons()));
	} layers.push_back(buffer);
}

void TimeDelayNetwork::pushTimeStep(vector<double> input) {
	timeSteps.erase(timeSteps.begin(), timeSteps.begin() + input.size());
	for (int i = 0; i < input.size(); i++)
		timeSteps.push_back(input[i]);
}

void TimeDelayNetwork::clearTimeSteps() {
	timeSteps.clear();
	timeSteps = vector<double>((inputSize * frameWindow), 0.0);
}

int TimeDelayNetwork::getTimeStepSize() {
	return timeSteps.size() / inputSize;
}

vector<double> TimeDelayNetwork::classify() {
	double *output = (double *)malloc(sizeof(double) * layers[layers.size() - 1].size()),
			*connections = (double *)malloc(sizeof(double) * timeSteps.size());
	copy(timeSteps.begin(), timeSteps.end(), connections);
	// calculate activations from bottom up
	for (int i = 0; i < (layers.size()); i++) {
		double *activations = (double *)malloc(sizeof(double) * layers[i].size());
		#pragma omp parallel for
		for (int j = 0; j < layers[i].size(); j++) {
			// compute the activation
			activations[j] = layers[i][j].forward(connections);
			// if at top of network, push to output
			if (i == (layers.size() - 1)) output[j] = (activations[j]);
		} connections = (double *)realloc(connections, (sizeof(double) * layers[i].size()));
		memcpy(&connections[0], &activations[0], sizeof(double) * layers[i].size());
		free(activations);
	} vector<double> result(&output[0], &output[layers[layers.size() - 1].size() - 1]);
	free(output);
	free(connections);
	return result;
}

vector<double> TimeDelayNetwork::train(vector<double> target) {
	double *output = (double *)malloc(sizeof(double) * layers[layers.size() - 1].size()),
			*connections = (double *)malloc(sizeof(double) * timeSteps.size());
	copy(timeSteps.begin(), timeSteps.end(), connections);
	if (layers[layers.size() - 1].size() == target.size()) {
		// start forward pass
		for (int i = 0; i < (layers.size()); i++) {
			double *activations = (double *)malloc(sizeof(double) * layers[i].size());
			#pragma omp parallel for
			for (int j = 0; j < layers[i].size(); j++) {
				// compute the activation
				activations[j] = layers[i][j].forward(connections);
				// if at top of network, push to output
				if (i == (layers.size() - 1)) output[j] = (activations[j]);
			} connections = (double *)realloc(connections, (sizeof(double) * layers[i].size()));
			memcpy(&connections[0], &activations[0], sizeof(double) * layers[i].size());
			free(activations);
		} free(connections);
		// start backward pass
		double *weightedError = (double *)malloc(sizeof(double) * layers[layers.size() - 1].size());
		#pragma omp parallel for
		for (int i = 0; i < layers[layers.size() - 1].size(); i++) {
			weightedError[i] = (output[i] - target[i]);
		} memcpy(&output[0], &weightedError[0], (sizeof(double) * layers[layers.size() - 1].size()));
		for (int i = (layers.size() - 1); i >= 0; i--) {
			double *errorSum = (double *)calloc(layers[i][0].connections, sizeof(double));
			#pragma omp parallel for
			for (int j = 0; j < layers[i].size(); j++) {
				// compute the activation
				double *contribution = layers[i][j].backward(weightedError[j], learningRate);
				#pragma omp critical
				for (int k = 0; k < layers[i][0].connections; k++) {
					errorSum[k] += contribution[k];
				}
				free(contribution);
			} weightedError = (double *)realloc(weightedError, (sizeof(double) * layers[i][0].connections));
			memcpy(&weightedError[0], &errorSum[0], (sizeof(double) * layers[i][0].connections));
			free(errorSum);
		} learningRate *= decayRate;
		vector<double> result(&output[0], &output[layers[layers.size() - 1].size() - 1]);
		free(weightedError);
		free(output);
		return result;
	} else return vector<double>();
}
