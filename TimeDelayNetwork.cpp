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
	vector<double> output(layers[layers.size() - 1].size()), connections = timeSteps;
	// calculate activations from bottom up
	for (int i = 0; i < (layers.size()); i++) {
		vector<double> activations(layers[i].size());
		#pragma omp parallel for
		for (int j = 0; j < layers[i].size(); j++) {
			// compute the activation
			activations[j] = layers[i][j].forward(connections);
			// if at top of network, push to output
			if (i == (layers.size() - 1)) output[j] = (activations[j]);
		} connections = activations;
	}
	return output;
}

vector<double> TimeDelayNetwork::train(vector<double> target) {
	vector<double> output(layers[layers.size() - 1].size()), connections = timeSteps;
		if (layers[layers.size() - 1].size() == target.size()) {
			// start forward pass
			for (int i = 0; i < (layers.size()); i++) {
				vector<double> activations(layers[i].size());
				#pragma omp parallel for
				for (int j = 0; j < layers[i].size(); j++) {
					// compute the activation
					activations[j] = layers[i][j].forward(connections);
					// if at top of network, push to output
					if (i == (layers.size() - 1)) output[j] = (activations[j]);
				} connections = activations;
			}
			// start backward pass
			vector<double> weightedError(output.size());
			#pragma omp parallel for
			for (int i = 0; i < output.size(); i++) {
				weightedError[i] = (output[i] - target[i]);
			} output = weightedError;
			for (int i = (layers.size() - 1); i >= 0; i--) {
				vector<double> errorSum(layers[i][0].weight.size(), 0.0);
				#pragma omp parallel for
				for (int j = 0; j < layers[i].size(); j++) {
					// compute the activation
					vector<double> contribution = layers[i][j].backward(weightedError[j], learningRate);
					#pragma omp critical
					for (int k = 0; k < contribution.size(); k++) {
						errorSum[k] += contribution[k];
					}
				}
				weightedError = errorSum;	// error is not passed correctly
			} learningRate *= decayRate;
			return output;
		} else return output;
}
