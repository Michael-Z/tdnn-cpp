/*
 * TimeDelayNetwork.cpp
 *
 *  Created on: Jul 27, 2016
 *      Author: trabucco
 */

#include "TimeDelayNetwork.h"

TimeDelayNetwork::TimeDelayNetwork(int is, double l, double d) {
	// TODO Auto-generated constructor stub
	inputSize = is;
	learningRate = l;
	decayRate = d;
}

TimeDelayNetwork::~TimeDelayNetwork() {
	// TODO Auto-generated destructor stub
}

int TimeDelayNetwork::getPreviousNeurons() {
	return (blocks.size() == 0) ? inputSize : blocks[blocks.size() - 1].size();
}

void TimeDelayNetwork::addLayer(int size) {
	vector<Neuron> buffer;
	for (int i = 0; i < size; i++) {
		buffer.push_back(Neuron(getPreviousNeurons()));
	} blocks.push_back(buffer);
}

void TimeDelayNetwork::loadTimeStep(vector<double> input) {
	for (int i = 0; i < input.size(); i++)
		timeSteps.push_back(input[i]);
}

void TimeDelayNetwork::pushTimeStep(vector<double> input) {
	timeSteps.erase(timeSteps.begin(), timeSteps.begin() + input.size());
	for (int i = 0; i < input.size(); i++)
		timeSteps.push_back(input[i]);
}

void TimeDelayNetwork::clearTimeSteps() {
	timeSteps.clear();
}

int TimeDelayNetwork::getTimeStepSize() {
	return timeSteps.size();
}

vector<double> TimeDelayNetwork::classify() {
	vector<double> output, connections = timeSteps;
	if (timeSteps.size() == inputSize) {
		// calculate activations from bottom up
		for (int i = 0; i < (blocks.size()); i++) {
			vector<double> activations;
			for (int j = 0; j < blocks[i].size(); j++) {
				// compute the activation
				activations.push_back(blocks[i][j].forward(connections));
				// if at top of network, push to output
				if (i == (blocks.size() - 1)) output.push_back(activations[j]);
			} connections = activations;
		}
		return output;
	} else return output;
}

vector<double> TimeDelayNetwork::train(vector<double> target) {
	vector<double> output, connections = timeSteps;
		if (timeSteps.size() == inputSize && blocks[blocks.size() - 1].size() == target.size()) {
			// start forward pass
			for (int i = 0; i < (blocks.size()); i++) {
				vector<double> activations;
				for (int j = 0; j < blocks[i].size(); j++) {
					// compute the activation
					activations.push_back(blocks[i][j].forward(connections));
					// if at top of network, push to output
					if (i == (blocks.size() - 1)) output.push_back(activations[j]);
				} connections = activations;
			}
			// start backward pass
			vector<double> weightedError;
			for (int i = 0; i < output.size(); i++) {
				weightedError.push_back(output[i] - target[i]);
			} output = weightedError;
			for (int i = (blocks.size() - 1); i >= 0; i--) {
				vector<double> errorSum(blocks[i][0].weight.size(), 0.0);
				for (int j = 0; j < blocks[i].size(); j++) {
					// compute the activation
					vector<double> contribution = blocks[i][j].backward(weightedError[j], learningRate);
					for (int k = 0; k < contribution.size(); k++) {
						errorSum[k] += contribution[k];
					}
				}
				weightedError = errorSum;	// error is not passed correctly
			} learningRate *= decayRate;
			return output;
		} else return output;
}
