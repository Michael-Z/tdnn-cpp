/*
 * SawtoothNetwork.h
 *
 *  Created on: Jul 27, 2016
 *      Author: trabucco
 */

#ifndef TIMEDELAYNETWORK_H_
#define TIMEDELAYNETWORK_H_

#include <vector>
#include "Neuron.h"
using namespace std;

class TimeDelayNetwork {
private:
	unsigned int inputSize;
	double learningRate;
	double decayRate;
	vector<vector<Neuron> > layers;
	vector<double> timeSteps;
	int getPreviousNeurons();
public:
	TimeDelayNetwork(int is, double l, double d);
	virtual ~TimeDelayNetwork();
	void addLayer(int size);
	vector<double> classify();
	vector<double> train(vector<double> target);
	void loadTimeStep(vector<double> input);
	void clearTimeSteps();
};

#endif /* TIMEDELAYNETWORK_H_ */
