/*
 * SawtoothNetwork.h
 *
 *  Created on: Jul 27, 2016
 *      Author: trabucco
 */

#ifndef TIMEDELAYNETWORK_H_
#define TIMEDELAYNETWORK_H_

#include <vector>
#include <stdlib.h>
#include "Neuron.h"
#include <string.h>
#include <omp.h>
using namespace std;

class TimeDelayNetwork {
private:
	int inputSize;
	int frameWindow;
	double learningRate;
	double decayRate;
	vector<vector<Neuron> > layers;
	vector<double> timeSteps;
	int getPreviousNeurons();
public:
	TimeDelayNetwork(int is, int fw, double l, double d);
	virtual ~TimeDelayNetwork();
	void addLayer(int size);
	vector<double> classify();
	vector<double> train(vector<double> target);
	void pushTimeStep(vector<double> input);
	void clearTimeSteps();
	int getTimeStepSize();
};

#endif /* TIMEDELAYNETWORK_H_ */
