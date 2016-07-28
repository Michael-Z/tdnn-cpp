/*
 * Neuron.h
 *
 *  Created on: Jun 22, 2016
 *      Author: trabucco
 */

#ifndef NEURON_H_
#define NEURON_H_

#include <math.h>
#include <vector>
#include <time.h>
#include <iostream>
#include <random>
using namespace std;

class Neuron {
private:
	static long long n;
	double sigmoid(double input);
	double sigmoidPrime(double input);
	double activate(double input);
	double activatePrime(double input);
public:
	vector<double> weight;
	vector<double> impulse;
	double activation;
	double activationPrime;
	Neuron(int connections);
	~Neuron();
	double forward(vector<double> input);
	vector<double> backward(double errorPrime, double learningRate);
};

#endif /* NEURON_H_ */
