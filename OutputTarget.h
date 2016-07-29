/*
 * OutputTarget.h
 *
 *  Created on: Jun 23, 2016
 *      Author: trabucco
 */

#ifndef OUTPUTTARGET_H_
#define OUTPUTTARGET_H_

#include <vector>
#include <cmath>
#include <iostream>

using namespace std;

class OutputTarget {
private:
	static const int nodes = 6;
	static const int classes = 6;
	static const double classifiers[classes][nodes];
public:
	static vector<double> getOutputFromTarget(int c);
	static int getTargetFromOutput(vector<double> output);
};

#endif /* OUTPUTTARGET_H_ */
