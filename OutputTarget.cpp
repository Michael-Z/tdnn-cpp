/*
 * OutputTarget.cpp
 *
 *  Created on: Jun 23, 2016
 *      Author: trabucco
 */

#include "OutputTarget.h"

const double OutputTarget::classifiers[classes][nodes] = {
		{1, -1, -1, -1, -1, -1},
		{-1, 1, -1, -1, -1, -1},
		{-1, -1, 1, -1, -1, -1},
		{-1, -1, -1, 1, -1, -1},
		{-1, -1, -1, -1, 1, -1},
		{-1, -1, -1, -1, -1, 1}
};

vector<double> OutputTarget::getOutputFromTarget(int c) {
	return vector<double>(classifiers[c], classifiers[c] + nodes);
}

int OutputTarget::getTargetFromOutput(vector<double> output) {
	for (int i = 0; i < classes; i++) {
		bool matches = true;
		for (int j = 0; j < nodes; j++) {
			if (abs(output[j] - classifiers[i][j]) >= 1) {
				matches = false;
				break;
			}
		}
		if (matches) return i;
	}
	return -1;
}

