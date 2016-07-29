/**
 *
 * A program to test a Sawtooth Neural Network
 * Author: Brandon Trabucco
 * Date: 2016/07/27
 *
 */

#include "TimeDelayNetwork.h"
#include "DatasetAdapter.h"
#include "OutputTarget.h"
#include <vector>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string.h>
#include <sys/time.h>
#include <stdlib.h>
#include <math.h>
using namespace std;

long long getMSec() {
	struct timeval tp;
	gettimeofday(&tp, NULL);
	return tp.tv_sec * 1000 + tp.tv_usec / 1000;
}

struct tm *getDate() {
	time_t t = time(NULL);
	struct tm *timeObject = localtime(&t);
	return timeObject;
}

int main(int argc, char *argv[]) {
	cout << "Program initializing" << endl;
	if (argc < 3) {
		cout << argv[0] << " <learning rate> <decay rate> <size ...>" << endl;
		return -1;
	}

	int frameWindow = 100;
	int updatePoints = 100;
	int savePoints = 10;
	int maxEpoch = 1000;
	double errorBound = 0.01;
	double mse = 0;
	double learningRate = atof(argv[1]), decayRate = atof(argv[2]);
	long long networkStart, networkEnd, sumTime = 0, iterationStart;

	const int _day = getDate()->tm_mday;


	/**
	 *
	 * 	Open file streams to save data samples from Neural Network
	 * 	This data can be plotted via GNUPlot
	 *
	 */
	ostringstream errorDataFileName;
	errorDataFileName << "/u/trabucco/Desktop/Temporal_Convergence_Data_Files/" <<
			(getDate()->tm_year + 1900) << "-" << (getDate()->tm_mon + 1) << "-" << _day <<
			"_Single-Core-TDNN-Error_" << learningRate <<
			"-learning_" << decayRate << "-decay.csv";
	ofstream errorData(errorDataFileName.str(), ios::app);
	if (!errorData.is_open()) return -1;


	ostringstream accuracyDataFileName;
	accuracyDataFileName << "/u/trabucco/Desktop/Temporal_Convergence_Data_Files/" <<
			(getDate()->tm_year + 1900) << "-" << (getDate()->tm_mon + 1) << "-" << _day <<
			"_Single-Core-TDNN-Accuracy_" << learningRate <<
			"-learning_" << decayRate << "-decay.csv";
	ofstream accuracyData(accuracyDataFileName.str(), ios::app);
	if (!accuracyData.is_open()) return -1;


	networkStart = getMSec();
	DatasetAdapter dataset = DatasetAdapter();
	networkEnd = getMSec();
	cout << "KTH Dataset loaded in " << (networkEnd - networkStart) << "msecs" << endl;


	TimeDelayNetwork network = TimeDelayNetwork(dataset.getFrameSize() * 100, learningRate, decayRate);


	for (int i = 0; i < (argc - 3); i++) {
		network.addLayer(atoi(argv[3 + i]));
	} network.addLayer(6);

	bool converged = false;
	for (int e = 0; (e < maxEpoch)/* && (!e || (((mse1 + mse2)/2) > errorBound))*/; e++) {
		vector<double> error;
		networkStart = getMSec();
		while (dataset.nextTrainingVideo()) {
			while (dataset.nextTrainingFrame()) {
				DatasetExample data = dataset.getTrainingFrame();
				if (network.getTimeStepSize() < frameWindow) {
					network.loadTimeStep(data.frame);
				} else {
					network.pushTimeStep(data.frame);
					error = network.train(OutputTarget::getOutputFromTarget(data.label));
				}
			}
		}

		int c = 0;
		while (dataset.nextTestVideo()) {
			vector<double> output;
			while (dataset.nextTestFrame()) {
				DatasetExample data = dataset.getTrainingFrame();
				if (network.getTimeStepSize() < frameWindow) {
					network.loadTimeStep(data.frame);
				} else {
					network.pushTimeStep(data.frame);
					output = network.classify();
					if (OutputTarget::getTargetFromOutput(output) == data.label) c++;
				}
			}
		} networkEnd = getMSec();

		mse = 0;
		for (int i = 0; i < error.size(); i++)
			mse += error[i] * error[i];
		mse /= error.size() * 2;

		if (((e + 1) % (maxEpoch / updatePoints)) == 0) {
			cout << "Epoch " << e << " completed in " << (networkEnd - networkStart) << "msecs" << endl;
			cout << "Error[" << e << "] = " << mse << endl;
			cout << "Accuracy[" << e << "] = " << (100.0 * (float)c / (float)dataset.getTestSize()) << endl;
		} errorData << e << ", " << mse << endl;
		accuracyData << e << ", " << (100.0 * (float)c / (float)dataset.getTestSize()) << endl;

		dataset.reset();
	}

	errorData.close();

	return 0;
}
