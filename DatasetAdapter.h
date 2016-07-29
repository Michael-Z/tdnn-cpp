/*
 * DatasetAdapter.h
 *
 *  Created on: Jul 26, 2016
 *      Author: trabucco
 */

#ifndef DATASETADAPTER_H_
#define DATASETADAPTER_H_

#include <string>
#include <vector>
#include <iostream>
#include <math.h>
#include <fstream>

using namespace std;

typedef struct {
	vector<vector<double> > trainingVideos;
	vector<vector<double> > testVideos;
	vector<int> trainingLabels;
	vector<int> testLabels;
} Dataset;

typedef struct {
	vector<double> frame;
	int label;
} DatasetExample;

class DatasetAdapter {
private:
	const int range = 255;
	const int frameSize = 300;
	const int encodedSize = 301;
	const char frameEnd = '\f';
	const char videoEnd = '\v';
	int trainingIndex;
	int testIndex;
	int trainingFrameIndex;
	int testFrameIndex;
	Dataset dataset;
public:
	DatasetAdapter();
	virtual ~DatasetAdapter();
	int getFrameSize();
	int getTrainingSize();
	int getTestSize();
	bool nextTrainingVideo();
	bool nextTestVideo();
	bool nextTrainingFrame();
	bool nextTestFrame();
	bool isLastTrainingFrame();
	bool isLastTestFrame();
	DatasetExample getTrainingFrame();
	DatasetExample getTestFrame();
	void reset();
};

#endif /* DATASETADAPTER_H_ */
