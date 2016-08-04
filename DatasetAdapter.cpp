/*
 * DatasetAdapter.cpp
 *
 *  Created on: Jul 26, 2016
 *      Author: trabucco
 */

#include "DatasetAdapter.h"

DatasetAdapter::DatasetAdapter() {
	// TODO Auto-generated constructor stub

	ifstream trainingDatasetFile("/stash/tlab/datasets/KTH/binary/training_dataset.txt");
	ifstream testDatasetFile("/stash/tlab/datasets/KTH/binary/test_dataset.txt");
	ifstream trainingLabelsFile("/stash/tlab/datasets/KTH/binary/training_labels.txt");
	ifstream testLabelsFile("/stash/tlab/datasets/KTH/binary/test_labels.txt");

	trainingIndex = -1;
	testIndex = -1;
	trainingFrameIndex = -1;
	testFrameIndex = -1;

	if (trainingDatasetFile.is_open() &&
			testDatasetFile.is_open() &&
			trainingLabelsFile.is_open() &&
			testLabelsFile.is_open()) {
		while (!trainingLabelsFile.eof()) {
			int buffer;
			trainingLabelsFile >> buffer;
			//cout << "Label: " << buffer << endl;
			dataset.trainingLabels.push_back(buffer);
		}

		cout << dataset.trainingLabels.size() << " Training labels loaded" << endl;

		while (!testLabelsFile.eof()) {
			int buffer;
			testLabelsFile >> buffer;
			dataset.testLabels.push_back(buffer);
		}

		cout << "Test labels loaded" << endl;

		vector<double> video;
		while (!trainingDatasetFile.eof()) {
			int buffer;
			trainingDatasetFile >> buffer;
			if (buffer == -1) {
				// end of frame

			} else if (buffer == -2) {
				// end of video
				dataset.trainingVideos.push_back(video);
				video.clear();
			} else {
				// inside a frame
				video.push_back(buffer);
			}
		} video.clear();

		cout << "Training videos loaded" << endl;

		while (!testDatasetFile.eof()) {
			int buffer;
			testDatasetFile >> buffer;
			if (buffer == -1) {
				// end of frame

			} else if (buffer == -2) {
				// end of video
				dataset.testVideos.push_back(video);
				video.clear();
			} else {
				// inside a frame
				video.push_back(buffer);
			}
		}

		cout << "Test videos loaded" << endl;

		trainingDatasetFile.close();
		testDatasetFile.close();
		trainingLabelsFile.close();
		testLabelsFile.close();
	} else cout << "Error opening files" << endl;
}

DatasetAdapter::~DatasetAdapter() {
	// TODO Auto-generated destructor stub
}

int DatasetAdapter::getFrameSize() {
	return frameSize;
}

int DatasetAdapter::getTrainingSize() {
	return dataset.trainingVideos.size();
}

int DatasetAdapter::getTestSize() {
	return dataset.testVideos.size();
}

bool DatasetAdapter::nextTrainingVideo() {
	//cout << "Training index " << trainingIndex << endl;
	trainingFrameIndex = -1;
	return (++trainingIndex) < dataset.trainingVideos.size();
}

bool DatasetAdapter::nextTestVideo() {
	testFrameIndex = -1;
	return (++testIndex) < dataset.testVideos.size();
}

bool DatasetAdapter::nextTrainingFrame() {
	//cout << "Frame count " << (dataset.trainingVideos[trainingIndex].size() / frameSize) << endl;
	return (++trainingFrameIndex) < (dataset.trainingVideos[trainingIndex].size() / frameSize);
}

bool DatasetAdapter::nextTestFrame() {
	return (++testFrameIndex) < (dataset.testVideos[testIndex].size() / frameSize);
}

bool DatasetAdapter::isLastTrainingFrame() {
	//cout << trainingFrameIndex << " " << (dataset.trainingVideos[trainingIndex].size() - 1) << endl;
	return (trainingFrameIndex == ((dataset.trainingVideos[trainingIndex].size() / frameSize) - 1));
}

bool DatasetAdapter::isLastTestFrame() {
	return (testFrameIndex == ((dataset.testVideos[testIndex].size() / frameSize) - 1));
}

DatasetExample DatasetAdapter::getTrainingFrame() {
	DatasetExample example;
	example.frame = vector<double>((dataset.trainingVideos[trainingIndex].begin() + (trainingFrameIndex * frameSize)),
			(dataset.trainingVideos[trainingIndex].begin() + ((trainingFrameIndex + 1) * frameSize)));
	example.label = dataset.trainingLabels[trainingIndex];
	return example;
}

DatasetExample DatasetAdapter::getTestFrame() {
	DatasetExample example;
	example.frame = vector<double>((dataset.testVideos[testIndex].begin() + (testFrameIndex * frameSize)),
			(dataset.testVideos[testIndex].begin() + ((testFrameIndex + 1) * frameSize)));
	example.label = dataset.testLabels[testIndex];
	return example;
}

void DatasetAdapter::reset() {
	trainingIndex = -1;
	testIndex = -1;
	trainingFrameIndex = -1;
	testFrameIndex = -1;
}


