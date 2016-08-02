/*
 * DatasetAdapter.cpp
 *
 *  Created on: Jul 26, 2016
 *      Author: trabucco
 */

#include "DatasetAdapter.h"

DatasetAdapter::DatasetAdapter() {
	// TODO Auto-generated constructor stub

	ifstream trainingDatasetFile("/stash/tlab/datasets/KTH/binary/training_dataset.bin", ios::in | ios::binary);
	ifstream testDatasetFile("/stash/tlab/datasets/KTH/binary/test_dataset.bin", ios::in | ios::binary);
	ifstream trainingLabelsFile("/stash/tlab/datasets/KTH/binary/training_labels.bin", ios::in | ios::binary);
	ifstream testLabelsFile("/stash/tlab/datasets/KTH/binary/test_labels.bin", ios::in | ios::binary);

	streampos trainingDatasetFileSize;
	streampos testDatasetFileSize;
	streampos trainingLabelsFileSize;
	streampos testLabelsFileSize;

	char *trainingDatasetFileData;
	char *testDatasetFileData;
	char *trainingLabelsFileData;
	char *testLabelsFileData;

	trainingIndex = -1;
	testIndex = -1;
	trainingFrameIndex = -1;
	testFrameIndex = -1;

	if (trainingDatasetFile.is_open() &&
			testDatasetFile.is_open() &&
			trainingLabelsFile.is_open() &&
			testLabelsFile.is_open()) {
		trainingDatasetFile.seekg(0, ios::end);
		testDatasetFile.seekg(0, ios::end);
		trainingLabelsFile.seekg(0, ios::end);
		testLabelsFile.seekg(0, ios::end);

		trainingDatasetFileSize = trainingDatasetFile.tellg();
		testDatasetFileSize = testDatasetFile.tellg();
		trainingLabelsFileSize = trainingLabelsFile.tellg();
		testLabelsFileSize = testLabelsFile.tellg();

		trainingDatasetFileData = new char[trainingDatasetFileSize];
		testDatasetFileData = new char[testDatasetFileSize];
		trainingLabelsFileData = new char[trainingLabelsFileSize];
		testLabelsFileData = new char[testLabelsFileSize];

		trainingDatasetFile.seekg(0, ios::beg);
		testDatasetFile.seekg(0, ios::beg);
		trainingLabelsFile.seekg(0, ios::beg);
		testLabelsFile.seekg(0, ios::beg);

		trainingDatasetFile.read(trainingDatasetFileData, trainingDatasetFileSize);
		testDatasetFile.read(testDatasetFileData, testDatasetFileSize);
		trainingLabelsFile.read(trainingLabelsFileData, trainingLabelsFileSize);
		testLabelsFile.read(testLabelsFileData, testLabelsFileSize);

		if (trainingDatasetFile.fail()) cout << "Read failure" << endl;
		if (testDatasetFile.fail()) cout << "Read failure" << endl;
		if (trainingLabelsFile.fail()) cout << "Read failure" << endl;
		if (testLabelsFile.fail()) cout << "Read failure" << endl;

		trainingDatasetFile.close();
		testDatasetFile.close();
		trainingLabelsFile.close();
		testLabelsFile.close();
	} else cout << "Error opening files" << endl;


	// copy the labels and images
	for (int i = 0; i < trainingLabelsFileSize; i++) {
		dataset.trainingLabels.push_back((int)trainingLabelsFileData[i]);
	}

	cout << "Imported training labels" << endl;

	for (int i = 0; i < testLabelsFileSize; i++) {
		dataset.testLabels.push_back((int)testLabelsFileData[i]);
	}

	cout << "Imported test labels" << endl;

	vector<double> buffer;
	for (int i = 0; i < (trainingDatasetFileSize / encodedSize); i++) {
		for (int j = 0; j < encodedSize; j++) {
			if (j == (encodedSize - 1)) {
				if (trainingDatasetFileData[(i * encodedSize) + j] == videoEnd) {
					dataset.trainingVideos.push_back(buffer);
					buffer.clear();
				}
			} else {
				unsigned int value = trainingDatasetFileData[(i * encodedSize) + j];
				buffer.push_back((((double)value / (double)range) > 0.5) ? 1.0 : 0.0);
			}
		}
	}

	cout << "Imported training videos" << endl;

	for (int i = 0; i < (testDatasetFileSize / encodedSize); i++) {
		for (int j = 0; j < encodedSize; j++) {
			if (j == (encodedSize - 1)) {
				if (testDatasetFileData[(i * encodedSize) + j] == videoEnd) {
					dataset.testVideos.push_back(buffer);
					buffer.clear();
				}
			} else {
				unsigned int value = testDatasetFileData[(i * encodedSize) + j];
				buffer.push_back((((double)value / (double)range) > 0.5) ? 1.0 : 0.0);
			}
		}
	}

	cout << "Imported test videos" << endl;

	delete[] trainingDatasetFileData;
	delete[] testDatasetFileData;
	delete[] trainingLabelsFileData;
	delete[] testLabelsFileData;
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


