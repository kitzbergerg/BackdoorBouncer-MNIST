all: data_generate train test feed_forward outlier_calculation train_new test_new

setup:
	mkdir data
	mkdir data/MNIST
	mkdir data/MNIST/raw
	mkdir data/MNIST/original
	mkdir data/MNIST/modified
	mkdir data/MNIST/filtered
	mkdir model

clean:
	rm data/MNIST/original/*
	rm data/MNIST/modified/*
	rm data/MNIST/filtered/*
	rm model/*
	rm data/feed_forward_output.pkl


data_generate:
	python data_generate.py

train:
	python model_train.py data/MNIST/modified/data_train_modified.pth model/trained_model.pth
test:
	python model_test.py model/trained_model.pth

feed_forward:
	python model_feed_forward.py

outlier_calculation:
	python data_outlier_calculation.py

train_new:
	python model_train.py data/MNIST/filtered/data_train_filtered.pth model/trained_model_filtered.pth
test_new:
	python model_test.py model/trained_model_filtered.pth