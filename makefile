path_model_poisoned = model/poisoned.pth
path_model_filtered = model/filtered.pth
path_data_train_modified = data/MNIST/modified/train.pth
path_data_train_filtered = data/MNIST/filtered/train.pth

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
	python src/data_generate.py

train:
	python src/model_train.py $(path_data_train_modified) $(path_model_poisoned)
test:
	python src/model_test.py $(path_model_poisoned)

feed_forward:
	python src/model_feed_forward.py $(path_model_poisoned)

outlier_calculation:
	python src/data_outlier_calculation.py

train_new:
	python src/model_train.py $(path_data_train_filtered) $(path_model_filtered)
test_new:
	python src/model_test.py $(path_model_filtered)