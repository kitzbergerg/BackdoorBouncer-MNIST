path_model_poisoned = model/poisoned.pth
path_model_filtered = model/filtered.pth
path_data_train_modified = data/modified/train.pth
path_data_train_filtered = data/filtered/train.pth
path_data_feed_forward_modified = data/feed_forward/modified.pkl
path_data_feed_forward_filtered = data/feed_forward/filtered.pkl

all: data_generate train test feed_forward outlier_calculation train_new test_new

setup:
	mkdir data
	mkdir data/modified
	mkdir data/filtered
	mkdir data/feed_forward
	mkdir model
	pip install torch torchvision numpy scipy matplotlib

clean:
	rm data/modified/*
	rm data/filtered/*
	rm data/feed_forward/*
	rm model/*


data_generate:
	python src/data_generate.py $(path_data_train_modified)

train:
	python src/model_train.py $(path_data_train_modified) $(path_model_poisoned)
test:
	python src/model_test.py $(path_model_poisoned)
feed_forward:
	python src/model_feed_forward.py $(path_model_poisoned) $(path_data_train_modified) $(path_data_feed_forward_modified)

outlier_calculation:
	python src/data_outlier_calculation.py $(path_data_feed_forward_modified) $(path_data_train_modified) $(path_data_train_filtered)

train_new:
	python src/model_train.py $(path_data_train_filtered) $(path_model_filtered)
test_new:
	python src/model_test.py $(path_model_filtered)
feed_forward_new:
	python src/model_feed_forward.py $(path_model_filtered) $(path_data_train_modified) $(path_data_feed_forward_filtered)