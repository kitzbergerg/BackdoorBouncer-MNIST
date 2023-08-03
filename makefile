all: generate_data train test

generate_data:
	python generate_data.py

train:
	python model_train.py

test:
	python model_test.py

setup:
	mkdir data
	mkdir data/MNIST
	mkdir data/MNIST/raw
	mkdir data/MNIST/modified
	mkdir model

clean:
	rm data/MNIST/modified/*
	rm model/*