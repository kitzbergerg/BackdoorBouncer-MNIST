all: generate_data train test

generate_data:
	python generate_data.py

train:
	python model_train.py

test:
	python model_test.py
	
clean:
	rm data/MNIST/modified/*
	rm model/*