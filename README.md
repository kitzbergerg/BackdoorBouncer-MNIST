# Backdoor attack & prevention

This project aims to demonstrate a simple backdoor attack on the MNIST dataset.

In a second step the poisoned data should be removed from the training data.
This is done by using outlier detection.

## Structure

-   `makefile`: Contains commands to generate data, train and test the model, ...
-   `data/`: Contains the data
-   `model/`: Contains the trained models
-   `custom_mnist.py`: Definition of the custom poisoned data
-   `generate_data.py`: Execute to generate the datasets
-   `model.py`: Definition of the model
-   `model_train.py`: Execute to train the model
-   `model_test.py`: Execute to test the model

## Makefile

-   `setup`: Create directories
-   `all`: Run the whole project from start to finish (run `setup` first)
-   `generate_data`: Generates the datasets
-   `train`: Trains the model (requires the generated datasets)
-   `test`: Test the model (requires the trained model)
-   `clean`: Remove datasets and models
