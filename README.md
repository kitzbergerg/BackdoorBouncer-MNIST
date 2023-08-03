# Backdoor attack & prevention

This project aims to demonstrate a simple backdoor attack on the MNIST dataset following the approach in [this paper](https://arxiv.org/pdf/1811.00636.pdf).

In a second step the poisoned data should be removed from the training data.
This is done by using outlier detection.

The cleaned data is then used to train a new model which should ignore backdoor attacks.

## Structure

-   `makefile`: Contains commands to generate data, train and test the model, ...
-   `data/`: Contains the data
-   `model/`: Contains the trained models
-   `visualization/`: Contains tools to visualize the data
-   `data.py`: Definition of the custom poisoned data
-   `data_generate.py`: Execute to generate the datasets
-   `data_outlier_calculation.py`: Performs outlier detection
-   `data_outlier_functions.py`: Contains helper functions for outlier detection
-   `model.py`: Definition of the model
-   `model_train.py`: Execute to train the model
-   `model_test.py`: Execute to test the model
-   `model_feed_forward.py`: Evaluates the models output for given input data. Also saves layer representations

## Makefile

-   `all`: Run the whole project from start to finish (run `setup` first)
-   `clean`: Remove datasets and models
-   `setup`: Create directories
-   `data_generate`: Generates the datasets
-   `train`: Trains the model (requires the generated datasets)
-   `test`: Test the model (requires the trained model)
-   `feed_forward`: Saves the output of the model + the representations for the second to layer for the training data (requires the generated datasets and model)
-   `outlier_calculation`: Performs outlier detection on the feed-forward data and save new training data with removed outliers (requires the feed forwards data)
-   `train_new`: Trains a new model on outlier free dataset (requires the filtered dataset)
-   `test_new`: Test the new model (requires the model trained on outlier free data)
