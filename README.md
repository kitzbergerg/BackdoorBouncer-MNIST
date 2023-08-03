# Backdoor attack & prevention

This project aims to demonstrate a simple backdoor attack on the MNIST dataset following the approach in [this paper](https://arxiv.org/pdf/1811.00636.pdf).

In a second step the poisoned data should be removed from the training data.
This is done by using outlier detection.

The cleaned data is then used to train a new model which should ignore backdoor attacks.

## Structure

-   `makefile`: Contains commands to generate data, train and test the model, ...
-   `data/`: Contains the data
-   `model/`: Contains the trained models
-   `src/`: Contains the actual python scripts. The scripts are supposed to be executed from the root folder using the makefile.
    -   `data.py`: Definition of the custom poisoned data
    -   `data_generate.py`: Execute to generate the datasets
    -   `data_outlier_calculation.py`: Performs outlier detection
    -   `data_outlier_functions.py`: Contains helper functions for outlier detection
    -   `model.py`: Definition of the model
    -   `model_train.py`: Execute to train the model
    -   `model_test.py`: Execute to test the model
    -   `model_feed_forward.py`: Evaluates the models output for given input data. Also saves layer representations
-   `visualization/`: Contains tools to visualize the data

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

## Data flow

1. `make data_generate` loads the MNIST data.  
   This command generates:
    - `data/MNIST/modified/train.pth`: Training data containing poisoned elements
    - `data/MNIST/original/test.pth`: Test data containing no poisoned elements
    - `data/MNIST/modified/test.pth`: Test data consisting only of poisoned elements (bottom right pixel is white and label is always 7)
2. `make train` then trains a model on the poisoned dataset.  
   This commands generates:
    - `model/poisoned.pth`: State of the trained poisoned model.
3. `make test` evaluates the model on the clean and poisoned test data.
4. `make feed_forward` saves the models outputs when feed-forwarding the poisoned training data. It additionally saves the values of the second layer in the neural net (which represent high level features).  
   This commands generates:
    - `data/feed_forward_output.pkl`: State of the poisoned model.
5. `make outlier_calculation` performs outlier detection and generates a new dataset out of the poisoned one with the bad data removed (currently doesn't work).  
   This commands generates:
    - `data/MNIST/filtered/train.pth`: Training data with poisoned elements removed.
6. `make train_new` trains a new model using the filtered data.  
   This commands generates:
    - `model/filtered.pth`: State of the trained clean model.
7. `make test_new` evaluates the model on the clean and poisoned test data.
