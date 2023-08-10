# Backdoor attack & prevention

This project aims to demonstrate a simple backdoor attack on the MNIST, CIFAR-10 and GTSRB datasets following the approach in [this paper](https://arxiv.org/pdf/1811.00636.pdf).

In a second step the poisoned data should be removed from the training data.
This is done by using outlier detection.

The cleaned data is then used to train a new model which should ignore backdoor attacks.

## Structure

-   `makefile`: Contains commands to generate data, train and test the model, ...
-   `data/`: Contains the data
-   `model/`: Contains the trained models
-   `src/`: Contains the python scripts. The scripts are supposed to be executed from the root folder using the makefile.
    -   `config.py`: Configuration of which dataset to use, which model is used for which dataset, how to modify the dataset, ...
    -   `data.py`: Definition of the custom poisoned data
    -   `data_generate.py`: Execute to generate the poisoned dataset
    -   `data_outlier_calculation.py`: Performs outlier detection
    -   `data_outlier_functions.py`: Contains helper functions for outlier detection
    -   `model.py`: Definition of the model
    -   `model_train.py`: Execute to train the model
    -   `model_test.py`: Execute to test the model
    -   `model_feed_forward.py`: Evaluates the models output for given input data. Also saves layer representations
-   `visualization/`: Contains tools to visualize the data

## Makefile

-   `all`: Run the whole project from start to finish (run `setup` first)
-   `setup`: Create directories
-   `clean`: Remove datasets and models
-   `data_generate`: Generates the training dataset
-   `train`: Trains the model (requires the generated training data)
-   `test`: Test the model (requires the trained model)
-   `feed_forward`: Saves the output of the model + the representations for the second to layer for the training data (requires the generated datasets and model)
-   `outlier_calculation`: Performs outlier detection on the feed-forward data and save new training data with removed outliers (requires the feed-forward data)
-   `train_new`: Trains a new model on the outlier free data (requires the filtered data)
-   `test_new`: Test the new model (requires the model trained on outlier free data)
-   `feed_forward_new`: Same as `feed_forward`, but uses newly trained model

## Data flow

1. `make data_generate` generates the training data containing additional UUIDs and positions where poisoned elements should be placed.  
   This command generates:
    - `data/modified/train.pth`: Generated training data (does not contains poisoned images, they will be applied later on)
2. `make train` then trains a model on the poisoned dataset.  
   This commands generates:
    - `model/poisoned.pth`: State of the trained poisoned model.
3. `make test` evaluates the model on the clean and poisoned test data.
4. `make feed_forward` saves the models outputs when feed-forwarding the poisoned training data. It additionally saves the values of the second to last layer in the neural net (which represent high level features).  
   This commands generates:
    - `data/feed_forward_output.pkl`: State of the poisoned model.
5. `make outlier_calculation` performs outlier detection and generates a new dataset out of the poisoned one with the bad data removed.  
   This commands generates:
    - `data/filtered/train.pth`: Training data with poisoned elements removed.
6. `make train_new` trains a new model using the filtered data.  
   This commands generates:
    - `model/filtered.pth`: State of the trained clean model.
7. `make test_new` evaluates the model on the clean and poisoned test data.
