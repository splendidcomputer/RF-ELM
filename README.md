# Random Fourier Extreme Learning Machine with $L_2^1$-norm regularization

This is an implementation of the Random Fourier Extreme Learning Machine algorithm in MATLAB, with $L_2^1$-norm regularization. 

## Dataset

The [Iris dataset](https://archive.ics.uci.edu/ml/datasets/Iris) is used as an example dataset for this implementation. The input data (`irisInputs`) is a 150 x 4 matrix containing the attributes of 150 iris flowers, while the target data (`irisTargets`) is a 150 x 3 matrix containing the corresponding class labels for the flowers.

## Parameters

- `nEpoch`: the number of epochs for training the ELM.
- `maxL`: the maximum number of neurons in the hidden layer.
- `stepL`: the hidden layer growth step size.
- `delta`: the weights matrix variance, which could be in the range $(2^{-24}, 2^{8})$.
- `C`: the regularization term, which could be in the range $(2^{-8}, 2^{24})$.
- `nO`: the number of neurons in the output layer.
- `nBetaUpdate`: the number of times that the output matrix is updated.
- `K`: the number of folds for K-Fold cross-validation.

## Implementation

The implementation is divided into several sections:

- **Load Dataset**: Loads the Iris dataset.
- **Initialization**: Initializes the variables and matrices used in the algorithm.
- **ELM Outer Loop**: Performs the outer loop of the ELM algorithm for different numbers of neurons in the hidden layer.
- **K-Fold Cross Validation**: Partitions the data into K folds for cross-validation.
- **Train Phase**: Creates the train phase matrix H.
- **Test Phase**: Creates the test phase matrix H.
- **Output Weight Matrix Calculation**: Calculates the output weight matrix beta.
- **Accuracy Calculation**: Calculates the accuracy of the ELM for different numbers of hidden neurons.
- **Results**: Plots the accuracy results for both the training and test sets.

## Running the Code

- Load the `iris_dataset.mat` file in the MATLAB workspace.
- Run the `RF_ELM.m` file.

## References

- Huang, G.-B., Zhu, Q.-Y., & Siew, C.-K. (2006). Extreme learning machine: theory and applications. Neurocomputing, 70(1-3), 489-501.

- Zhou, Y., Feng, J., & Zhao, Z. (2015). $L_2^1$-norm regularized extreme learning machine. Neurocomputing, 150, 178-185.
