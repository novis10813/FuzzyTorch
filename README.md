# FuzzyTorch

## Brief Description
This repository presents an implementation of the hierarchical fused deep neural network model proposed by Deng et al. (2017). It is applied to a COVID-19 dataset from Kaggle for regression purposes.

## Model Architecture
In this code, the model comprises the following components:
* The Fuzzy Component
    1. FuzzyLayer:
        This layer obtains fuzzy logic representations by passing each dimension of the input vector through a fuzzy membership function, utilizing Gaussian distribution.

    2. FuzzyRuleLayer:
        Element-wise multiplication is performed on all fuzzy logic representations.

    3. MFLayer:
        A wrapper layer that encapsulates FuzzyLayer and FuzzyRuleLayer. Please note that in the `forward` method, I divided the input tensors by features to calculate logic representations individually.

* The Dense Layer Component
    1. DenseLayer:
        Consists of two `nn.Linear` layers.

* The Fusion Layer Component
    Combines the output of the fuzzy component and the dense layer component through concatenation. After concatenation, the two representations are fused using another `nn.Linear` layer.

Note: I switched the activation function from sigmoid to ReLU because using sigmoid was resulting in slow convergence.