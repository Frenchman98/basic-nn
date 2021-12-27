# Number Recognition Neural Net

A barebone neural network classifier for recognizing numbers from the [MNIST database](http://yann.lecun.com/exdb/mnist/).

## To Run
Using Python 3.7 or greater:

- Open the data directory
- Run the mnist_csv3.py script (courtesy of USC CSCI 561) to convert the mnist images into csv data

Four csv files should then be present in the data directory. Then,

- Change directory back to the root of this repository
- Run NeuralNetwork3.py

## Details

### Structure
- Expects 784 inputs (28x28 pixels in each image)
- 2 hidden layers with 32 neurons each
- 10 output neurons representing the NN's choice of 0 through 9
- Sigmoid activation used on the inner neurons 
- Softmax activation used on the output neurons

### Training
- Weights initialized using Xavier Initialization
- Uses Stochastic Gradient Descent with momentum
- Uses cross-entropy loss
