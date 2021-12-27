
import numpy as np
import csv
import time
from sys import argv
from os.path import join
import matplotlib.pyplot as plt

import matplotlib.style
matplotlib.style.use('dark_background')


class MyNeuralNetwork:

    I = np.eye(10, 10)

    def __init__(self) -> None:
        """
        Initializes the neural network's shape and various extra variables
        """
        
        # Basic
        self.visual = False
        self.start_time = time.time()
        self.rng = np.random.default_rng()

        # Hyperparameters
        self.epochs = 100
        self.batch_size = 16
        self.alpha = 0.00008
        self.momentum = self.alpha / 4

        # SGD with momentum
        self.old_db_out = self.old_db_hl2 = self.old_db_hl1 = 0
        self.old_dw_out = self.old_dw_hl2 = self.old_dw_hl1 = 0

        # --- Structural ---

        # Sizes
        self.input_length = 28*28
        self.hl1_length = 32
        self.hl2_length = 32

        # Layers
        self.hl1 = None  # Layer 1 Shape: (batch_size, hl1_length)
        self.hl2 = None  # Layer 2 Shape: (batch_size, hl2_length)
        self.out = None  # Output Shape: (batch_size, 10)

        # Weights (Initialized using Xavier initialization)
        self.w_hl1 = (2 * self.rng.random((self.input_length, self.hl1_length)) - 1) * np.sqrt(1 / self.input_length)  # input neuron -> each layer 1 weight to apply
        self.w_hl2 = (2 * self.rng.random((self.hl1_length, self.hl2_length)) - 1) * np.sqrt(1 / self.hl1_length)  # layer 1 neuron -> each layer 2 weight to apply
        self.w_out = (2 * self.rng.random((self.hl2_length, 10)) - 1) * np.sqrt(1 / self.hl2_length)  # layer 2 neuron -> weights for each of the 10 possible outputs
        
        # Biases
        self.b_hl1 = 2 * self.rng.random(self.hl1_length) - 1
        self.b_hl2 = 2 * self.rng.random(self.hl2_length) - 1
        self.b_out = 2 * self.rng.random(10) - 1

    def sigmoid(self, x):
        """
        Computes the sigmoid of all values in x
        """

        return 1 / (1 + np.exp(-x))
    
    def d_sigmoid(self, x):
        """
        Applies the derivative of the sigmoid function 
        onto all elements of x.

        Assumes sigmoid() has already been called on x.
        """

        return x * (1 - x)
    
    def softmax(self, x):
        """
        The softmax function applied on each row of x (because x should be a batch)

        Subtracts max value from all x so that exp(x) doesn't overflow
        """

        tmp = np.exp(x-np.max(x))
        return tmp / np.sum(tmp, axis=1, keepdims=True)

    def forward(self, batch, predict=False):
        """
        Forward passes using a sigmoid activation function 
        on the hidden layers and a softmax function on the 
        output layer.

        Images should be centered using normalize_images()
        before being passed into this function.

        Returns the decisions as a batch_size by 1 array of 
        integers of value 0-9 (the choices) if predict is 
        true
        """
        
        # Forward pass the values from/to each layer and apply the activation functions
        self.hl1 = self.sigmoid(batch.dot(self.w_hl1) + self.b_hl1)
        self.hl2 = self.sigmoid(self.hl1.dot(self.w_hl2) + self.b_hl2)
        self.out = self.softmax(self.hl2.dot(self.w_out) + self.b_out)

        # Make predictions if requested
        if predict or self.visual:
            return self.out.argmax(1)[np.newaxis].T

    def backward(self, batch, expected):
        """
        Backpropagates the given expected values for the batch that was input

        The derivative of loss with respect to the weights can be expressed as

        dl/dw = dl/dbias * dbias/dw

        so, solve for all the dl/dbiases first then use those values to 
        solve for dl/dw, and apply them to the weights and biases.
        """

        # Transform the expected values into arrays
        t_expected = np.array([self.I[x[0]] for x in expected])

        # Propagating from output to HL2
        db_out = self.out - t_expected
        dw_out = self.hl2.T.dot(db_out)
        db_out = db_out.sum(0)

        # Propagating from HL2 to HL1
        db_hl2 = self.d_sigmoid(self.hl2) * db_out.dot(self.w_out.T)
        dw_hl2 = self.hl1.T.dot(db_hl2)
        db_hl2 = db_hl2.sum(0)

        # Propagating from HL1 to the input
        db_hl1 = self.d_sigmoid(self.hl1) * db_hl2.dot(self.w_hl2.T)
        dw_hl1 = batch.T.dot(db_hl1)
        db_hl1 = db_hl1.sum(0)

        # Update all values, scaling by the learning rate
        self.b_out -= self.alpha * db_out + self.momentum * self.old_db_out
        self.b_hl2 -= self.alpha * db_hl2 + self.momentum * self.old_db_hl2
        self.b_hl1 -= self.alpha * db_hl1 + self.momentum * self.old_db_hl1
        self.w_out -= self.alpha * dw_out + self.momentum * self.old_dw_out
        self.w_hl2 -= self.alpha * dw_hl2 + self.momentum * self.old_dw_hl2
        self.w_hl1 -= self.alpha * dw_hl1 + self.momentum * self.old_dw_hl1

        # Keep track of values
        self.old_db_out = db_out
        self.old_db_hl2 = db_hl2
        self.old_db_hl1 = db_hl1
        self.old_dw_out = dw_out
        self.old_dw_hl2 = dw_hl2
        self.old_dw_hl1 = dw_hl1
        
    def train(self, images, labels):
        """
        Trains the network for the given number of epochs, 
        splitting the input data into the given number of batches,
        and using the provided labels for backpropagation.
        """

        num_images = len(images)
        batches = int(np.ceil(num_images / self.batch_size))
        e_results, l_results = [], []  # For display

        epoch_time = time.time()
        epoch = 0

        if self.visual:
            import os
            os.system('cls')

        while epoch < self.epochs:
            correct = 0
            is_this_loss = []  # For display
            expected = []
            batch = []

            # Shuffle the training data to introduce more randomness
            s = self.rng.permutation(len(images))
            s_images = images[s]
            s_labels = labels[s]

            for i in range(batches):
                # Get the range of inputs/labels to get
                start = i * self.batch_size
                end = min(start + self.batch_size, len(images))
                
                # Build the batch and the expected outputs
                batch = s_images[start:end]
                expected = s_labels[start:end]

                # Forward pass the batch
                results = self.forward(batch)

                # Track progress
                if self.visual:
                    is_this_loss.append(np.sum(-expected * np.log(.000000000000001 + self.out)))
                    correct += np.count_nonzero(results == expected)

                    print(f'\rBatch {i} of {batches} in epoch {epoch} of {self.epochs}', end='')

                # Then backpropagate using the expected outcomes
                self.backward(batch, expected)
            
            # Uncomment to do the most amount of epochs possible in 30 min
            # if epoch == 0:
            #     now = time.time()
            #     epoch_time = now - epoch_time
            #     target = self.start_time + 1750  # 1800 is 30 minutes, leave 50 sec for saving, etc

            #     self.epochs = np.floor((target - now)/epoch_time)
            
            epoch += 1
            
            # Add epoch average to list and update UI
            if self.visual:
                e_results.append(correct / num_images)
                l_results.append(np.mean(is_this_loss))
                
                plt.clf()
                plt.plot(list(range(epoch)), e_results, color='orange', label='Accuracy')
                plt.plot(list(range(epoch)), np.array(l_results)/np.max(l_results), color='green', label='  l o s s  ')
                plt.title('Learning Curve')
                plt.xlabel('Epoch')
                plt.ylim(0, 1)
                plt.legend()

                plt.pause(0.01)
    
        if self.visual:
            print(f'\rThe plot is now interactable. Took {time.time() - self.start_time} seconds.')
            plt.show()


def normalize_images(images):
    """
    Attempts to center all images, in order to be more consistent
    """

    # return np.array(images, dtype=np.float64) / 256  # Uncomment to use without normalization

    out = np.zeros_like(images, dtype=np.float64)

    for i, image in enumerate(images):
        tmp = np.array(image, dtype=int).reshape((28, 28)) / 256  # Cast to 0-1 scale
        
        # Get nonzero rows + columns
        x = tmp.sum(1).nonzero()[0]
        y = tmp.sum(0).nonzero()[0]

        # If an array of zeros leave it alone
        if not len(x):
            out[i] = image
            continue
        
        # Target center (13.5) minus the current center to get desired movement
        dx, dy = round(13.5 - (x[0] + x[-1]) / 2), round(13.5 - (y[0] + y[-1]) / 2)
        start_x, end_x = x[0] + dx, x[-1] + dx
        start_y, end_y = y[0] + dy, y[-1] + dy

        # Make fresh image and copy+paste content to center
        img = np.zeros((28, 28))
        img[start_x:end_x, start_y:end_y] = tmp[x[0]:x[-1], y[0]:y[-1]]

        # Flatten and put into the output array
        out[i] = img.flatten()
    
    return out


if __name__ == "__main__":

    nn = MyNeuralNetwork()

    local = True
    out = join('.', 'test_predictions.csv')
    data_dir = join('.', 'data')
    ans = join(data_dir, 'test_label.csv')

    if local:
        # For ease of use

        t_images = join(data_dir, 'train_image.csv')
        t_labels = join(data_dir, 'train_label.csv')
        test = join(data_dir, 'test_image.csv')
    else:
        if len(argv) != 4:
            print(argv)
            print("Must supply 3 commandline arguments in order to run.")
            exit()
        
        t_images, t_labels, test = argv[1], argv[2], argv[3]
    
    # Get all CSV data
    with open(t_images, 'r') as f:
        training_imgs = normalize_images(list(csv.reader(f)))
    with open(t_labels, 'r') as f:
        training_lbls = np.array(list(csv.reader(f)), dtype=int)
    with open(test, 'r') as f:
        test_imgs = normalize_images(list(csv.reader(f)))

    # Train the network
    nn.train(training_imgs, training_lbls)

    # Get the results from the test data
    results = nn.forward(test_imgs, predict=True)

    if local:
        with open(ans, 'r') as f:
            labels = np.array(list(csv.reader(f)), dtype=int)

        print(f'Correctly classified {np.count_nonzero(results == labels)} out of {len(labels)} test images.')

    # Write to output file
    with open(out, 'w') as f:
        w = csv.writer(f)
        w.writerows(results)
