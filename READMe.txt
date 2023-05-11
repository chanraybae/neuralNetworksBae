For the purposes of the Part 3, the script "test.py" is the appropriate script to run. The other "main.py" serves a dynamic test that works only after waiting a while for
the program to recognize the camera feed sign. 

- A justification of your neural network architecture (number and type of layers, loss function, optimization algorithm, etc.) and -- if that applies -- 
non-NN feature extractor:
In the program test.py, the neural network architecture implemented is a Convolutional Neural Network, which was chosen due to its familiarity in image
analysis and data handling. Image classification was important to the basis of the project itself. There were (13) layers in total, as follows:
(1)In the input layer, grayscale images through preprocessing are
gray-scaled in a (64,64,1) basis for (H, W, Color Channels). Switching to grayscale seemed to increase some accuracy and the computation time, as the slow
time it takes for the program to complete training and evaluation is still quite long. 
(3)In the convolutional type, there are three layers with 32, 64, and 128 filters to have the neural network learn patterns and features in locality. I 
researched the sliding window approach through various YouTube videos to implement the analysis of the input images. 
(3)With respect to the maxpooling layers respective to each convolutional, they are added to downsample and reduce the computational complexity in learning.
(3) The BatchNormalization layers are added to help reduce internal covariate shift and allows for higher learning rates, although I set mine in the program very 
low to increase the accuracy by a hefty margin.
(1)The flatten layer converts the output of the final convolutional layer into a one-dimensional array.
(1) A dense fully-connected layer with 256 units and a ReLU activation function is used to learn the high-level features and patterns. A dropout layer with a rate 
of 0.4 was added to prevent overfitting by randomly dropping a fraction of the input units during the training process. 
(1) In the output layer, there is a dense layer with 26 units and a softmax activation function to convert the output into a probability distribution over the 
English-respective 26 letters (classes) in ASL.

The loss function implemented is categorical cross-entropy loss, which matches in this project for multi-class classification tasks; it measures the 
dissimilarity between the predicted probability distribution and the true probability distribution of the target classes.

Similar to our Practical, The Adam optimization algorithm is used to train the neural network due to its adaptive learning rate 
capabilities, which lead to faster convergence and improved performance compared to traditional stochastic gradient descent. Efficiency was an issue throughout
the process of developing this project.

The non-NN feature extractor I used was the one that Professor Czajika recommended, which was the MediaPipe hand detector to isolate the hand better from]
the background, which I used to crop the image. As aforementioned, pre-processing was also involved to make the input grayscale.

- A classification accuracy achieved on the training and validation sets. That is, how many samples were classified correctly and how many of them were 
classified incorrectly (It is better to provide a percentage instead of numbers):

During first runs, the accuracy was very low due to lack of filters and an extremely unoptimal learning rate. However, in the end, I was able to attain a 
test accuracy output of "Test accuracy: 0.7377049326896667" (~74%) from the script, which I optimized from less than 50%. I hope to optimize it for the final
rendition of the project. 53 out of the 71 test file images were identified correctly in a final run. During the training epochs, the accuracy was hitting 
rates of 0.80 to over 0.90 (max 92%) with validation topping at 69%, so I need to further optimize my learning rate and hyperparameters, such as the batch size and dropout rate. 

- A short commentary related to the observed accuracy and ideas for improvements (to be implemented in the final solution). For instance, if you see almost 
perfect accuracy on the training set, and way worse on the validation set, what does it mean? Is it good? If not, what do you think you could do to improve 
the generalization capabilities of your neural network?

I could improve it by applying data augmentation to the code itself by implementing ImageDataGenerator to the program to add to the training set, applying 
rotations and noise to help it classify during high noise. I think the accuracy should hit at least 80%, so that is what I hope for by the end of the project
submission. Additionally, I could alter the learning rate, as mine seems too low, although that helped prevent overfitting. I will find a better
ratio for the optimizer/learning rate by the end of the project - I could also use a pre-trained model for transfer learning to further bolster
the project. I saw one online discussing their implementation with a different data set and neural network architecture. Additionally as I mentioned above,
I need to further optimize my learning rate and hyperparameters to see if overfitting continues to be an issue and how I can resolve it.
