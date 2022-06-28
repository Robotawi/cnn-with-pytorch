# Convolutional Neural Networks (CNN)

## Concepts: 
Convolutional Neural Networks (CNN) is the type of deep neural networks (DNN) that is most powerful in image processing tasks, such as sorting images into groups.

While DNN uses many fully-connected layers, CNN contains mostly convolutional layers. In its simplest form, CNN is a network with a set of layers that transform an image to a set of class probabilities. 

A CNN first takes in an input image and then passes it through these layers. There are a few different types of layers, and we'll start by learning about the most commonly used layers: convolutional, pooling, and fully-connected layers.


## Convolutional Layer
The first layer in this network, that processes the input image directly, is a convolutional layer.

- A convolutional layer takes in an image as input.
- A convolutional layer, as its name suggests, is made of a set of convolutional filters.
- Each filter is responsible for finding a pattern in the image by extracting a specific kind of feature, like a high-pass filter used to detect edges of an object. This means those are learnable filters.
- The output of a given convolutional layer is a set of feature maps (also called activation maps), which result from applying the filters to the original input image.

### Activation Function
The ReLu function stands for Rectified Linear Unit (ReLU) activation function. This activation function is zero when the input x <= 0 and then linear with a slope = 1 when x > 0. ReLu's. This function and other activation functions are typically placed after a convolutional layer to slightly transform the output so that it's more efficient to perform backpropagation and effectively train the network. The advantage of the ReLu function is that it prevents the vanishing gradient problem. That problem causes the gradient to be too small, hence too small learning steps, slow training, and deteriorated performance. 

## Pooling Layer
After a couple of convolutional layers (+ReLu's), in the VGG-16 network, you'll see a maxpooling layer. [Here](https://pytorch.org/docs/stable/nn.html#pooling-layers) is PyTorch's documentation about pooling layers.

- Pooling layers take in an image (usually a filtered image) and output a reduced version of that image
- More filters means bigger stack and higher dimentionality. The problem with high dimentionality is the need for using more paramters, which will lead to over fitting. **Pooling layers reduce the dimensionality of an input.**
- Maxpooling layers look at areas in an input image (like the 4x4 pixel area pictured below) and choose to keep the maximum pixel value in that area, in a new, reduced-size area.
- Maxpooling is the most common type of pooling layer in CNN's, but there are also other types such as average pooling.

## Fully-Connected Layer

A fully-connected layer's job is to connect the input it sees to a desired form of output. Typically, this means converting a matrix of image features into a feature vector whose dimensions are 1xC, where C is the number of classes. As an example, say we are sorting images into ten classes, you could give a fully-connected layer a set of [pooled, activated] feature maps as input and tell it to use a combination of these features (multiplying them, adding them, combining them, etc.) to output a 10-item long feature vector. This vector compresses the information from the feature maps into a single feature vector.

### Softmax

The very last layer you see in this network is a softmax function. The softmax function, can take any vector of values as input and returns a vector of the same length whose values are all in the range (0, 1) and, together, these values will add up to 1. This function is often seen in classification models that have to turn a feature vector into a **probability distribution**.

Consider the same example again; a network that groups images into one of 10 classes. The fully-connected layer can turn feature maps into a single feature vector that has dimensions 1x10. Then the softmax function turns that vector into a 10-item long probability distribution in which each number in the resulting vector represents the probability that a given input image falls in class 1, class 2, class 3, ... class 10. This output is sometimes called the class scores and from these scores, you can extract the most likely class for the given image. 

### Overfitting
Convolutional, pooling, and fully-connected layers are all you need to construct a complete CNN, but there are additional layers that you can add to avoid overfitting, too. One of the most common layers to add to prevent overfitting is a [dropout layer](http://pytorch.org/docs/stable/nn.html#dropout-layers).

Dropout layers essentially turn off certain nodes in a layer with some probability, p. This ensures that all nodes get an equal chance to try and classify different images during training, and it reduces the likelihood that only a few, heavily-weighted nodes will dominate the process.

Now, you're familiar with all the major components of a complete convolutional neural network, and given some examples of PyTorch code, you should be well equipped to build and train your own CNN's! Next, it'll be up to you to define and train a CNN for clothing recognition!
___

## Practice 

## Define a Network Architecture
The various layers that make up any neural network are documented [here](https://pytorch.org/docs/stable/nn.html). For a convolutional neural network, we'll use a simple series of layers:

- Convolutional layers.
- Maxpooling layers.
- Fully-connected (linear) layers.

To define a neural network in PyTorch, we need to create and name a new neural network class, define the layers of the network in the constructor __init__ and define the feedforward behavior of the network that employs those initialized layers in the function forward, which takes in an input image tensor, x.

Note: During training, PyTorch will be able to perform backpropagation by keeping track of the network's feedforward behavior and using autograd to calculate the update to the weights in the network.

This [notebook](./1_Define_layers_in_pytorch.ipynb) shows a simple example of defining a CNN in PyTorch

## Visualizing a dataset 

The first step in any classification problem is to look at the dataset you are working with. This will give you some details about the format of images and labels, as well as some insight into how you might approach defining a network to recognize patterns in such an image set. In this [notebook](./4_Load_and_Visualize_FashionMNIST.ipynb), we load and look at images from the [Fashion-MNIST database](https://github.com/zalandoresearch/fashion-mnist). 

Here is a [link](https://pytorch.org/vision/stable/datasets.html) to the datasets available in PyTorch and how to load them. 

## Training in PyTorch 

After loading the training dataset, our next job will be to define a CNN and train it to classify for example a set of images.

### Loss and Optimizer

To train a model, you'll need to define how it trains by selecting a loss function and optimizer. These functions decide how the model updates its parameters as it trains and can affect how quickly the model converges, as well.

Learn more about [loss functions](http://pytorch.org/docs/master/nn.html#loss-functions) and [optimizers](http://pytorch.org/docs/master/optim.html) in the online documentation.

For a classification problem like this, one typically uses cross entropy loss, which can be defined in code like: `criterion = nn.CrossEntropyLoss()`. PyTorch also includes some standard stochastic optimizers like stochastic gradient descent and Adam. You're encouraged to try different optimizers and see how your model responds to these choices as it trains.

### Clasisification vs. Regression

The loss function you should choose depends on the kind of CNN you are trying to create; **cross entropy is generally good for classification tasks**, but you might choose a different loss function for, say, a regression problem that tried to predict (x,y) locations for the center or edges of clothing items instead of class scores.

### Training the Network
Typically, we train any network for a number of epochs or cycles through the training dataset

Here are the steps that a training function performs as it iterates over the training dataset:

1. Prepares all input images and label data for training
2. Passes the input through the network (forward pass)
3. Computes the loss (how far is the predicted classes are from the correct labels)
4. Propagates gradients back into the network’s parameters (backward pass)
Updates the weights (parameter update)

The training function repeats thoses steps until the average loss has sufficiently decreased.

This [notebook](./5_1_Classify_FashionMNIST_exercise.ipynb) shows the details about how to train and test a CNN for clothing classification. Here is the [first solution](./5_2_Classify_FashionMNIST_solution_1.ipynb), and here is the [second solution](./5_3_Classify_FashionMNIST_solution_2.ipynb) that uses Dropout and Momentum. 

### Dropout and Momentum

We can obtain improved models for clothing classification with including two features to the neural network:

1. An additional dropout layer.
2. A momentum term in the optimizer: stochastic gradient descent.

Why are the improvements? 

### Dropout

Dropout randomly turns off perceptrons (nodes) that make up the layers of our network, with some specified probability. It may seem counterintuitive to throw away a connection in our network, but as a network trains, some nodes can dominate others or end up making large mistakes, and dropout gives us a way to balance our network so that every node works equally towards the same goal, and if one makes a mistake, it won't dominate the behavior of our model. You can think of dropout as a technique that makes a network resilient; it makes all the nodes work well as a team by making sure no node is too weak or too strong. In fact it makes me think of the Chaos Monkey tool that is used to test for system/website failures.

have a look at the PyTorch [dropout documentation](https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html), to see how to add these layers to a network.

### Momentum

When you train a network, you specify an optimizer that aims to reduce the errors that your network makes during training. The errors that it makes should generally reduce over time but there may be some bumps along the way. Gradient descent optimization relies on finding a local minimum for an error, but it has trouble finding the global minimum which is the lowest an error can get. So, we add a momentum term to help us find and then move on from local minimums and find the global minimum!

This [exercise repo](https://github.com/udacity/CVND_Exercises/tree/master/1_5_CNN_Layers) shows multiple solutions to the clothing classification training challenge. 


## Visualizing Different Layers Outputs

A well-known criticism of the NN field was that the intermediate output of the connected layers was not understood. This made it hard to interpret why NN really work. We can now visualize the output of the convolutional layers as shown in this [example](./2_Convolutional_layer_visualization.ipynb) as well as the polling layers as shown [here](./3_Pooling_layer_visualization.ipynb).

### Visualizing layers activation maps

More into understanding the inner workings of neural networks (not to be thought of as a black box, given some input, they learn to produce some output). CNN's are actually learning to recognize a variety of spatial patterns and we can visualize what each convolutional layer has been trained to recognize by looking at the weights that make up each convolutional kernel and applying those one at a time to a sample image. These techniques are called **feature visualization** and they are useful for grasping the internal nature/working of a CNN.

After extracting the kernels/filters weights, we can use OpenCV's filter2D function to apply these filters to a sample test image and produce a series of activation maps as a result. This is done in this [notebook](./6_Feature_viz_for_FashionMNIST.ipynb). 