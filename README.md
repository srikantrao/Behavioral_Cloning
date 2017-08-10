# Behavioral Cloning – Implementation Description 

1. Running the Code
> python3.5 drive.py model.json


2. Additional modifications to drive.py from base python file - 

	1. Pre-processing step – The images are resized to mimic the training process. The pre-processing steps are discussed in detail in the next section.

	2. Throttle control – In order to make sure that the turns are smooth, the throttle has a max value of 0.2 and a min value.
	
throttle = 1 /(1 + K * steering_angle) * throttle_nom

This makes sense since we want to reduce the acceleration (and thereby speed) when going into turns to mimic human (and sensible ?) driving.

3. Neural Network Architecture:

Three different architectures were explored 

1. NVIDIA paper based architecture – It consisted of 4 layers of Conv2D with dropout of 0.5,RELU activation and final depth of 64. This was followed by 3 Fully connected linear layers with dropout of 0.5 and RELU activation for the first two layers.

Drawbacks – This was a Network with a lot of training parameters  and problems of over-fitting were seen initially. I used a low value Regularizer (L2(0.00001)) to avoid this but still was not able to get good performance with this. An additional practical drawback was the long training time ( 2.5 hours). 

2. Tranfer learning using VGG19  - Transfer learning with VGG19 model trained on ‘ImageNet’ was also analyzed. There were 4 additional Fully Connected layers after the VGG19 model with dropout and RELU activation.

I believe I can achieve good performance with this model based on what I have learnt later on – I am going to circle back to this model and try to develop a working model as an extension. 
I  think that the earlier attempt did not work because I had a very large weight matrices and it took a long time to optimize and hence I did not use enough samples ( tried with 20K samples). 



FINAL MODEL THAT IS USED IN THIS SUBMISSION

This model is a close cousin of the comma.ai steering angle training model 

The architecture is as follows  - All layers used were from keras.layers 

1. Lambda layer for normalization of pixel values between (-1.0, 1.0)
2. Convolutional 2D 8x8 filter with a filter depth of 16 with 4x4 sub-sampling. The output shape therefor is [20,80,16]
3. Exponential Linear Unit** - Activation layer 
4. Convolutional 2D 5x5 filter with a depth of 32 with 2x2 sub-sampling.  The output shape is
[10,40,32]
4.  Exponential Linear Unit** - Activation layer
5. Convolutional 2D 5x5 filter with a depth of 64 with 2x2 sub-sampling.  The output shape is
[5,20,64]
6. Flatten – A flatten layer is used to reshape the layer to a 1D layer for linear optimization using fully connected layers.  The output is now a 1D row of 6400 features.
7. A Dropout layer is used with fraction of 0.5 to avoid overfitting.
8. Exponential Linear Unit** - Activation layer  
9. Dense keras layer – linear weights with a matrix of dim (6400 x 512) + 512 for bias 
10. A Dropout layer is used with fraction of 0.5 to avoid overfitting.
11. Exponential Linear Unit** - Activation layer  
12. Dense keras layer – linear weights with a matrix of dim (512 x 128) + 128 for bias
13. Exponential Linear Unit** - Activation layer
14. Fully connected layer with linear weights of dim (128 x 1) + 1 for bias.


Optimization Specifics – Since this is a regression problem, a mean square error loss was chosen for optimization. 

Adam Optimizer was used with default Beta1 and Beta2 values. 

I experimented around with the learning rate and I saw that the best learning rate was actually quite low ( 0.0001).  I experimented in steps of lr/3 to achieve the sweetspot and looked at the training and validation loss values at the end of 10 epochs. 

The optimization was run for 10 epochs on 25K samples. 

Data specifics – The data consisted of three disparate sets which were combined. 
Dataset 1 – Udacity training dataset 
Dataset 2 – Turn specific dataset
Dataset 3 – Turn after bridge + recovery dataset 


A few examples of the dataset helpful in training the turns are included in the attached .zip file. 



Exponential Linear Unit 

The exponential linear Unit is an activation with two salient features 

1. The identity function for positive values ensures that the gradient does not vanish to zero. This is shared with other activation functions such as ReLU and leaky ReLU.
2. It also allows for negative values making sure that the mean activation value is close to zero. This is not accomplished by ReLU but is still accomplished by a leaky ReLU. The ELU activation function does a better job of it.


ELU activation function formula - 

The exponential linear unit (ELU) with 0 < α is 
f(x) =  x                        if x > 0 
f(x) = α (exp(x) − 1)     if x ≤ 0 

The derivative used for the gradient is given by - 

f’(x) =  1                       if x > 0 
f’(x) =   f(x) + α            if x ≤ 0

Reference - arXiv:1511.07289 [cs.LG]
