from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """
    
    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Width/height of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian centered at 0.0   #
        # with standard deviation equal to weight_scale; biases should be          #
        # initialized to zero. All weights and biases should be stored in the      #
        #  dictionary self.params. Store weights and biases for the convolutional  #
        # layer using the keys 'W1' and 'b1'; use keys 'W2' and 'b2' for the       #
        # weights and biases of the hidden affine layer, and keys 'W3' and 'b3'    #
        # for the weights and biases of the output affine layer.                   #
        #                                                                          #
        # IMPORTANT: For this assignment, you can assume that the padding          #
        # and stride of the first convolutional layer are chosen so that           #
        # **the width and height of the input are preserved**. Take a look at      #
        # the start of the loss() function to see how that happens.                #                           
        ############################################################################
        ## initialize weights and biases for ConvNet layer ##
        self.params['W1'] = np.random.normal(scale = weight_scale,
                                             size = (num_filters, 3, filter_size, filter_size)) 
                                             # 3 is the number of image channels
        self.params['b1'] = np.zeros(num_filters)
        
        # compute size of x after gone through ConvNet and Max_Pooling
        input_H, input_W = input_dim[1], input_dim[2]
        
        # for ConvNet
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}
        conv_H = 1 + (input_H + 2 * conv_param['pad'] - filter_size) // conv_param['stride']
        conv_W = 1 + (input_W + 2 * conv_param['pad'] - filter_size) // conv_param['stride']
        
        # for Max_Pooling
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
        max_pool_H = 1 + (conv_H - pool_param['pool_height']) // pool_param['stride']
        max_pool_W = 1 + (conv_W - pool_param['pool_width']) // pool_param['stride']
        
        ## initialize weights and biases for 2 layers in fully connected net ##
        # for 1st FCNet (hidden layer)
        input_size = num_filters * max_pool_H * max_pool_W
        self.params['W2'] = np.random.normal(scale = weight_scale,
                                             size = (input_size, hidden_dim)
                                            )
        self.params['b2'] = np.zeros(hidden_dim)
        
        # for 2nd FCNet (output layer)
        self.params['W3'] = np.random.normal(scale = weight_scale,
                                             size = (hidden_dim, num_classes)
                                            )
        self.params['b3'] = np.zeros(num_classes)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        # The code here is modified a bit as we are reusing the implementation in FullyConnectedNet
        # original count is W1, W2, W3
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        # Padding and stride chosen to preserve the input spatial size
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        #                                                                          #
        # Remember you can use the functions defined in cs231n/fast_layers.py and  #
        # cs231n/layer_utils.py in your implementation (already imported).         #
        ############################################################################
        fwd_history, cache_history = {}, {}
        
        # feed forward for ConvNet layer
        scores = X # prepare scores for ConvNet layer
        alias = 1
        
        fwd, cache = conv_forward_fast(scores, W1, b1, conv_param)
        fwd_history['conv%d' % alias], cache_history['conv%d' % alias] = fwd, cache
        
        fwd, cache = max_pool_forward_fast(fwd, pool_param)
        fwd_history['max_pool%d' % alias], cache_history['max_pool%d' % alias] = fwd, cache
        
        # Compute affine - relu (1st FCNet layer)
        scores = fwd
        alias += 1
        
        fwd, cache = affine_forward(scores, W2, b2)
        fwd_history['affine%d' % alias], cache_history['affine%d' % alias] = fwd, cache
        
        fwd, cache = relu_forward(fwd)
        fwd_history['ReLu%d' % alias], cache_history['ReLu%d' % alias] = fwd, cache
        
        # Compute affine (2nd FCNet layer)
        scores = fwd
        alias += 1
        
        # computer affine from the output layer
        fwd, cache = affine_forward(scores, W3, b3)
        fwd_history['affine%d' % alias], cache_history['affine%d' % alias] = fwd, cache
        
        # scores produced from the output layer
        scores = fwd
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        loss, dout = softmax_loss(scores, y)
        # add regularization
        for layer in range(3): # 3 is num_layers
          alias = layer + 1
          loss += 0.5  * self.reg * np.sum(self.params['W%d' % alias] ** 2)
          
        ## compute backpropagation grads ##
        dx, dW, db = {}, {}, {}
        
        # grad for output layer
        alias = 3
        dout, dW[alias], db[alias] = affine_backward(dout, cache_history['affine%d' % alias])
        
        # grad for the hidden layer
        alias = 2
        dout = relu_backward(dout, cache_history['ReLu%d' % alias])
        dout, dW[alias], db[alias] = affine_backward(dout, cache_history['affine%d' % alias])
        
        # grad for ConvNet layer
        alias = 1
        dout = max_pool_backward_fast(dout, cache_history['max_pool%d' % alias])
        _, dW[alias], db[alias] = conv_backward_strides(dout, cache_history['conv%d' % alias])

        # add regularization
        for layer in range(3): # 3 is num_layers
            alias = layer + 1
            grads['W%d' % alias] = dW[alias] + self.reg * self.params['W%d' % alias]
            grads['b%d' % alias] = db[alias]
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
