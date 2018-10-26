import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_trains = X.shape[0]
  num_labels = W.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in range(num_trains):
        scores = X[i].dot(W)
        
        scores -= scores.max() # to avoid overflow problems
        exp_scores = np.exp(scores)
        sum_exp_scores = np.sum(np.exp(scores))        
        correct_prob = -np.log(exp_scores[y[i]] / sum_exp_scores)
        loss += correct_prob
        
        for j in range(num_labels):
            if j == y[i]:
                dW[:, j] += ((exp_scores[j] / sum_exp_scores) - 1) * X[i]
            else:
                dW[:, j] += (exp_scores[j] / sum_exp_scores) * X[i]
  
  # calculate the loss and grad on the whole training set
  # regularization is also taken into account
  loss /= num_trains
  loss += reg * np.sum(W ** 2)

  dW /= num_trains
  dW += 2 * reg * W
        
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
    
  num_trains = X.shape[0]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  row_cor = np.arange(num_trains)
  col_cor = y
  # calculate and reduce scores to avoid overflow, 
  # keepdims to reserve shape of (1, -1)
  scores = X.dot(W)   
  scores -= np.amax(scores, axis = 1, keepdims = True)
  
  # calculate probs
  exp_scores = np.exp(scores)
  sum_exp_scores = np.sum(exp_scores, axis = 1, keepdims = True).reshape((-1, 1))  
  probs = exp_scores / sum_exp_scores

  # calculate loss of each correct class   
  loss_correct = -np.log(probs)[row_cor, col_cor].reshape((-1, 1))  
    
  # calculate loss by taking regularization into account
  loss = np.mean(loss_correct)
  loss += reg * np.sum(W ** 2)
  
  # calculate grad
  dProbs = probs
  dProbs[row_cor, col_cor] -= 1
  dW = X.T.dot(dProbs)
  dW /= num_trains
  dW += 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

