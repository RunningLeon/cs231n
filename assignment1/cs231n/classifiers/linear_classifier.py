from __future__ import print_function

import numpy as np
from cs231n.classifiers.linear_svm import *
from cs231n.classifiers.softmax import *
from past.builtins import xrange


class LinearClassifier(object):

  def __init__(self):
    self.W = None

  def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100,
            batch_size=200, validate=False, train_validate_split=0.1, verbose=False, print_every=100):
    """
    Train this linear classifier using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) containing training data; there are N
      training samples each of dimension D.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c
      means that X[i] has label 0 <= c < C for C classes.
    - learning_rate: (float) learning rate for optimization.
    - reg: (float) regularization strength.
    - num_iters: (integer) number of steps to take when optimizing
    - batch_size: (integer) number of training examples to use at each step.
    - verbose: (boolean) If true, print progress during optimization.

    Outputs:
    A list containing the value of the loss function at each training iteration.
    """
    num_train, dim = X.shape
    num_classes = np.max(y) + 1 # assume y takes values 0...K-1 where K is number of classes
    # train validation split 
    if validate:
      num_valid = int(num_train * train_validate_split)
      valid_idx = np.random.choice(num_train, num_valid, replace=False)
      train_idx = list(set(range(num_train)) - set(valid_idx))
      X_valid, y_valid = X[valid_idx], y[valid_idx]
      X, y = X[train_idx], y[train_idx]
      num_train -= num_valid
    if self.W is None:
      # lazily initialize W
      self.W = 0.001 * np.random.randn(dim, num_classes)
    # Init history that contain loss and acc on train and validation
    history = {}
    meric_names = ['train loss', 'train acc', 'valid loss', 'valid acc']
    for me in meric_names:
      history.setdefault(me, [])
    
    # Run stochastic gradient descent to optimize W
    for it in xrange(num_iters):
      X_batch = None
      y_batch = None

      #########################################################################
      # TODO:                                                                 #
      # Sample batch_size elements from the training data and their           #
      # corresponding labels to use in this round of gradient descent.        #
      # Store the data in X_batch and their corresponding labels in           #
      # y_batch; after sampling X_batch should have shape (dim, batch_size)   #
      # and y_batch should have shape (batch_size,)                           #
      #                                                                       #
      # Hint: Use np.random.choice to generate indices. Sampling with         #
      # replacement is faster than sampling without replacement.              #
      #########################################################################
      batch_idx = np.random.choice(num_train, batch_size, replace=True)
      X_batch =X[batch_idx]
      y_batch = y[batch_idx]
      #########################################################################
      #                       END OF YOUR CODE                                #
      #########################################################################

      # evaluate loss and gradient on train dataset
      train_loss, grad = self.loss(X_batch, y_batch, reg)
      history['train loss'].append(train_loss)

      # evaluate on validation dataset

      if validate:
        valid_loss, _ = self.loss(X_valid, y_valid, reg)
        history['valid loss'].append(valid_loss)
        # compute valid acc
        valid_pred = self.predict(X_valid)
        valid_acc = self.compute_accuracy(y_valid, valid_pred)
        history['valid acc'].append(valid_acc)
        # compute train acc
        train_pred = self.predict(X)
        train_acc = self.compute_accuracy(y, train_pred)
        history['train acc'].append(train_acc)


      # perform parameter update
      #########################################################################
      # TODO:                                                                 #
      # Update the weights using the gradient and the learning rate.          #
      #########################################################################
      self.W += - learning_rate * grad
      #########################################################################
      #                       END OF YOUR CODE                                #
      #########################################################################

      if verbose and it % print_every == 0:
        if validate:
          print('iteration %d / %d: loss %f, accuracy %f on train set, loss %f, accuracy %f on valid set' % 
            (it, num_iters, train_loss, train_acc, valid_loss, valid_acc))
        else:
          print('iteration %d / %d: loss %f on train set' % (it, num_iters, train_loss))
          
        
    return history

  def predict(self, X):
    """
    Use the trained weights of this linear classifier to predict labels for
    data points.

    Inputs:
    - X: A numpy array of shape (N, D) containing training data; there are N
      training samples each of dimension D.

    Returns:
    - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
      array of length N, and each element is an integer giving the predicted
      class.
    """
    y_pred = np.zeros(X.shape[0])
    ###########################################################################
    # TODO:                                                                   #
    # Implement this method. Store the predicted labels in y_pred.            #
    ###########################################################################
    scores = X.dot(self.W)
    y_pred = np.argmax(scores, axis = 1)
    ###########################################################################
    #                           END OF YOUR CODE                              #
    ###########################################################################
    return y_pred
  
  def loss(self, X_batch, y_batch, reg):
    """
    Compute the loss function and its derivative. 
    Subclasses will override this.

    Inputs:
    - X_batch: A numpy array of shape (N, D) containing a minibatch of N
      data points; each point has dimension D.
    - y_batch: A numpy array of shape (N,) containing labels for the minibatch.
    - reg: (float) regularization strength.

    Returns: A tuple containing:
    - loss as a single float
    - gradient with respect to self.W; an array of the same shape as W
    """
    pass

  def compute_accuracy(self, y_true=None, y_pred=None):
    """
    Compute accuracy
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean(y_true == y_pred)



class LinearSVM(LinearClassifier):
  """ A subclass that uses the Multiclass SVM loss function """

  def loss(self, X_batch, y_batch, reg):
    return svm_loss_vectorized(self.W, X_batch, y_batch, reg)


class Softmax(LinearClassifier):
  """ A subclass that uses the Softmax + Cross-entropy loss function """

  def loss(self, X_batch, y_batch, reg):
    return softmax_loss_vectorized(self.W, X_batch, y_batch, reg)

