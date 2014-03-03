
import cPickle
import gzip
import os
import sys
import time

import numpy

import pylab

import theano
import theano.tensor as T

import scipy.io.wavfile as wv

from ift6266h14_wt.utils import load_data_npz


class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = numpy.asarray(rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)), dtype=theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (lin_output if activation is None
                       else activation(lin_output))
        # parameters of the model
        self.params = [self.W, self.b]


class OutputLinear(object):
    def __init__(self, input, n_in, n_out):
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(value=numpy.zeros((n_in, n_out),
                               dtype=theano.config.floatX),
                               name='W', borrow=True)
        # initialize the baises b as a vector of n_out 0s
        self.b = theano.shared(value=numpy.zeros((n_out,),
                               dtype=theano.config.floatX),
                               name='b', borrow=True)

        # compute vector of real values in symbolic form
        self.y_pred = T.reshape(T.dot(input, self.W) + self.b, (input.shape[0],))

        # parameters of the model
        self.params = [self.W, self.b]

    def errors(self, y):

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                            ('y', y.type, 'y_pred', self.y_pred.type))
        else:
            return T.mean((self.y_pred - y)**2)


class MLP(object):
    """Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    """

    def __init__(self, rng, input, n_in, n_hidden, n_out):
        """Initialize the parameters for the multilayer perceptron

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_hidden: int
        :param n_hidden: number of hidden units

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie

        """

        # Since we are dealing with a one hidden layer MLP, this will translate
        # into a HiddenLayer with a tanh activation function connected to the
        # LogisticRegression layer; the activation function can be replaced by
        # sigmoid or any other nonlinear function
        self.hiddenLayer = HiddenLayer(rng=rng, input=input,
                                       n_in=n_in, n_out=n_hidden,
                                       activation=T.tanh)

        # The logistic regression layer gets as input the hidden units
        # of the hidden layer
        self.outputLayer = OutputLinear(
            input=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out)

        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = abs(self.hiddenLayer.W).sum() + abs(self.outputLayer.W).sum()

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (self.hiddenLayer.W ** 2).sum() + (self.outputLayer.W ** 2).sum()

        # computing the mean square errors
        self.errors = self.outputLayer.errors

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = self.hiddenLayer.params + self.outputLayer.params

    def save_model(self, filename='params.pkl',
                   save_dir='output_folder'):
        """ Save the parameters of the model """

        print '... saving model'
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        save_file = open(os.path.join(save_dir, filename), 'wb')
        cPickle.dump(self.params, save_file, protocol=cPickle.HIGHEST_PROTOCOL)
        save_file.close()

    def load_model(self, filename='params.pkl',
                   load_dir='output_folder'):
        """ Load the parameters """
        print '... loading model'

        save_file = open(os.path.join(load_dir, filename), 'r')
        params = cPickle.load(save_file)
        save_file.close()

        self.hiddenLayer.W.set_value(params[0].get_value(), borrow=True)
        self.hiddenLayer.b.set_value(params[1].get_value(), borrow=True)
        self.outputLayer.W.set_value(params[2].get_value(), borrow=True)
        self.outputLayer.b.set_value(params[3].get_value(), borrow=True)


def test_mlp(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000,
             dataset='mnist.pkl.gz', batch_size=20, n_hidden=500):
    """
    Demonstrate stochastic gradient descent optimization for a multilayer
    perceptron

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
    gradient

    :type L1_reg: float
    :param L1_reg: L1-norm's weight when added to the cost (see
    regularization)

    :type L2_reg: float
    :param L2_reg: L2-norm's weight when added to the cost (see
    regularization)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: the path of the MNIST dataset file from
                 http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz


   """
    datasets = load_data_npz(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    sentence_x, sentence_y = datasets[3]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.dvector('y')  # the labels are presented as 1D vector
    previous_samples = T.matrix()

    rng = numpy.random.RandomState(1234)

    # construct the MLP class
    regression = MLP(rng=rng, input=x, n_in=train_set_x.get_value(borrow=True).shape[1],
                     n_hidden=n_hidden, n_out=1)

    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = regression.errors(y) \
         + L1_reg * regression.L1 \
         + L2_reg * regression.L2_sqr

    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    test_model = theano.function(inputs=[index],
            outputs=regression.errors(y),
            givens={
                x: test_set_x[index * batch_size:(index + 1) * batch_size],
                y: test_set_y[index * batch_size:(index + 1) * batch_size]})

    validate_model = theano.function(inputs=[index],
            outputs=regression.errors(y),
            givens={
                x: valid_set_x[index * batch_size:(index + 1) * batch_size],
                y: valid_set_y[index * batch_size:(index + 1) * batch_size]})

    # compiling a Theano function that reconstructs a sentence
    yrec_model = theano.function(inputs=[],
        outputs=regression.outputLayer.y_pred,
        givens={x: sentence_x})

    # compling a Theano function that generates the next sample
    ygen_model = theano.function(inputs=[previous_samples],
        outputs=regression.outputLayer.y_pred,
        givens={x: previous_samples})

    # compute the gradient of cost with respect to theta (sotred in params)
    # the resulting gradients will be stored in a list gparams
    gparams = []
    for param in regression.params:
        gparam = T.grad(cost, param)
        gparams.append(gparam)

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs
    updates = []
    # given two list the zip A = [a1, a2, a3, a4] and B = [b1, b2, b3, b4] of
    # same length, zip generates a list C of same size, where each element
    # is a pair formed from the two lists :
    #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
    for param, gparam in zip(regression.params, gparams):
        updates.append((param, param - learning_rate * gparam))

    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(inputs=[index], outputs=cost,
            updates=updates,
            givens={
                x: train_set_x[index * batch_size:(index + 1) * batch_size],
                y: train_set_y[index * batch_size:(index + 1) * batch_size]})

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'

    best_validation_loss = numpy.inf
    best_epoch = 0
    test_score = 0.
    start_time = time.clock()

    train_err = []
    valid_err = []

    epoch = 0
    done_looping = False

    while (epoch < n_epochs):
        epoch = epoch + 1

        # training set
        train_losses = [train_model(i) for i in xrange(n_train_batches)]
        this_train_loss = numpy.mean(train_losses)

        # validation set
        validation_losses = [validate_model(i) for i in xrange(n_valid_batches)]
        this_validation_loss = numpy.mean(validation_losses)

        # save both errors
        train_err.append(this_train_loss)
        valid_err.append(this_validation_loss)

        # print error
        print('epoch %i, train error %f, validation error %f' %
             (epoch, this_train_loss, this_validation_loss))

        # if we got the best validation score until now
        if this_validation_loss < best_validation_loss:
            best_validation_loss = this_validation_loss
            best_epoch = epoch

            # Saving the model set
            regression.save_model()

    # Load the best model
    regression.load_model()
    test_losses = [test_model(i) for i in xrange(n_test_batches)]
    test_score = numpy.mean(test_losses)
    print(('    test error of best model %f') %
          (test_score))

    # Reconstruct the sentence
    print '... ... Reconstructing'
    y_pred = yrec_model()
    # Save in wav format
    output = numpy.int16(y_pred*560)
    wv.write('predicted_data.wav', 16000, output)

    # Generate the sentence
    print '... ... Generating'
    y_gen = numpy.zeros(30000)
    presamples = sentence_x.get_value()[2500]
    for i in xrange(30000):
        # without gaussian noise
#        y_gen[i] = ygen_model(presamples.reshape((1, 240)))
        # with gaussian noise
        y_gen[i] = numpy.random.normal(ygen_model(presamples.reshape((1, 240))),
                                       numpy.sqrt(min(train_err)))
        presamples = numpy.roll(presamples, -1)
        presamples[-1] = y_gen[i]
    output = numpy.int16(y_gen*560)
    wv.write('generated_data.wav', 16000, output)
    pylab.figure()
    pylab.plot(numpy.arange(30000)+2500, y_gen)
    pylab.xlabel('Samples')
    pylab.ylabel('Amplitude')
    pylab.savefig('generated_data.png', format='png')

    end_time = time.clock()
    print(('Optimization complete. Best validation score of %f '
           'obtained at epoch %i, with test performance %f') %
          (best_validation_loss, best_epoch, test_score))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
    pylab.figure()
    pylab.plot(range(epoch), train_err)
    pylab.plot(range(epoch), valid_err)
    pylab.xlabel('epoch')
    pylab.ylabel('MSE')
    pylab.legend(['train', 'valid'])
    pylab.savefig('error.png', format='png')

if __name__ == '__main__':
    theano.config.exception_verbosity = 'high'
    test_mlp(batch_size=20, n_epochs=100, n_hidden=10, dataset='timit_train.npz')
