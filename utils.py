
import cPickle
import gzip

import numpy as np
import theano
import theano.tensor as T

from numpy.lib.stride_tricks import as_strided


def shared_dataset(data_xy, borrow=True):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(np.asarray(data_x,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    shared_y = theano.shared(np.asarray(data_y,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to float. This little hack
    # lets ous get around this issue
    return shared_x, T.cast(shared_y, theano.config.floatX)


def load_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    '''
    print '... loading data'

    # Load the dataset
#    f = gzip.open(dataset, 'rb')
    f = file(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    #train_set, valid_set, test_set format: tuple(input, target)
    #input is an numpy.ndarray of 2 dimensions (a matrix)
    #witch row's correspond to an example. target is a
    #numpy.ndarray of 1 dimensions (vector)) that have the same length as
    #the number of rows in the input. It should give the target
    #target to the example with the same index in the input.

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval


def load_data2(dataset):
    ''' Loads the dataset
    does the same thing as load_data() but with an additional dataset

    :type dataset: string
    '''
    print '... loading data'

    f = file(dataset, 'rb')
    train_set, valid_set, test_set, sentence_set = cPickle.load(f)
    f.close()

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)
    sentence_x, sentence_y = shared_dataset(sentence_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y), (sentence_x, sentence_y)]
    return rval


def load_data_w_phn(data_wav, data_phn):
    ''' Loads the dataset
        does the same thing as load_data2() but for .npz file

        :type dataset: string
        '''

    print '... loading data'

    npzfile = np.load(data_wav, 'rb')
    train_wav = npzfile['train']
    valid_wav = npzfile['valid']
    test_wav = npzfile['test']
    npzfile.close()

    npzfile = np.load(data_phn, 'rb')
    train_phn = npzfile['train']
    valid_phn = npzfile['valid']
    test_phn = npzfile['test']
    npzfile.close()

    train_set_x, train_set_y = shared_dataset((np.concatenate((np.vstack(train_wav[0]), train_phn), axis=1), train_wav[1]))
    valid_set_x, valid_set_y = shared_dataset((np.concatenate((np.vstack(valid_wav[0]), valid_phn), axis=1), valid_wav[1]))
    test_set_x, test_set_y = shared_dataset((np.concatenate((np.vstack(test_wav[0]), test_phn), axis=1), test_wav[1]))

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval


def load_sentence(sentence):
    npzfile = np.load(sentence, 'rb')
    stnc = npzfile['sentence']
    npzfile.close()
    return stnc.astype(theano.config.floatX)


def load_data_npz(dataset):
    ''' Loads the dataset
    does the same thing as load_data2() but for .npz file

    :type dataset: string
    '''

    print '... loading data'

    npzfile = np.load(dataset, 'rb')
    train_set = npzfile['train']
    valid_set = npzfile['valid']
    test_set = npzfile['test']
    sentence_set = npzfile['sentence']
    npzfile.close()

    test_set_x, test_set_y = shared_dataset((np.vstack(test_set[0]), test_set[1]))
    valid_set_x, valid_set_y = shared_dataset((np.vstack(valid_set[0]), valid_set[1]))
    train_set_x, train_set_y = shared_dataset((np.vstack(train_set[0]), train_set[1]))
    sentence_x, sentence_y = shared_dataset((np.vstack(sentence_set[0]), sentence_set[1]))

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y), (sentence_x, sentence_y)]
    return rval


#segmentaxis code.
#This code has been implemented by Anne Archibald.
def segment_axis(a, length, overlap=0, axis=None, end='cut', endvalue=0):
    """Generate a new array that chops the given array along the given axis
    into overlapping frames.

    Parameters
    ----------
    a : array-like
        The array to segment
    length : int
        The length of each frame
    overlap : int, optional
        The number of array elements by which the frames should overlap
    axis : int, optional
        The axis to operate on; if None, act on the flattened array
    end : {'cut', 'wrap', 'end'}, optional
        What to do with the last frame, if the array is not evenly
        divisible into pieces.

            - 'cut'   Simply discard the extra values
            - 'wrap'  Copy values from the beginning of the array
            - 'pad'   Pad with a constant value

    endvalue : object
        The value to use for end='pad'


    Examples
    --------
    >>> segment_axis(arange(10), 4, 2)
    array([[0, 1, 2, 3],
           [2, 3, 4, 5],
           [4, 5, 6, 7],
           [6, 7, 8, 9]])

    Notes
    -----
    The array is not copied unless necessary (either because it is
    unevenly strided and being flattened or because end is set to
    'pad' or 'wrap').

    use as_strided

    """

    if axis is None:
        a = np.ravel(a) # may copy
        axis = 0

    l = a.shape[axis]

    if overlap>=length:
        raise ValueError, "frames cannot overlap by more than 100%"
    if overlap<0 or length<=0:
        raise ValueError, "overlap must be nonnegative and length must be "\
                          "positive"

    if l<length or (l-length)%(length-overlap):
        if l>length:
            roundup = length + \
                      (1+(l-length)//(length-overlap))*(length-overlap)
            rounddown = length + \
                        ((l-length)//(length-overlap))*(length-overlap)
        else:
            roundup = length
            rounddown = 0
        assert rounddown<l<roundup
        assert roundup==rounddown+(length-overlap) or \
               (roundup==length and rounddown==0)
        a = a.swapaxes(-1,axis)

        if end=='cut':
            a = a[...,:rounddown]
        elif end in ['pad','wrap']: # copying will be necessary
            s = list(a.shape)
            s[-1]=roundup
            b = np.empty(s,dtype=a.dtype)
            b[...,:l] = a
            if end=='pad':
                b[...,l:] = endvalue
            elif end=='wrap':
                b[...,l:] = a[...,:roundup-l]
            a = b

        a = a.swapaxes(-1,axis)


    l = a.shape[axis]
    if l==0:
        raise ValueError, "Not enough data points to segment array in 'cut' "\
                          "mode; try 'pad' or 'wrap'"
    assert l>=length
    assert (l-length)%(length-overlap) == 0
    n = 1+(l-length)//(length-overlap)
    s = a.strides[axis]
    newshape = a.shape[:axis] + (n,length) + a.shape[axis+1:]
    newstrides = a.strides[:axis] + ((length-overlap)*s, s) + \
                 a.strides[axis+1:]

    try:
        return as_strided(a, strides=newstrides, shape=newshape)
    except TypeError:
        warnings.warn("Problem with ndarray creation forces copy.")
        a = a.copy()
        # Shape doesn't change but strides does
        newstrides = a.strides[:axis] + ((length-overlap)*s, s) + \
                     a.strides[axis+1:]
        return as_strided(a, strides=newstrides, shape=newshape)
