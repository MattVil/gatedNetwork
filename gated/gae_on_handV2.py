import pylab
import numpy
import cv2
from numpy import *
import numpy.random
import gatedAutoencoder
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams


def dispims(M, height, width, border=0, bordercolor=0.0, layout=None, **kwargs):
    """ Display a whole stack (colunmwise) of vectorized matrices. Useful
        eg. to display the weights of a neural network layer.
    """
    from pylab import cm, ceil
    numimages = M.shape[1]
    if layout is None:
        n0 = int(numpy.ceil(numpy.sqrt(numimages)))
        n1 = int(numpy.ceil(numpy.sqrt(numimages)))
    else:
        n0, n1 = layout
    im = bordercolor * numpy.ones(((height+border)*n0+border,(width+border)*n1+border),dtype='<f8')
    for i in range(n0):
        for j in range(n1):
            if i*n1+j < M.shape[1]:
                im[i*(height+border)+border:(i+1)*(height+border)+border,
                   j*(width+border)+border :(j+1)*(width+border)+border] = numpy.vstack((
                            numpy.hstack((numpy.reshape(M[:,i*n1+j],(height, width)),
                                   bordercolor*numpy.ones((height,border),dtype=float))),
                            bordercolor*numpy.ones((border,width+border),dtype=float)
                            ))
    pylab.imshow(im, cmap=cm.gray, interpolation='nearest', **kwargs)
    pylab.show()


class GraddescentMinibatch(object):
    """ Gradient descent trainer class.

    """
    def __init__(self, model, data, batchsize, learningrate, momentum=0.9, rng=None, verbose=True):
        self.model         = model
        self.data          = data
        self.learningrate  = learningrate
        self.verbose       = verbose
        self.batchsize     = batchsize
        self.numbatches    = self.data.get_value().shape[0] / batchsize
        self.momentum      = momentum
        if rng is None:
            self.rng = numpy.random.RandomState(1)
        else:
            self.rng = rng
        self.epochcount = 0
        self.index = T.lscalar()
        self.incs = dict([(p, theano.shared(value=numpy.zeros(p.get_value().shape,
                            dtype=theano.config.floatX), name='inc_'+p.name)) for p in self.model.params])
        self.inc_updates = {}
        self.updates = {}
        self.n = T.scalar('n')
        self.noop = 0.0 * self.n
        self.set_learningrate(self.learningrate)
    def set_learningrate(self, learningrate):
        self.learningrate = learningrate
        for _param, _grad in zip(self.model.params, self.model._grads):
            self.inc_updates[self.incs[_param]] = self.momentum * self.incs[_param] - self.learningrate * self.model.layer.learningrate_modifiers[_param.name] * _grad
            self.updates[_param] = _param + self.incs[_param]
        self._updateincs = theano.function([self.index], self.model._cost,
                                     updates = self.inc_updates,
                givens = {self.model.inputs:self.data[self.index*self.batchsize:(self.index+1)*self.batchsize]})
        self._trainmodel = theano.function([self.n], self.noop, updates = self.updates)
    def step(self):
        cost = 0.0
        stepcount = 0.0
        for batch_index in self.rng.permutation(self.numbatches-1):
            stepcount += 1.0
            cost = (1.0-1.0/stepcount)*cost + (1.0/stepcount)*self._updateincs(batch_index)
            self._trainmodel(0)
            self.model.layer.normalizefilters()
        self.epochcount += 1
        if self.verbose:
            print 'epoch: %d, cost: %f' % (self.epochcount, cost)

###
import pickle

print '... loading data'
#train_features_x = numpy.load('./shiftsuniform_x.npy').astype(theano.config.floatX) #OLD
#train_features_y = numpy.load('./shiftsuniform_y.npy').astype(theano.config.floatX) #OLD

pkl_file = open('../dataset/frame_only.txt', 'rb') #NEW

#dataSetImageMain = pickle.load(pkl_file) #(2000, 160, 160, 3)
train_features = pickle.load(pkl_file) #(2000, 160, 160, 3)

pkl_file.close()

def rgb2gray(rgb):
    return numpy.dot(rgb[...,:3], [0.299, 0.587, 0.114])

train_features   = numpy.array(train_features)
train_features   = rgb2gray(train_features)

# LIMIT THE RESOLUTION
spatial_res=1
train_features_x = train_features[::2,::spatial_res,::spatial_res]
train_features_y = train_features[1::2,::spatial_res,::spatial_res]
print("train feature X : {}".format(train_features_x.shape))
print("train feature Y : {}".format(train_features_y.shape))


# CUT THE IMAGE
# train_features_x = train_features[::2,60:,:160]
# train_features_y = train_features[1::2,60:,:160]

#NORMALIZE DATA:
train_features_x -= train_features_x.mean(0)[None, :]
train_features_y -= train_features_y.mean(0)[None, :]
train_features_x /= train_features_x.std(0)[None, :] + train_features_x.std() * 0.1
train_features_y /= train_features_y.std(0)[None, :] + train_features_y.std() * 0.1

#NORMALIZE DATA 2: NEW
train_features_x/=amax(train_features_x)
train_features_y/=amax(train_features_y)

#SHUFFLE TRAINING DATA TO MAKE SURE ITS NOT SORTED:
R = numpy.random.permutation(train_features_y.shape[0])
train_features_x = train_features_x[R, :]
train_features_y = train_features_y[R, :]

####
# FIGURE 1 IMAGE IN AND OUT
from matplotlib.pylab import *

matplotlib.rcParams.update({'font.size': 16})
ion()

R= numpy.random.randint(train_features_x.shape[0])

delay=0
img_1=train_features_x[R,:,:]
img_2=train_features_y[R+delay,:,:]

figure(1)
clf()
subplot(131)
imshow(img_1,interpolation='none',cmap='Greys')
#imshow(reshape(img_1,(28,20)),interpolation='none',cmap='Greys')
xlabel('X')
ylabel('Y')
title('Frame #%d' %(R))
subplot(132)
imshow(img_2,interpolation='none',cmap='Greys')
xlabel('X')
yticks([])
title('Frame #%d' %(R+delay))
subplot(133)
imshow(img_1*img_2,interpolation='none',cmap='Greys')
#imshow(reshape(matmult,(28,20)),interpolation='none',cmap='Greys')
xlabel('X')
yticks([])
title('#%dx%d' %(R,R+delay))
suptitle('MatMult Frame #%dx%d' %(R,R+delay))
###


train_features_x = reshape(train_features_x, (shape(train_features_x)[0], shape(train_features_x)[1]*shape(train_features_x)[2])) #NEW
train_features_y = reshape(train_features_y, (shape(train_features_y)[0], shape(train_features_y)[1]*shape(train_features_y)[2])) #NEW
train_features_numpy = numpy.concatenate((train_features_x, train_features_y), 1) # OLD

train_features = theano.shared(train_features_numpy)
#train_features_x = theano.shared(train_features_x)
#train_features_y = theano.shared(train_features_y)
print '... done'

numfac = 100
nummap = 25
numhid = 0
weight_decay_vis = 0.0
weight_decay_map = 0.0
corruption_type = 'none' # NEW
corruption_type = 'zeromask'
corruption_level = 0.5 #NEW
init_topology = None
batchsize = 100
numvisX = train_features_x.shape[1] #OLD
numvisY = train_features_y.shape[1] #OLD
numbatches = train_features.get_value().shape[0] / batchsize


# INSTANTIATE MODEL
print '... instantiating model'
numpy_rng  = numpy.random.RandomState(1)
theano_rng = RandomStreams(1)
model = gatedAutoencoder.FactoredGatedAutoencoder(numvisX=numvisX,
                                                  numvisY=numvisY,
                                                  numfac=numfac, nummap=nummap, numhid=numhid,
                                                  output_type='real',
                                                  weight_decay_vis=weight_decay_vis,
                                                  weight_decay_map=weight_decay_map,
                                                  corruption_type=corruption_type,
                                                  corruption_level=corruption_level,
                                                  init_topology = init_topology,
                                                  numpy_rng=numpy_rng,
                                                  theano_rng=theano_rng)

print '... done'



# TRAIN MODEL
numepochs = 10 # TODO: 100
learningrate = 0.01
#trainer = gatedAutoencoder.GraddescentMinibatch(model, train_features, batchsize, learningrate)
trainer = GraddescentMinibatch(model, train_features, batchsize, learningrate)

#learningrate = 0.1
trainer.set_learningrate(learningrate)

#trainer.step()

for epoch in xrange(numepochs):
    trainer.step()


# TRAIN MODEL
figure(2)
clf()
try:
    pylab.subplot(1, 2, 1)
    #dispims(model.layer.wxf.get_value(), sqrt(numvisX), sqrt(numvisX))
    #dispims(model.layer.wxf.get_value(), 160, 160)
    #dispims(model.layer.wxf.get_value(), 160, 160)
    dispims(model.layer.wxf.get_value(), 100, 160)
    pylab.subplot(1, 2, 2)
    #dispims(model.layer.wyf.get_value(), sqrt(numvisY), sqrt(numvisY), 2)
    dispims(model.layer.wyf.get_value(), 100, 160)
except Exception:
    pass

figure(3)
clf()
try:
    pylab.subplot(1, 2, 1)
    dispims(model.layer.whf_in.get_value().T, 10, 10, 2) #25,100
    #dispims(model.layer.wxf.get_value(), sqrt(numvisX), sqrt(numvisX))
    #dispims(model.layer.wxf.get_value(), 160, 160)
    #dispims(model.layer.wxf.get_value(), 160, 160)
    pylab.subplot(1, 2, 2)
    dispims(model.layer.whf.get_value(), 5, 5,2) #25,100
    #dispims(model.layer.wyf.get_value(), sqrt(numvisY), sqrt(numvisY), 2)
    #dispims(model.layer.wyf.get_value(), 100, 160)
except Exception:
    pass

cv2.waitKey(0)

#f = file('models_learned.save', 'wb')
#cPickle.dump(ca, f, protocol=cPickle.HIGHEST_PROTOCOL)
#f.close()

#model.save('model_learned.save')
#model.load('model_learned.save.npy')

# numpy.save('dataSetImageMain',dataSetImageMain)# (2000, 160, 160, 3)
numpy.save('train_features_numpy', train_features_numpy) #1000,32000
numpy.save('train_features', train_features) #1000,32000

idx=randint(100)
imshow(reshape(model.layer.wxf.get_value()[:,idx], (160, 160)),interpolation='none')
cv2.waitKey(0)
colorbar()
cv2.waitKey(0)
imshow(reshape(model.layer.wxf.get_value()[:,idx], (160, 160)),interpolation='none')
cv2.waitKey(0)
