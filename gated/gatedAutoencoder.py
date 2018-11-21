import numpy
import pylab
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

SMALL = 0.000001


class FactoredGatedNetworkLayer(object):
    """ A factored gated neural network layer, that is, a network layer whose parameters 
        are modulated by inputs. 
    """

    def __init__(self, inputs, numvisX, numvisY, numfac, nummap, numhid, output_type, 
                 single_input=False, corruption_type='zeromask', corruption_level=0.0, 
                 weight_decay_vis=0.0, weight_decay_map=0.0, weight_decay_hid=0.0, 
                 init_topology=None, numpy_rng=None, theano_rng=None, params=None):
        self.numvisX = numvisX
        self.numvisY = numvisY
        self.numfac  = numfac
        self.nummap  = nummap
        self.numhid  = numhid
        self.init_topology = init_topology
        self.output_type  = output_type
        self.single_input = single_input
        self.use_mean_units = self.numhid > 0
        self.corruption_type = corruption_type
        self.corruption_level = corruption_level
        self.weight_decay_vis = weight_decay_vis
        self.weight_decay_map = weight_decay_map
        self.weight_decay_hid = weight_decay_hid
        self.learningrate_modifiers = {}
        self.learningrate_modifiers['wxf']     = 1.0
        self.learningrate_modifiers['wyf']     = 1.0
        self.learningrate_modifiers['wmx']     = 1.0
        self.learningrate_modifiers['wmy']     = 1.0
        self.learningrate_modifiers['whf_in']  = 0.1
        self.learningrate_modifiers['whf']     = 0.1
        self.learningrate_modifiers['bmap']    = 0.1
        self.learningrate_modifiers['bhidX']   = 0.1
        self.learningrate_modifiers['bhidY']   = 0.1
        self.learningrate_modifiers['bvisX']   = 0.1
        self.learningrate_modifiers['bvisY']   = 0.1

        self.meannxf = 0.01
        self.meannyf = 0.01
        self.meannhf = 0.01

        self.numxf = self.numvisX * self.numfac
        if not self.single_input:
            self.numyf = self.numvisY * self.numfac
        else:
            self.numyf = 0
        self.numhf = self.nummap * self.numfac

        if not numpy_rng:  
            numpy_rng = numpy.random.RandomState(1)

        if not theano_rng:  
            theano_rng = RandomStreams(1)

        # SET UP VARIABLES AND PARAMETERS 
        if params is None:
            wxf_init = numpy_rng.normal(size=(numvisX, numfac)).astype(theano.config.floatX)*0.01
            wyf_init = numpy_rng.normal(size=(numvisY, numfac)).astype(theano.config.floatX)*0.01

            if init_topology is None:
                self.whf_init = numpy.exp(numpy_rng.uniform(low=-3.0, high=-2.0, size=(nummap, numfac)).astype(theano.config.floatX))
            elif init_topology == '1d': 
                assert numpy.mod(numfac, nummap) == 0
                #first, we generate a simple stride-based, stretched-out 'eye'-matrix 
                #then, we convolve (in horizontal direction only) 
                stride = numfac / nummap
                self.whf_init = numpy.zeros((nummap, numfac)).astype(theano.config.floatX)
                convkernel = numpy.ones(stride*2)
                convkernel /= convkernel.max()
                for i in range(nummap):
                    self.whf_init[i, i*stride] = 1
                    self.whf_init[i, :] = numpy.convolve(self.whf_init[i,:], convkernel, mode='same')
            elif init_topology == '2d':
                import scipy.signal
                stride = numfac / nummap
                convkernel2d = numpy.ones((numpy.sqrt(stride)*2, numpy.sqrt(stride)*2))
                self.whf_init = numpy.zeros((numpy.sqrt(nummap), numpy.sqrt(nummap), numpy.sqrt(numfac), numpy.sqrt(numfac)))
                for i in range(numpy.int(numpy.sqrt(nummap))):
                    for j in range(numpy.int(numpy.sqrt(nummap))):
                        self.whf_init[i, j, i*numpy.int(numpy.sqrt(stride)), j*numpy.int(numpy.sqrt(stride))] = 1
                        self.whf_init[i, j, :, :] = scipy.signal.convolve2d(self.whf_init[i, j, :, :], convkernel2d, mode='same').astype('float32')
                        #self.whf_init[i, j, :, :] = scipy.signal.convolve2d(self.whf_init[i, j, :, :], convkernel2d, mode='same', old_behavior=False).astype('float32')
                self.whf_init = self.whf_init.reshape(numpy.sqrt(nummap), numpy.sqrt(nummap), -1)
                self.whf_init = self.whf_init.transpose(2,0,1).reshape(-1, numpy.sqrt(nummap)**2).transpose(1,0)
                self.whf_init = self.whf_init.astype(theano.config.floatX)

            self.whf_in_init = numpy_rng.uniform(low=-0.01, high=+0.01, size=(nummap, numfac)).astype(theano.config.floatX)
            self.whf = theano.shared(value = self.whf_init, name='whf')
            self.whf_in = theano.shared(value = self.whf_in_init, name='whf_in')
            self.wxf = theano.shared(value = wxf_init, name = 'wxf')
            self.bvisX = theano.shared(value = numpy.zeros(numvisX, dtype=theano.config.floatX), name='bvisX')
            if not self.single_input:
                self.wyf = theano.shared(value = wyf_init, name = 'wyf')
            else:
                self.wyf = self.wxf
                self.bvisY = self.bvisX
            self.bvisY = theano.shared(value = numpy.zeros(numvisY, dtype=theano.config.floatX), name='bvisY')
            self.bmap = theano.shared(value = 0.0*numpy.ones(nummap, dtype=theano.config.floatX), name='bmap')
            wmx_init = numpy_rng.uniform(low=-0.01, high=+0.01, size=(numvisX, numhid)).astype(theano.config.floatX)
            wmy_init = numpy_rng.uniform(low=-0.01, high=+0.01, size=(numvisY, numhid)).astype(theano.config.floatX)
            self.wmx = theano.shared(value = wmx_init, name = 'wmx')
            self.wmy = theano.shared(value = wmy_init, name = 'wmy')
            self.bhidX = theano.shared(value = numpy.zeros(numhid, dtype=theano.config.floatX), name='bhidX')
            self.bhidY = theano.shared(value = numpy.zeros(numhid, dtype=theano.config.floatX), name='bhidY')
            self.inputs = inputs
        else:
            self.wxf = params['wxf']
            self.wyf = params['wyf']
            self.whf_in = params['whf_in']
            self.whf = params['whf']
            self.bvisX = params['bvisX']
            self.bvisY = params['bvisY']
            self.wmx = theano.shared(value = wmx_init, name = 'wmx')
            self.wmy = theano.shared(value = wmy_init, name = 'wmy')

        self.params = [self.wxf, self.wyf, self.whf_in, self.whf, self.bmap, self.bvisX, self.bvisY]
        self.paramsWithWeightcost = [self.wxf, self.wyf, self.whf_in, self.whf]
        if self.use_mean_units:
            self.params.extend([self.wmx, self.wmy, self.bhidX, self.bhidY])
            self.paramsWithWeightcost.extend([self.wmx, self.wmy])

        # DEFINE THE FUNCTIONALITY OF THE LAYER 
        if not self.single_input:
            self.inputsX = self.inputs[:, :numvisX]
            self.inputsY = self.inputs[:, numvisX:]
        else:
            self.inputsX = self.inputs
            self.inputsY = self.inputs
        if self.corruption_level > 0.0: 
            if self.corruption_type=='zeromask':
                self._corruptedX = theano_rng.binomial(size=self.inputsX.shape, n=1, p=1.0-self.corruption_level, dtype=theano.config.floatX) * self.inputsX
                self._corruptedY = theano_rng.binomial(size=self.inputsY.shape, n=1, p=1.0-self.corruption_level, dtype=theano.config.floatX) * self.inputsY
            elif self.corruption_type=='gaussian':
                self._corruptedX = theano_rng.normal(size=self.inputsX.shape, avg=0.0, std=self.corruption_level, dtype=theano.config.floatX) + self.inputsX
                self._corruptedY = theano_rng.normal(size=self.inputsY.shape, avg=0.0, std=self.corruption_level, dtype=theano.config.floatX) + self.inputsY
            elif self.corruption_type=='none':
                self._corruptedX = self.inputsX
                self._corruptedY = self.inputsY
            else:
                assert False, "unknown corruption type"
        else:
            self._corruptedX = self.inputsX
            self._corruptedY = self.inputsY

        self._factorsX = T.dot(self._corruptedX, self.wxf)
        self._factorsY = T.dot(self._corruptedY, self.wyf)
        self._factorsXNonoise = T.dot(self.inputsX, self.wxf)
        self._factorsYNonoise = T.dot(self.inputsY, self.wyf)
        self._mappings = T.nnet.sigmoid(T.dot(self._factorsX*self._factorsY, self.whf_in.T)+self.bmap)
        self._mappingsNonoise = T.nnet.sigmoid(T.dot(self._factorsXNonoise*self._factorsYNonoise, self.whf_in.T)+self.bmap)
        self._factorsH = T.dot(self._mappings, self.whf)
        if self.use_mean_units:
            self._hiddensX = T.nnet.sigmoid(T.dot(self._corruptedX, self.wmx)+self.bhidX)
            self._hiddensY = T.nnet.sigmoid(T.dot(self._corruptedY, self.wmy)+self.bhidY)
            self._hiddensXNonoise = T.nnet.sigmoid(T.dot(self.inputsX, self.wmx)+self.bhidX)
            self._hiddensYNonoise = T.nnet.sigmoid(T.dot(self.inputsY, self.wmy)+self.bhidY)
            self._outputX_acts = T.dot(self._factorsY*self._factorsH, self.wxf.T)+self.bvisX+T.dot(self._hiddensX, self.wmx.T)
            self._outputY_acts = T.dot(self._factorsX*self._factorsH, self.wyf.T)+self.bvisY+T.dot(self._hiddensY, self.wmy.T)
            self._mappingsandhiddens = T.concatenate((self._mappings, self._hiddensX, self._hiddensY), 1)
            self._mappingsandhiddensNonoise = T.concatenate((self._mappingsNonoise, self._hiddensXNonoise, self._hiddensYNonoise), 1)
        else:
            self._outputX_acts = T.dot(self._factorsY*self._factorsH, self.wxf.T)+self.bvisX
            self._outputY_acts = T.dot(self._factorsX*self._factorsH, self.wyf.T)+self.bvisY
            self._mappingsandhiddens = self._mappings
            self._mappingsandhiddensNonoise = self._mappingsNonoise
        if self.output_type == 'binary':
            self._reconsX = T.nnet.sigmoid(self._outputX_acts)
            self._reconsY = T.nnet.sigmoid(self._outputY_acts)
        elif self.output_type == 'real':
            self._reconsX = self._outputX_acts
            self._reconsY = self._outputY_acts
        else:
            assert False, "unknown output type (has to be either 'binary' or 'real')"

        # DEFINE WEIGHTCOST, TO POTENTIALLY PASS ON TO THE MODEL
        self._weightcost = self.weight_decay_vis*( (self.wxf**2).sum() + (self.wyf**2).sum()) \
                         + self.weight_decay_map*(self.whf**2).sum()\
                         + self.weight_decay_map*(self.whf_in**2).sum()\
                         + self.weight_decay_hid*((self.wmx**2).sum() + (self.wmy**2).sum())
        self._weightcostgrads = T.grad(self._weightcost, self.paramsWithWeightcost)

        # COMPILE SOME FUNCTIONS THAT MAY BE USEFUL LATER 
        self.corruptedX = theano.function([self.inputs], self._corruptedX)
        self.corruptedY = theano.function([self.inputs], self._corruptedY)
        self.mappings = theano.function([self.inputs], self._mappingsNonoise)
        if self.use_mean_units:
            self.hiddensX = theano.function([self.inputs], self._hiddensX)
            self.hiddensY = theano.function([self.inputs], self._hiddensY)

        self.mappingsandhiddens = theano.function([self.inputs], self._mappingsandhiddensNonoise)
        self.reconsX = theano.function([self.inputs], self._reconsX)
        self.reconsY = theano.function([self.inputs], self._reconsY)

    def updateparams(self, newparams):
        def inplaceupdate(x, new):
            x[...] = new
            return x

        paramscounter = 0
        for p in self.params:
            pshape = p.get_value().shape
            pnum = numpy.prod(pshape)
            p.set_value(inplaceupdate(p.get_value(borrow=True), newparams[paramscounter:paramscounter+pnum].reshape(*pshape)), borrow=True)
            paramscounter += pnum 

    def get_params(self):
        return numpy.concatenate([p.get_value().flatten() for p in self.params])

    def normalizefilters(self, center=True, normalize_map_filters=False):
        def inplacemult(x, v):
            x[:, :] *= v
            return x
        def inplacesubtract(x, v):
            x[:, :] -= v
            return x
        nwxf = (self.wxf.get_value().std(0)+SMALL)[numpy.newaxis, :]
        nwyf = (self.wyf.get_value().std(0)+SMALL)[numpy.newaxis, :]
        nwhf = (numpy.sqrt((self.whf.get_value()**2).sum(0))+SMALL)[numpy.newaxis, :]
        meannxf = nwxf.mean()
        meannyf = nwyf.mean()
        meannhf = nwhf.mean()
        self.meannxf = 0.95 * self.meannxf + 0.05 * meannxf
        self.meannyf = 0.95 * self.meannyf + 0.05 * meannyf
        self.meannhf = 0.95 * self.meannhf + 0.05 * meannhf
        #print 'nxf: ', self.meannxf
        #print 'nyf: ', self.meannyf
        #print 'nhf: ', self.meannhf
        if self.meannxf > 1.5:
            self.meannxf = 1.5
        if self.meannyf > 1.5:
            self.meannyf = 1.5
        if self.meannhf > 1.5:
            self.meannhf = 1.5
        wxf = self.wxf.get_value(borrow=True)
        wyf = self.wyf.get_value(borrow=True)
        whf = self.whf.get_value(borrow=True)
        # CENTER FILTERS 
        if center:
            self.wxf.set_value(inplacesubtract(wxf, wxf.mean(0)[numpy.newaxis,:]), borrow=True)
            self.wyf.set_value(inplacesubtract(wyf, wyf.mean(0)[numpy.newaxis,:]), borrow=True)
            if normalize_map_filters:
                self.whf.set_value(inplacesubtract(whf, whf.mean(0)[numpy.newaxis,:]), borrow=True)
        # FIX STANDARD DEVIATION 
        self.wxf.set_value(inplacemult(wxf, self.meannxf/nwxf),borrow=True)
        self.wyf.set_value(inplacemult(wyf, self.meannyf/nwyf),borrow=True)
        if normalize_map_filters:
            self.whf.set_value(inplacemult(whf, self.meannhf/nwhf),borrow=True)


class FactoredGatedAutoencoder(object):
    """ A gated autoencoder network.

    """

    def __init__(self, numvisX, numvisY, numfac, nummap, numhid, output_type, 
                 single_input=False, corruption_type='zeromask', corruption_level=0.0, 
                 weight_decay_vis=0.0, weight_decay_map=0.0, weight_decay_hid=0.0, 
                 init_topology=None, inputs=None, numpy_rng=None, theano_rng=None):
        self.numvisX = numvisX
        self.numvisY = numvisY
        self.numfac  = numfac
        self.nummap  = nummap
        self.numhid  = numhid
        self.init_topology = init_topology
        self.output_type  = output_type
        self.use_mean_units = self.numhid > 0
        self.corruption_type = corruption_type
        self.corruption_level = corruption_level
        self.weight_decay_vis = weight_decay_vis
        self.weight_decay_map = weight_decay_map
        self.weight_decay_hid = weight_decay_hid
        self.single_input = single_input
        if inputs is None:
            self.inputs = T.matrix(name = 'inputs') 
        else:
            self.inputs = inputs

        self.numxf = self.numvisX * self.numfac
        if not self.single_input:
            self.numyf = self.numvisY * self.numfac
        else:
            self.numyf = 0
        self.numhf = self.nummap * self.numfac

        if not numpy_rng:  
            numpy_rng = numpy.random.RandomState(1)

        if not theano_rng:  
            theano_rng = RandomStreams(1)

        # SET UP MODEL USING LAYER
        self.layer = FactoredGatedNetworkLayer(self.inputs, numvisX, numvisY, numfac, nummap, numhid, output_type, 
                                               single_input=single_input, 
                                               corruption_type=self.corruption_type, corruption_level=corruption_level,
                                               weight_decay_vis=weight_decay_vis, weight_decay_map=weight_decay_map,
                                               weight_decay_hid=weight_decay_hid,
                                               init_topology=self.init_topology, 
                                               numpy_rng=numpy_rng, theano_rng=theano_rng)
        self.params = self.layer.params
        self._reconsX = self.layer._reconsX
        self._reconsY = self.layer._reconsY

        # ATTACH COST FUNCTIONS
        if self.output_type == 'binary':
            self._costpercase = - T.sum(
                  0.5* (self.layer.inputsY*T.log(self._reconsY) + (1.0-self.layer.inputsY)*T.log(1.0-self._reconsY))
                  +0.5* (self.layer.inputsX*T.log(self._reconsX) + (1.0-self.layer.inputsX)*T.log(1.0-self._reconsX)),
                                   axis=1)
        elif self.output_type == 'real':
            self._costpercase = T.sum(0.5*((self.layer.inputsX-self._reconsX)**2)
                                     +0.5*((self.layer.inputsY-self._reconsY)**2), axis=1)
        else:
            assert False, "unknown output type (has to be either 'binary' or 'real')"
        self._weightcost = self.layer._weightcost
        self._cost = T.mean(self._costpercase) + self._weightcost
        self._cost_pure = T.mean(self._costpercase) 
        self._grads = T.grad(self._cost, self.params)

        # COMPILE SOME FUNCTIONS THAT MAY BE USEFUL LATER 
        self.mappings = theano.function([self.inputs], self.layer._mappingsNonoise)
        if self.use_mean_units:
            self.hiddensX = theano.function([self.inputs], self.layer._hiddensX)
            self.hiddensY = theano.function([self.inputs], self.layer._hiddensY)

        self.mappingsandhiddens = theano.function([self.inputs], self.layer._mappingsandhiddensNonoise)
        self.reconsX = theano.function([self.inputs], self._reconsX)
        self.reconsY = theano.function([self.inputs], self._reconsY)
        self.cost = theano.function([self.inputs], self._cost)
        self.cost_pure = theano.function([self.inputs], self._cost_pure)
        self.grads = theano.function([self.inputs], self._grads)
        def get_cudandarray_value(x):
            if type(x)==theano.sandbox.cuda.CudaNdarray:
                return numpy.array(x.__array__()).flatten()
            else:
                return x.flatten()
        self.grad = lambda x: numpy.concatenate([get_cudandarray_value(g) for g in self.grads(x)])

    def updateparams(self, newparams):
        def inplaceupdate(x, new):
            x[...] = new
            return x

        paramscounter = 0
        for p in self.params:
            pshape = p.get_value().shape
            pnum = numpy.prod(pshape)
            p.set_value(inplaceupdate(p.get_value(borrow=True), newparams[paramscounter:paramscounter+pnum].reshape(*pshape)), borrow=True)
            paramscounter += pnum 

    def get_params(self):
        return numpy.concatenate([p.get_value().flatten() for p in self.params])

    def save(self, filename):
        numpy.save(filename, self.get_params())

    def load(self, filename):
        self.updateparams(numpy.load(filename))

    def set_corruption(self, corruption_type, corruption_level):
        self.corruption_type = corruption_type
        self.corruption_level = corruption_level
        self.layer.corruption_type = corruption_type
        self.layer.corruption_level = corruption_level


