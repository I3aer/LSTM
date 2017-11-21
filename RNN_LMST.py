import numpy as np

import caffe

from caffe import draw

from caffe.proto import caffe_pb2

from google.protobuf import text_format

import matplotlib.pyplot as plt

import earlyStopping as es

import dataGen as dg
            
def drawNet(netPrototxt):
    
    # Create the protobuf mbessage 
    net_pb = caffe_pb2.NetParameter()
    
    # open pre-defined network structure 
    with open(netPrototxt) as f:
        fprototxt = f.read()
        
    # load the structure to the protobuf message 
    text_format.Merge(fprototxt, net_pb)
    
    draw.draw_net_to_file(net_pb, "RNN_LSTM.png", rankdir="LR")


# number of mean and scale samples 
Ns = 10
mu = [i*1e-1 for i in range(-Ns/2,Ns/2)]
sc = [i*1e-2 for i in range(1,Ns+1)]
   
# minibatch size
mbSize = 10  
# generate training and validation data
trainDG = dg.dataGen(mu,sc,True,True,mbSize)   
trainD,d,lenS,valD,dVal = trainDG.getData()

# number of mini batches
Nmb = len(trainD)/mbSize

# plot showing training data
fig1 = plt.figure(1)
ax = fig1.add_subplot(2,1,1)

for i in range(len(trainD)):
    ax.plot(np.arange(i,(i+1),1.0/lenS),d[i], color = 'r', linestyle = '-', linewidth =1, marker = '.', markersize = 5)
    
# subplot showing predictions for validation data
ax = fig1.add_subplot(2,1,2)
for i in range(1):
    ax.plot(np.arange(i,(i+1),1.0/lenS), dVal[i], color = 'b', linestyle = '-', marker = '.', markersize = 5)  

# set computing mode
caffe.set_mode_cpu()

# draw the network
drawNet("RNN_LSTM.prototxt")

# load the SGD solver to optimize the params of the net
solver = caffe.SGDSolver('solver.prototxt')

'''Training Part:'''
# early stopping object
earlyStop = es.earlyStopping();

# data clips start with 0 and continues with 1. At each 
# new clip hidden states and memory cell are set to zero.
solver.net.blobs['clip'].data[:,0] = 1
solver.net.blobs['clip'].data[::lenS,0] = 0

# the bias of the forget gate (ft) is often initialized to a large
# positive value. Thus, the lstm initially remember the cell value

for name,layer in solver.net.params.items():
    if (name[0:4] == "lstm"):
        # Nb is the number of biases in each gate. There are four
        # gates: input, forget, output, input-modification gates.
        Nb = len(layer[2].data)/4
        layer[2].data[Nb:2*Nb] = 0.5


# suffle indices of training data
randIdx = np.arange(len(trainD))

# save flag for model parameters
sFlag  = False

# number of epoches (iterations over whole data)
epoch = solver.param.max_iter

# prediction losses 
train_loss = np.zeros(epoch)
val_loss = np.zeros(epoch)

for ep in range(epoch):
    
    '''train the network'''
    np.random.shuffle(randIdx)
    
    # mini-batch index
    for j in range(Nmb):
    
        # form mini batches
        for i,mIdx in enumerate(randIdx[j*mbSize:(j+1)*mbSize]):
            
            # load the label blob with the ground truth
            solver.net.blobs['label'].data[i*lenS:(i+1)*lenS,0] = d[mIdx][:,np.newaxis]
    
            # load the previous prediction, which is shifted ground truth
            solver.net.blobs['p2red'].data[i*lenS,0] = np.random.randint(1)
            solver.net.blobs['p2red'].data[i*lenS+1:(i+1)*lenS,0] = d[mIdx][0:-1, np.newaxis]
            
            # load the data blob with current data
            solver.net.blobs['data'].data[i*lenS:(i+1)*lenS,0] = trainD[mIdx][:, np.newaxis]

        # one step of SGD: forward evaluation, backward propagation, and update.
        solver.step(1)

        # Euclidean loss = 1/(mbSize*lenS)*sum_{i=0}^{mbSize*lenS}( norm2(d_i - pred_i)^2 )
        train_loss[ep] += solver.net.blobs['loss'].data
    
    # normalize the loss for each epoches        
    train_loss[ep] /= Nmb
    
    '''evaluate the performance of the trained network on validation data'''
    # form mini batch for validation
    for k in range(mbSize):
        
        # load the label blob with the ground truth
        solver.net.blobs['label'].data[k*lenS:(k+1)*lenS,0] = dVal[k][:,np.newaxis]

        # load the previous prediction, which is shifted ground truth
        solver.net.blobs['p2red'].data[k*lenS,0] = np.random.randint(1)
        solver.net.blobs['p2red'].data[k*lenS+1:(k+1)*lenS,0] = dVal[k][0:-1, np.newaxis]
        
        # load the data blob with current data
        solver.net.blobs['data'].data[k*lenS:(k+1)*lenS,0] = valD[k][:, np.newaxis]

    # forward the validation data through layers
    solver.net.forward()

    val_loss[ep] = solver.net.blobs['loss'].data
    
    lines = ax.plot(np.arange(0,1,1.0/lenS),solver.net.blobs['ip1'].data[0:lenS], color = 'g',linestyle='--', marker = '*', markersize = 3)
    
    # update and wait the subplot
    #plt.pause(0.005)
    
    status = earlyStop.check(train_loss[ep],ep)
    
    if (status == es.ESCondition[0]):
        print("saved the parameters of the trained network")
        solver.net.save("LSTM2.caffemodel")
        sFlag = True
        
    elif (status == es.ESCondition[1]):
        print("Early Stopping at epoch of {0}".format(ep))
        break;
    
    elif (ep == epoch-1 and not(sFlag)): 
        solver.net.save("LSTM2.caffemodel")
        
    if (ep < epoch-1):
        # remove the last drawn line
        l = lines.pop(-1).remove()
        
    print("ep:{0}".format(ep))
        
# figure showing training and validation losses, and predictions on test data
fig2 = plt.figure(2)
ax = fig2.add_subplot(2,1,1)
ax.plot(np.arange(ep), train_loss[:ep],color = 'b', linestyle = '-', marker = '.', markersize = 5)
ax.plot(np.arange(ep), val_loss[:ep],color = 'r', linestyle = '-', marker = '*', markerfacecolor = 'none', markersize = 5)

'''Testing Part:'''

# generate the 3test data without mean normalization and scaling
testDGen = dg.dataGen([2.2e-1],[1e-1], False, False, mbSize)
testD,d,lenS,_ ,_ = testDGen.getData()

stadMu, stadStd = trainDG.getStdParams()
testD = testDGen.standardization(stadMu, stadStd)

# predictions
preds = np.zeros(lenS)

# load the best network parameters
test_net = caffe.Net("RNN_LSTM.prototxt", "LSTM2.caffemodel", caffe.TEST)

# reshape inputs
test_net.blobs['clip'].reshape(1,1,1)
test_net.blobs['p2red'].reshape(1,1,1)
test_net.blobs['data'].reshape(1,1,1)

# propagate the new input shape to higher layers. 
test_net.reshape()

for j in range(lenS):
    
    test_net.blobs['data'].data[0] = testD[0][j]
    
    if (j == 0):
        # starts a new sequence
        test_net.blobs['clip'].data[0] = 0
        
        # load the initial random estimate 
        test_net.blobs['p2red'].data[0] = np.random.randint(low=1)
    else:
        # continue the existing sequence
        test_net.blobs['clip'].data[0] = 1
        
        # load the previous estimate 
        test_net.blobs['p2red'].data[0] = preds[j-1];
 
    test_net.forward()

    preds[j] = test_net.blobs['ip1'].data[0]
    
# subplot showing predictions and ground truth (i.e.,test signal)
ax = fig2.add_subplot(2,1,2)
ax.plot(np.arange(0,1,1.0/lenS),d[0], color = 'b',linestyle = '--', marker = '*',markersize = 3)
ax.plot(np.arange(0,1,1.0/lenS), preds, color = 'r',linestyle='--', marker = 'o', markerfacecolor = 'none', markersize = 3)
plt.show()