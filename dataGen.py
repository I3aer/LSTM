import numpy as np

class dataGen(object):
    '''Data generation and preprocessing'''
    
    def __init__(self, mu, sc, normF, valF, mbSize):
    
        # Input sampling interval 
        self.x = np.arange(0,1,0.01) 
        
        # test/train and desired (label) signal
        self.ts = []
        self.ds = []
        
        # sample the validation data and the desired validation data
        self.valD = []
        self.dVal = []
        
        # scale factor to introduce offset to the desired signal
        scale = 10
        
        for m in mu:
            for s in sc:
                self.ds.append(self.signal(scale*m))
                self.ts.append(self.ds[-1] + self.noise(m,s))   
        
        if(normF):
            self.standardization()
        
        if(valF):   
            self.valSample(mbSize)
        
    def getData(self):
            return self.ts, self.ds, len(self.x), self.valD, self.dVal
        
    def getStdParams(self):
        return self.stadMu, self.stadStd
        
    
    def valSample(self,mbSize):
        '''Randomly sample validation data and desired validation data'''
        for i in range(mbSize):
            idx = i*mbSize + np.random.randint(low=-i,high=mbSize-i)
            self.dVal.append(self.ds.pop(idx))
            self.valD.append(self.ts[idx])
            self.ts = np.delete(self.ts,idx,0)
        
    def standardization(self,mu=None,std=None):
        '''Make all data clip have zero mean and unit variance'''
        if (mu is None):
            self.stadMu = np.mean(self.ts)
        else:
            self.stadMu = mu
            
        if (std is None):
            self.stadStd = np.std(self.ts)
        else:
            self.stadStd = std
            
        self.ts = (self.ts - self.stadMu)/self.stadStd
        
        return self.ts

        
    def signal(self,offset):
        return 1.5*np.sin(2*np.pi*self.x) + offset
    
    def noise(self,m,s):
        return np.random.normal(loc=m, scale = s, size=len(self.x)) 