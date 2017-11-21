# Early Stopping conditions
ESCondition = ("save", "stop", "neglect")

class earlyStopping(object):
    '''Early-stopping combats overfitting by monitoring the model's performance on a validation set. '''
    def __init__(self, pt = 5, th = 1e-2, mNItr = 100):
    
        # the number of epochs with no improvement after which training will be stopped.
        self.patience = pt
        
        self.cnt = 0
        
        # improvement threshold to increase the patience
        self.imp_th = th
        
        # improvement scale factor
        self.imp_sc = 5
        
        # maximum patience to continue training 
        self.max_pt = 20
        
        # best_losses
        self.valL = None
        
        # minimum number of iterations before early stopping
        self.min_NItr = mNItr
        
    def check(self,valL,ep):
        
        if (ep < self.min_NItr):    
            if (self.valL is None):
                self.valL = valL
            else:
                self.valL = min(self.valL,valL)
                
            return "neglect"
        
        # compute the rate of change in losses
        delta_valL = (self.valL - valL)/self.valL
        
        self.valL = min(self.valL,valL)
        
        if (delta_valL > 0):
            self.cnt = 0
    
            # improve patience if loss improvement is good enough
            if (delta_valL > self.imp_th):
                self.patience = min(self.imp_sc*self.patience, self.max_pt)
            
            return "save"
        
        else:
            
            self.cnt += 1
            
            # over-fitting test
            if (self.cnt > self.patience):
                return "stop"
            else:
                return "neglect"