#---------------------------------------------#
#-------| Written By: Syed Zain Raza |-------#
#---------------------------------------------#


#---------------Instructions------------------#

# You will be writing a super class named WeakLearner
# and then will be implmenting its sub classes
# RandomWeakLearner and LinearWeakLearner. Remember
# all the overridded functions in Python are by default
# virtual functions and every child classes inherits all the
# properties and attributes of parent class.

# Your task is to  override the train and evaluate functions
# of superclass WeakLearner in each of its base classes. 
# For this purpose you might have to write the auxiliary functions as well.

#--------------------------------------------------#

# Now, go and look for the missing code sections and fill them.
#-------------------------------------------#


import numpy as np
import scipy.stats as stats

class WeakLearner: # A simple weaklearner you used in Decision Trees...
    """ A Super class to implement different forms of weak learners...


    """
    def __init__(self):
        """
        Input:
            

        """
        #print "   "        
        pass

    def train(self,feat, Y):
        '''
            Trains a weak learner from all numerical attribute for all possible split points for
            possible feature selection
            
            Input:
            ---------
            feat: a contiuous feature
            Y: labels
            
            Returns:
            ----------
            v: splitting threshold
            score: splitting score
            Xlidx: Index of examples belonging to left child node
            Xridx: Index of examples belonging to right child node
            
        '''
        nexamples,nfeatures=X.shape

        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        
        self.classes=np.unique(Y)
            
        
        #---------End of Your Code-------------------------#
        return score, Xlidx,Xridx
    def evaluate(self,X):
        """
        Evalute the trained weak learner  on the given example...
        """ 
        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        
        findBestRandomSplit(X,self.classes)    
        
        #---------End of Your Code-------------------------#
    def evaluate_numerical_attribute(self,feat, Y):
        '''
            Evaluates the numerical attribute for all possible split points for
            possible feature selection
            
            Input:
            ---------
            feat: a contiuous feature
            Y: labels
            
            Returns:
            ----------
            v: splitting threshold
            score: splitting score
            Xlidx: Index of examples belonging to left child node
            Xridx: Index of examples belonging to right child node
            
        '''
        
        classes=np.unique(Y)
        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        # Same code as you written in DT assignment...
        
        nclasses=len(classes)
        
        sidx = np.argsort(feat) # sorted index of features
        
        f = feat[sidx] # sorted features
        
        sY = Y[sidx] # sorted features class labels...

        middle_Points = np.unique(f)
        
        middle_Points = (middle_Points[:-1] + middle_Points[1:])/2.0
         

      
        
        Information_Gain_Minimum = 0.0 #Its actually maximum :p
        
        req_Split_point = -1
  
        dataset_Entropy = 0.0 
        
        
        
        for i in self.classes:
            req_Prob = (np.sum(sY==i)*1.0)/len(sY)
            dataset_Entropy+=(req_Prob*np.log2(np.max([np.min([req_Prob-np.spacing(1),1]), np.spacing(1)])))
        
        dataset_Entropy=-1*dataset_Entropy

        for middle_Point in middle_Points:
            Req_DY_Entropy = 0.0
            Req_DN_Entropy = 0.0
            DY = sY[f<=middle_Point]
            DN = sY[f>middle_Point]
            
                     
            for i in self.classes:
                req_Prob_CY=(np.sum(DY==i)*1.0)/len(DY)
                Req_DY_Entropy+=(req_Prob_CY*np.log2(np.max([np.min([req_Prob_CY-np.spacing(1),1]), np.spacing(1)])))
                req_Prob_CN=(np.sum(DN==i)*1.0)/len(DN)
                Req_DN_Entropy+=(req_Prob_CN*np.log2(np.max([np.min([req_Prob_CN-np.spacing(1),1]), np.spacing(1)])))
            
            
            
            Req_DY_Entropy = -1*Req_DY_Entropy
            Req_DN_Entropy = -1*Req_DN_Entropy
            Req_Entropy_Split = ((len(DY)*1.0)/len(sY)*Req_DY_Entropy) + ((len(DN)*1.0)/len(sY)*Req_DN_Entropy)
            IG = dataset_Entropy - Req_Entropy_Split
            

            if IG>Information_Gain_Minimum:
                Information_Gain_Minimum=IG
                req_Split_point=middle_Point
        
            
                 
            
            
        #return Information_Gain_Minimum,req_Split_point
        
        
        
        #---------End of Your Code-------------------------#
            
        return req_Split_point,Information_Gain_Minimum

class RandomWeakLearner(WeakLearner):  # Axis Aligned weak learner....
    """ An Inherited class to implement Axis-Aligned weak learner using 
        a random set of features from the given set of features...


    """
    def __init__(self, nsplits=+np.inf, nrandfeat=None):
        """
        Input:
            nsplits = How many nsplits to use for each random feature, (if +inf, check all possible splits)
            nrandfeat = number of random features to test for each node (if None, nrandfeat= sqrt(nfeatures) )
        """
        WeakLearner.__init__(self) # calling base class constructor...        
        self.nsplits=nsplits
        self.nrandfeat=nrandfeat
        pass

    def train(self,X, Y):
        '''
            Trains a weak learner from all numerical attribute for all possible split points for
            possible feature selection
            
            Input:
            ---------
            X: a [m x d]  features matrix
            Y: a [m x 1] labels matrix
            
            Returns:
            ----------
            v: splitting threshold
            score: splitting score
            Xlidx: Index of examples belonging to left child node
            Xridx: Index of examples belonging to right child node
            
        '''
        nexamples,nfeatures=X.shape

        
        if(not self.nrandfeat):
            self.nrandfeat=np.round(np.sqrt(nfeatures))

        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        
        score = 0
        
        req_Split_point = np.array([])
        
        self.fidx = np.random.random(0,nfeatures,self.nrandfeat)
        
        for fid in fidx:
            self.rf = np.range(X[:,fid])
            splitvalue,minscore,bXl,bXr = findBestRandomSplit(fid,Y)
            if(minscore>score):
                req_Split_point = splitvalue
                score = minscore
                
        
        #---------End of Your Code-------------------------#
        return score, bXl,bXr

    def findBestRandomSplit(self,feat,Y):
        """
            
            Find the best random split by randomly sampling "nsplits"
            splits from the feature range...

            Input:
            ----------
            feat: [n X 1] nexamples with a single feature
            Y: [n X 1] label vector...

        """
        frange=np.max(feat)-np.min(feat)
        
        msh[] = np.ones(1)*2
        

        #import pdb;         pdb.set_trace()
        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        
        for s in range(0,frange):
            splitvalue = np.random.rand(1)*self.rf + min(X[feat,:])
            
        XL = X[:,feat] > splitvalue
        XR = np.logical - not(XL)
        
        msh[0] = X[XL]
        msh[1] = X[XR]
        
        req_Ent = calculateEntropy(Y,msh)
        
        
        #---------End of Your Code-------------------------#
        return splitvalue, req_Ent, msh[0], msh[1]
    def calculateEntropy(self,Y, mship):
        """
            calculates the split entropy using Y and mship (logical array) telling which 
            child the examples are being split into...

            Input:
            ---------
                Y: a label array
                mship: (logical array) telling which child the examples are being split into, whether
                        each example is assigned to left split or the right one..
            Returns:
            ---------
                entropy: split entropy of the split
        """

        lexam=Y[mship]
        rexam=Y[np.logical_not(mship)]

        pleft= len(lexam) / float(len(Y))
        pright= 1-pleft

        pl= stats.itemfreq(lexam)[:,1] / float(len(lexam)) + np.spacing(1)
        pr= stats.itemfreq(rexam)[:,1] / float(len(rexam)) + np.spacing(1)

        hl= -np.sum(pl*np.log2(pl)) 
        hr= -np.sum(pr*np.log2(pr)) 

        sentropy = pleft * hl + pright * hr

        return sentropy



# build a classifier ax+by+c=0
class LinearWeakLearner(RandomWeakLearner):  # A 2-dimensional linear weak learner....
    """ An Inherited class to implement 2D line based weak learner using 
        a random set of features from the given set of features...


    """
    def __init__(self, nsplits=10):
        """
        Input:
            nsplits = How many splits to use for each choosen line set of parameters...
            
        """
        RandomWeakLearner.__init__(self,nsplits)
        
        pass

    def train(self,X, Y):
        '''
            Trains a weak learner from all numerical attribute for all possible 
            
            Input:
            ---------
            X: a [m x d] data matrix ...
            Y: labels
            
            Returns:
            ----------
            v: splitting threshold
            score: splitting score
            Xlidx: Index of examples belonging to left child node
            Xridx: Index of examples belonging to right child node
            
        '''
        nexamples,nfeatures=X.shape

        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        for i in range(0,sqrt(nfeatures)):
            self.f1 = np.random.rand(1)
            self.f2 = np.random.rand(1)
            
        
        #---------End of Your Code-------------------------#

        return minscore, bXl, bXr

    

    

    def evaluate(self,X):
        """
        Evalute the trained weak learner  on the given example...
        """ 
        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        
        findBestRandomSplit1(X,self.classes)    
            
        
        #---------End of Your Code-------------------------#
        

        
    def findBestRandomSplit1(self,feat,Y):
        """
            
            Find the best random split by randomly sampling "nsplits"
            splits from the feature range...

            Input:
            ----------
            feat: [n X 1] nexamples with a single feature
            Y: [n X 1] label vector...

        """
        frange=np.max(feat)-np.min(feat)
        
        msh[] = np.ones(1)*2
        

        #import pdb;         pdb.set_trace()
        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        
        for s in range(0,frange):
            splitvalue = np.random.rand(3)
            res = X[:,self.f1]*split[0] + X[:,self.f2]*split[1] + split[2]
            
         
            
        XL = res < 0
        XR = res > 0
        
        msh[0] = X[XL]
        msh[1] = X[XR]
        
        req_Ent = calculateEntropy(Y,msh)
        
        
        #---------End of Your Code-------------------------#
        return splitvalue, req_Ent, msh[0], msh[1]
