#---------------------------------------------#
#-------| Written By: Syed Zain Raza |-------#
#---------------------------------------------#

# A good heuristic is to choose sqrt(nfeatures) to consider for each node...
import weakLearner as wl
import numpy as np
import scipy.stats as stats


#---------------Instructions------------------#

# Here you will have to reproduce the code you have already written in
# your previous assignment.

# However one major difference is that now each node non-terminal node of the
# tree  object will have  an instance of weaklearner...

# Look for the missing code sections and fill them.
#-------------------------------------------#

class Node:
    def __init__(self,klasslabel='',pdistribution=[],score=0,wlearner=None):
        """
               Input:
               --------------------------
               klasslabel: to use for leaf node
               pdistribution: posteriorprob class probability at the node
               score: split score 
               weaklearner: which weaklearner to use this node, an object of WeakLearner class or its childs...

        """

        self.lchild=None       
        self.rchild=None
        self.klasslabel=klasslabel
        self.pdistribution=pdistribution
        self.score=score
        self.wlearner=wlearner
        
    def set_childs(self,lchild,rchild):
        """
        function used to set the childs of the node
        input:
            lchild: assign it to node left child
            rchild: assign it to node right child
        """
        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        self.rchild = rchild
        
        self.lchild = lchild
            
        
        #---------End of Your Code-------------------------#

    def isleaf(self):
        """
            return true, if current node is leaf node
        """
        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        
        if self.lchild == None and self.rchild == None:
            return True
        else:
            return False
        
            
        
        #---------End of Your Code-------------------------#
    def isless_than_eq(self, X):
        """
            This function is used to decide which child node current example 
            should be directed to. i.e. returns true, if the current example should be
            sent to left child otherwise returns false.
        """

        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        
        # Here you will call the evaluate funciton of weaklearn on
        # the current example and return true or false...
        
        wl.evaluate(X)
        
        
        
        #---------End of Your Code-------------------------#

    def get_str(self):
        """
            returns a string representing the node information...
        """
        if self.isleaf():
            return 'C(posterior={},class={},Purity={})'.format(self.pdistribution, self.klasslabel,self.purity)
        else:
            return 'I(Fidx={},Score={},Split={})'.format(self.fidx,self.score,self.split)
    

class DecisionTree:
    ''' Implements the Decision Tree For Classification With Information Gain 
        as Splitting Criterion....
    '''
    def __init__(self, exthreshold=5, maxdepth=10,
     weaklearner="Conic", pdist=False, nsplits=10, nfeattest=None):        
        ''' 
        Input:
        -----------------
            exthreshold: Number of examples to stop splitting, i.e. stop if number examples at a given node are less than exthreshold
            maxdepth: maximum depth of tree upto which we should grow the tree. Remember a tree with depth=10 
            has 2^10=1K child nodes.
            weaklearner: weaklearner to use at each internal node.
            pdist: return posterior class distribution or not...
            nsplits: number of splits to use for weaklearner
        ''' 
        self.maxdepth=maxdepth
        self.exthreshold=exthreshold
        self.weaklearner=weaklearner
        self.nsplits=nsplits
        self.pdist=pdist
        self.nfeattest=nfeattest
        assert (weaklearner in ["Conic", "Linear","Axis-Aligned","Axis-Aligned-Random"])
        pass
    def getWeakLearner(self):
        if self.weaklearner == "Conic":
            return wl.ConicWeakLearner(self.nsplits)            
        elif self.weaklearner== "Linear":
            return wl.LinearWeakLearner(self.nsplits)
        elif self.weaklearner == "Axis-Aligned":
            return wl.WeakLearner()    
        else:
            return wl.RandomWeakLearner(self.nsplits,self.nfeattest)

        pass
    def train(self, X, Y):
        ''' Train Decision Tree using the given 
            X [m x d] data matrix and Y labels matrix
            
            Input:
            ------
            X: [m x d] a data matrix of m d-dimensional examples.
            Y: [m x 1] a label vector.
            
            Returns:
            -----------
            Nothing
            '''
        nexamples,nfeatures=X.shape
        ## now go and train a model for each class...
        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        
        self.classes=np.unique(Y)
        
        self.tree = self.build_tree(X,Y)
            
        #return self.tree
        
        #---------End of Your Code-------------------------#
    
    def build_tree(self, X, Y, depth = 0):
        """ 

            Function is used to recursively build the decision Tree 
          
            Input
            -----
            X: [m x d] a data matrix of m d-dimensional examples.
            Y: [m x 1] a label vector.
            
            Returns
            -------
            root node of the built tree...


        """
        nexamples, nfeatures=X.shape
      
        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        
        klasses = np.unique(Y);
        
        Purity_Maximum = 0
        
        Class_index_maximum = 0
        
        for i,j in enumerate(klasses):
            purity = np.sum(Y == j) / (len(Y)*1.0)
            if purity > Purity_Maximum:
                Purity_Maximum = purity
                Class_index_maximum = i
                

        if (nexamples < self.exthreshold) or (Purity_Maximum > self.purity) or (depth > self.maxdepth):
            req_node = Node(purity = Purity_Maximum,klasslabel = klasses[Class_index_maximum])
            return req_node
        
        req_Split_point = np.array([])
        
        score = 0
        
        #fidx = -1
        
        
        
        #for examples in range(nfeatures):
        #    Req_new_Score,newreq_Split_point = self.evaluate_numerical_attribute(X[:,examples],Y)
        #    if(Req_new_Score>score):
        #        req_Split_point = newreq_Split_point
        #        score = Req_new_Score
        #        fidx = examples
        z1 = X[Req_new_Crd]
        
        z2 = Y[Req_new_Crd]
        
        z3 = X[Rev_newCrd]
        
        z4 = Y[Rev_newCrd]
        
        for examples in range(nfeatures):
            a = isless_than_eq(examples)
        
        #Req_new_Crd = X[:,fidx]<=req_Split_point
        
        #Rev_newCrd = np.logical_not(Req_new_Crd)
        if(a==true):
            node = Node(purity = Purity_Maximum, score = score ,split = req_Split_point, fidx = fidx)
            node.set_childs(self.build_tree(z1,z2,depth+1),0)
        else:
            node = Node(purity = Purity_Maximum, score = score ,split = req_Split_point, fidx = fidx)
            node.set_childs(0,  self.build_tree(z3,z4,depth+1))
        
        
        #node = Node(purity = Purity_Maximum, score = score ,split = req_Split_point, fidx = fidx)
        
        #node.set_childs(self.build_tree(z1,z2,depth+1),  self.build_tree(z3,z4,depth+1))
        
       
        #---------End of Your Code-------------------------#
        
        return node
        
        
    def test(self, X):
        
        ''' Test the trained classifiers on the given set of examples 
        
                   
            Input:
            ------
            X: [m x d] a data matrix of m d-dimensional test examples.
           
            Returns:
            -----------
                pclass: the predicted class for each example, i.e. to which it belongs
        '''
        
        nexamples, nfeatures=X.shape
        pclasses=self.predict(X)
        
        # your code go here...
        
        return np.array(pclasses)
    
    def predict(self, X):
        
        """
        Test the trained classifiers on the given example X
        
                   
            Input:
            ------
            X: [1 x d] a d-dimensional test example.
           
            Returns:
            -----------
                pclass: the predicted class for the given example, i.e. to which it belongs
        """
        z=[]
        
        ans=[]
        
        for idx in range(X.shape[0]):
            self._predict(self.tree,X[idx,:],ans)
            z.append(ans[0])
        
        return z 
    
    def _predict(self,node, X, ans):
        """
            recursively traverse the tree from root to child and return the child node label
            for the given example X
        """

        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        
        if node.isleaf()==True:
            lbl= node.klasslabel
            ans.append(lbl)
            return ans
       
        
        if X[node.fidx]<=node.split:
            self._predict(node.lchild,X,ans)
        else:
            self._predict(node.rchild,X,ans)
            
        
        #---------End of Your Code-------------------------#
    

    def __str__(self):
        """
            overloaded function used by print function for printing the current tree in a
            string format
        """
        str = '---------------------------------------------------'
        str += '\n A Decision Tree With Depth={}'.format(self.find_depth())
        str += self.__print(self.tree)
        str += '\n---------------------------------------------------'
        return str  # self.__print(self.tree)        
        
     
    def _print(self, node):
        """
                Recursive function traverse each node and extract each node information
                in a string and finally returns a single string for complete tree for printing purposes
        """
        if not node:
            return
        if node.isleaf():
            return node.get_str()
        
        string = node.get_str() + self._print(node.lchild)
        return string + node.get_str() + self._print(node.rchild)
    
    def find_depth(self):
        """
            returns the depth of the tree...
        """
        return self._find_depth(self.tree)
    def _find_depth(self, node):
        """
            recursively traverse the tree to the depth of the tree and return the depth...
        """
        if not node:
            return
        if node.isleaf():
            return 1
        else:
            return max(self._find_depth(node.lchild), self._find_depth(node.rchild)) + 1
    def __print(self, node, depth=0):
        """
        
        """
        ret = ""

        # Print right branch
        if node.rchild:
            ret += self.__print(node.rchild, depth + 1)

        # Print own value
        
        ret += "\n" + ("    "*depth) + node.get_str()

        # Print left branch
        if node.lchild:
            ret += self.__print(node.lchild, depth + 1)
        
        return ret         
