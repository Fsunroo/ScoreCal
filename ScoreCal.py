import numpy as np 
#X--> input , Y-->Out put
X=np.array(([2,9],[1,5],[3,6]),dtype=float)
Y=np.array(([92],[86],[89]),dtype=float)
X=X/np.amax(X,axis=0)
Y=Y/100
Q=np.array(([4,8]),dtype=float)
Q=Q/np.amax(Q,axis=0)
class scorecal(object):
	"""docstring for scorecal"""
	def __init__(self):
		#declearing sizes
		self.inputSize=2
		self.outputSize=1
		self.hiddenSize=3
		#declearing random parameters
		self.w1=np.random.randn(self.inputSize, self.hiddenSize) # an 2*3 array   3*1 . 2*3 = 3*3
		self.w2=np.random.randn(self.hiddenSize, self.outputSize) # an 3*1 array    3*3 . 3.1 = 3.1
	def sigmoid(self,s):
		return 1/(1+np.exp(-s))
	def Forward(self,X):
		self.z=np.dot(X, self.w1)
		self.z2=self.sigmoid(self.z)
		self.z3=np.dot(self.z2, self.w2)
		o=self.sigmoid(self.z3)
		return o
	def outputa(self,a):
		return a*(1-a)
	def Backward(self,X,Y,o):
		"""ajustmant = eror*input*output*(1-output)"""
		self.eror_o=Y-o
		self.delta_o= self.eror_o*self.outputa(o)
		self.w2+=self.z2.T.dot(self.delta_o)

		self.eror_z2=self.delta_o.dot(self.w2.T)
		self.delta_z2=self.eror_z2*self.outputa(self.z2)
		self.w1+=X.T.dot(self.delta_z2)
	def train(self,X,Y):
		o=self.Forward(X)
		self.Backward(X,Y,o)
SC=scorecal()
for i in range(50000):
	SC.train(X,Y)
print X
print "result is \n" +str(SC.Forward(X))
print "result is \n" +str(SC.Forward(Q))
		