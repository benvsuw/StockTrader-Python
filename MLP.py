import numpy as np
#from CSVREAD import getData

#Inputs = getData()# first column is the target other 13 are the inputs
#t = Inputs[:,0]
#13


class MLP(object):
	def __init__(self,NumInputs,Layers,HiddenN):

		#Step1 - Initialize
		self.numInputs = NumInputs
		self.NumHiddenNeurons = HiddenN
		self.NumHiddenLayers = Layers
		self.outN = self.NumHiddenLayers*self.NumHiddenNeurons+self.numInputs
		self.weights = np.matrix(0*np.random.random(((self.outN+1), (self.outN+1))))
		#Input layer weights
		for i in range(self.numInputs,(self.NumHiddenNeurons+self.numInputs)):
			for j in range(0,self.numInputs):
				self.weights[i,j] = np.random.random(1)
		#Hidden Layers
		for k in range(1,self.NumHiddenLayers):
			for h in range((self.numInputs+self.NumHiddenNeurons*k),(self.numInputs+self.NumHiddenNeurons*(k+1))):#ex ---5,8
				for j in range((self.numInputs+self.NumHiddenNeurons*(k-1)),(self.numInputs+self.NumHiddenNeurons*k)):#2,5
					self.weights[h,j] = np.random.random(1)
					#print "Layer: " +str(k) + " h: " + str(h) + " j: " + str(j) 
		#Output Layer
		for w in range((self.numInputs+(self.NumHiddenLayers-1)*self.NumHiddenNeurons),self.outN):
			self.weights[self.outN,w] = np.random.random(1)
		#print self.weights
		#print str(np.shape(self.weights))

	def Sigmoid(self,x):
		return 1/(1+np.exp(-x))

	def SigmoidPrime(self,x):
		return (np.exp(x))/((np.exp(x)+1)**2)
	
	def Binary(self,num):
		if num>=0.5:
			return 1
		if num<0.5:
			return 0


	def Train(self,Inps,Outputs,epochs):
		n,Emax,E = 0.9,0.01,0
		# setting max error, learning rates and others
		Inputs = Inps
		t = Outputs
		#dim = str(np.shape(Inputs))
		#N = int(dim[dim.find("(")+1:dim.find("L")])
		#M = int(dim[dim.find(" "):-2])
		M = len(Inps[0])
		N = len(Inps)
		#Step1 - Initialize
		numInputs = M
		#Training Exp function network
		for epoch in range(1,epochs+1): #how many epochs
			#print "Epoch - " + str(epoch)
			for i in range(0,N):
				sum=[0]*(self.outN+1)
				#Step2 - Apply Input Pattern
				O = [0]*(self.outN+1)
				delta = [0]*(self.outN+1)
				#Assigning inputs to the outputs of the input neuron layer
				for input in range(0,self.numInputs):
					O[input] = Inputs[i][input]
			
			#	print "-----------------Forward Propagation for Point " + str(i) + "-------------------------"
				#First Hidden Layer Forward Prop
				for num in range(self.numInputs,(self.numInputs+self.NumHiddenNeurons)):
					for input in range(0,self.numInputs):
						sum[num] +=self.weights[num,input]*(Inputs[i][input])
					O[num] = self.Sigmoid(sum[num])
				#	print "Output "+str(num) +" = " + str(O[num])
					
				#All other HiddenLayers
				for k in range(1,self.NumHiddenLayers):
					for h in range((numInputs+self.NumHiddenNeurons*k),(numInputs+self.NumHiddenNeurons*(k+1))):#ex ---5,8
						for j in range((numInputs+self.NumHiddenNeurons*(k-1)),(numInputs+self.NumHiddenNeurons*k)):#2,5
							sum[h] += self.weights[h,j]*O[j] 
						#	print "Layer: " +str(k) + " h: " + str(h) + " j: " + str(j)
						O[h] = self.Sigmoid(sum[h])
						
				#Output Layer			
				for num in range(self.numInputs+self.NumHiddenNeurons*(self.NumHiddenLayers-1),self.outN):#5,6,7
					sum[self.outN] += self.weights[self.outN,num]*O[num] 
				
				O[self.outN] = self.Sigmoid(sum[-1])
			#	print "-----------------Error Calculations for Point " + str(i) + "-------------------------"
				E = E + 0.5*(t[i] - O[self.outN])**2 
				#E += O[-1]*(1-O[-1])*(t[i,0]-O[-1])
				#print "Error = " + str(E)
				delta[self.outN] = self.SigmoidPrime(sum[-1])*(t[i] - O[-1])
				#print "Error Signal Delta = " +str(delta[self.outN])
				#print "-----------------Step 5 Output Layer Back Prop-------------------------"
				for node in range(self.numInputs+self.NumHiddenNeurons*(self.NumHiddenLayers-1),self.outN):
					self.weights[self.outN,node] += n*delta[self.outN]*O[node]
					
				#print "-----------------Step 5 Last Hidden Layer Error Signal Back Prop-------------------------"
				for node in range(self.numInputs+self.NumHiddenNeurons*(self.NumHiddenLayers-1),self.outN):#5-8
						delta[node] = O[node]*(1-O[node])*self.weights[self.outN,node]*delta[self.outN] #Calculate deltas 5-7
						
				dSum=0	
				for k in range(self.NumHiddenLayers,1,-1):
					for j in range(M + self.NumHiddenNeurons*(k-2),self.NumHiddenNeurons*(k-1)+M):#2,5#2->5
						for node in range((self.NumHiddenNeurons*(k-1)+M),(M+self.NumHiddenNeurons*k)):#ex ---5,8
							self.weights[node,j] +=n*delta[node]*O[j]
							dSum +=delta[node]*self.weights[node,j]
							#print "Layer: " +str(k) + " j: " + str(j) + " node: " + str(node) + " dSum: " + str(dSum)
						delta[j] = O[j]*(1-O[j])*dSum #Calculate deltas 2->5
						dSum=0
						
					#for node in range((self.NumHiddenNeurons*(k-1)+M),(M+self.NumHiddenNeurons*k)):#ex ---5,8
						#for j in range(M + self.NumHiddenNeurons*(k-2),self.NumHiddenNeurons*(k-1)+M):#2,5#2->5
							#self.weights[node,j] += n*delta[node]*O[j]
							
				for node in range(numInputs,numInputs + self.NumHiddenNeurons):#2-5
					for input in range(0,numInputs):#0-2
						self.weights[node,input] += n*delta[node]*O[input]
						#print  " Node: " + str(node) + " Input: " + str(input)
			#print delta	
			#print self.weights
			if abs(E)<Emax:
			#	print "Broke Out, Error is Smaller than Emax"
				break
	
	def Run(self,Data):
		M = len(Data[0])
		N = len(Data)
		#dim = str(np.shape(Data))
		#N = int(dim[dim.find("(")+1:dim.find("L")])
		#M = int(dim[dim.find(" "):-2])#num of inputs
		# setting max error, learning rates and others
		numInputs = M

		#Retrieve
		for i in range(0,N):
			
			sum=[0]*(self.outN+1)
			#Step2 - Apply Input Pattern
			O = [0]*(self.outN+1)
			delta = [0]*(self.outN+1)
			Outs = []
			#Assigning inputs to the outputs of the input neuron layer
			for input in range(0,M):
				O[input] = Data[i][input]
		
			#print "-----------------Forward Propagation for Point " + str(i) + "-------------------------"
			#First Hidden Layer Forward Prop
			for num in range(M,(M+self.NumHiddenNeurons)):
				for input in range(0,M):
					sum[num] +=self.weights[num,input]*Data[i][input]
					#print "Layer: 1" +" num: " + str(num) + " input: " + str(input)
				O[num] = self.Sigmoid(sum[num])
				#print "Output "+str(num) +" = " + str(O[num])
					
			#All other HiddenLayers
			#All other HiddenLayers
			for k in range(1,self.NumHiddenLayers):
				for h in range((numInputs+self.NumHiddenNeurons*k),(numInputs+self.NumHiddenNeurons*(k+1))):#ex ---5,8
					for j in range((numInputs+self.NumHiddenNeurons*(k-1)),(numInputs+self.NumHiddenNeurons*k)):#2,5
						sum[h] += self.weights[h,j]*O[j] 
					#	print "Layer: " +str(k) + " h: " + str(h) + " j: " + str(j)
					O[h] = self.Sigmoid(sum[h])
						
			#Output Layer			
			for num in range(M+self.NumHiddenNeurons*(self.NumHiddenLayers-1),self.outN):
				sum[self.outN] += self.weights[self.outN,num]*O[num] 
			#print sum[-1]	
			O[self.outN] = self.Sigmoid(sum[-1])
			#print O
			#print "Output "+str(self.outN) +" = " + str(O[self.outN])
			Outs.append(O[-1])
		return Outs


if __name__ == '__main__':		
	NumInputs = 3
	Layers =1#1
	HiddenN = 10#11
	MLP_S = MLP(NumInputs,Layers,HiddenN)
	MLP_B = MLP(NumInputs,Layers,HiddenN)

	In_S = [[1,0.03,0],[0,-0.03,0],[1,0.1,0.12],[0,-0.1,-0.2],[0.5,0.01,0.2],[0.85,0.03,-0.05]]
	OutS = [1,0,1,0,0,1]
	In2_S = [[0.8,0.03,0],[0.3,-0.03,0],[0.9,0.1,0.12],[0.4,-0.1,-0.2]]
	
	In_B = [[0,-0.07,0.2],[0,-0.03,0.2],[1,0.1,0.12],[0,-0.1,0.1],[0.2,0.03,0.1],[1,0.03,-0.05],[0.4,-0.02,0.1]]
	OutB = [1,1,0,1,1,0,1]
	In2_B = [[0.8,0.03,0],[0,-0.03,0.1],[0.9,0.1,0.12],[0.4,-0.1,-0.2],[0.4,0.01,0.2]]
	
	
	epochs = 1000#1000
	#MLP_S.Train(In_S,Out_S,epochs)
	#MLP_S.Run(In2_S)
	MLP_B.Train(In_B,OutB,epochs)
	MLP_B.Run(In2_B)
	
	

