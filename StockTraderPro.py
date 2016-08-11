#Neural Network 1 - For Buying
	#Input Current Price normalized by the daily/yearly high
	#Output Buy or Hold
#Neural Network 2 - For Selling
	#If we have bought this stock we use this NN
	#Input Current Price normalized by the daily/yearly high
	#Threshold will be the difference between the input and bough price to be = 1% for example
	#Output Sell or Hold
	
#Can store previous 10 or so prices so that we can estimate rates of change
#Based on these rates of change we can 

import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from yahoo_finance import*
from pprint import pprint
from MLP import MLP
#from StockRunner import*

class StockData(object):
	def __init__(self,Stock):
		self.High = 0
		self.Low = 9999
		self.Commission = 19.98
		self.BuyThresh = 1
		self.SellThresh = 0.02
		self.normPrice = 1
		self.normPriceRange = 1
		self.count = 1.0
		self.spread = 0.02
		self.priceTotal = 0
		self.PriceList = [0]*50
		self.ROC = [0]*50
		self.ROCROC = [0]*50
		self.avg50D = 0
		
	def calc50DAvg(self):
		sum = 0
		count = 1
		for P in self.PriceList:
			if P ==0:
				break
			sum+=P
			self.avg50D = sum/count
	
		

class Bank(object):
	def __init__(self,balance):
		self.origBalance = balance
		self.cashBalance = balance
		self.stockBalance = 0
		self.quantity = 0 #how many shares owned
		self.price = 0 #price bought at
		self.numTrades=0
		
	
	def Buy(self,price,investment):
		quantity=self.CalcQuantity(price,investment)
		if self.cashBalance< price*quantity:
			print "Not Enough Cash"
			return -1
		self.cashBalance -=price*quantity
		self.stockBalance +=price*quantity
		self.quantity = quantity
		self.price = price
		return 1
		
	def Sell(self,price,quantity):
		if self.quantity<quantity:
			print "Trying to sell too many shares"
			return -1
		if self.quantity == quantity:
			self.price = 0
		self.quantity -= quantity
		self.stockBalance -= price*quantity
		self.cashBalance +=price*quantity
		return 1
	
	def UpdateStockBalance(self,price):
		self.stockBalance = self.quantity*price
		return
	
	def CalcProfit(self,price,quantity):
		return (price*quantity - self.price*quantity )
	
	def CalcQuantity(self,price,investment):
		return investment//price
		
	def NormProfit(self,price):
		return self.CalcProfit(price,self.quantity)/(self.price*self.quantity)

def sigmoid(Profit,Th):
	if Profit-Th <= 0:
		return -1 #Hold don't sell
	if Profit - Th >0:
		return 1

def GenPrices(range,stdev,mean):
	Prices = np.round(np.random.normal(mean,stdev,range),2)
	return Prices

def getHistory(Stock,start_date,end_date):
	history = Stock.get_historical(start_date, end_date) #format is 2014-04-25
	Dates = []
	Prices=[]
	PriceDate = {}
	count = len(history)-1
	index = 0
	#pprint(history)
	
	for h in history:
		h = str(history[count])
		h = h[h.find("Date")+8:]
		Dates.append(h[:h.find("'")])
		
		p = str(history[count])
		p = p[p.find("Adj_Close")+13:]
		Prices.append(p[:p.find("'")])
		
		PriceDate[Dates[index]] = Prices[index]
		
		count -=1
		index +=1
	
	#print sorted(PriceDate)
	#pprint(PriceDate)
	return PriceDate

def sortPrices(Prices):
	P = []
	for Date in sorted(Prices):
		P.append(float((Prices[Date])))
	return P

def ROCALL(Prices,startDate,endDate):
	Prices = sortPrices(Prices)
	return (Prices[-1] - Prices[0])/Prices[0]
	

def HNNB(ROC,ROC2,BT):
	States = [[1,1,1],[1,1,-1],[1,-1,1],[1,-1,-1],[-1,-1,-1],[-1,1,-1],[-1,1,1],[-1,-1,1]]

def HNNS(ROC,ROC2,ST):
	States = [[1,1,1],[1,1,-1],[1,-1,1],[1,-1,-1],[-1,-1,-1],[-1,1,-1],[-1,1,1],[-1,-1,1]]	

def Trade(Prices, Wallet,Stock,MLP_S,MLP_B):
	Data = StockData(Stock)
	
	for Date in sorted(Prices):
		SellConf = 0
		BuyConf = 0
		P = float((Prices[Date]))
		Data.PriceList.append(P)
		Data.PriceList.pop(0)
		Data.calc50DAvg()
		
		if Data.count > 3:
			Data.ROC.append((Data.PriceList[-1]-Data.PriceList[-2])/Data.PriceList[-2])#Rate of change as a percentage between -1,1
			Data.ROC.pop(0)
			Data.ROCROC.append((Data.ROC[-1]-Data.ROC[-2]))
			Data.ROCROC.pop(0)
		
		if Data.count%20 == 0:
			Data.Low = Data.High
		if P > Data.High:
			Data.High = P +0.000001 #to make sure we don't divide by 0 later on
			#print "P: " + str(P) + " ROC: " + str(Data.ROC[-1]) + " ROC2: " + str(Data.ROCROC[-1] ) + "Date: " + Date
			#Date = Date.replace('-','/')
			#plt.plot(dt.datetime.strptime(Date, '%Y/%m/%d').date(), P,marker='x', color='y')
		if P<Data.Low:
			Data.Low = P
			
			#Conditions to Sell
		if Wallet.quantity>0:
			Wallet.UpdateStockBalance(P)
			normPrice = (P-Data.Low)/(Data.High-Data.Low)
			SellConf=MLP_S.Run([[normPrice,Data.ROC[-1],Data.ROCROC[-1]]])
			if (Wallet.NormProfit(P)>(Data.SellThresh+Data.spread+(Data.Commission/Wallet.stockBalance)) and SellConf[0]>0.95):
				print "Sell Confidence = " + str(SellConf)
				Wallet.Sell(P,Wallet.quantity)
				Date = Date.replace('-','/')
				plt.plot(dt.datetime.strptime(Date, '%Y/%m/%d').date(), P,marker='o', color='g')
				Wallet.numTrades +=1
				print "Sold at Price: $" + str(P) + " Cash on Hand: $" + str(Wallet.cashBalance) + " On: " + str(Date)
		
		#avg = priceTotal/count
		#BuyThresh = (avg/High)-0.1
		
			#Conditions to Buy
		if Data.count>5 and Data.Low!=Data.High:#Ensures we have seen at least 5% of the data before we decie to buy
			normPrice = (P-Data.Low)/(Data.High-Data.Low)
			normAvg = (Data.avg50D-Data.Low)/(Data.High-Data.Low)
			if Wallet.price==0: #If we have not bought in
				BuyConf=MLP_B.Run([[normPrice,Data.ROC[-1],Data.ROCROC[-1]]])
				if BuyConf[0]>0.9:
					print "Buy Confidence = " + str(BuyConf)
					Wallet.Buy(P,Wallet.cashBalance)
					Date = Date.replace('-','/')
					plt.plot(dt.datetime.strptime(Date, '%Y/%m/%d').date(), P,marker='o', color='r')
					print "Bought at Price: $" + str(P) + " Investing: $" + str(Wallet.stockBalance) + " On: " + str(Date)
		Data.count +=1
		
	print "Cash Balance = $" + str(Wallet.cashBalance)
	print "Stock Balance = $" + str(Wallet.stockBalance)
	print "Total Profit = $" + str(Wallet.cashBalance+Wallet.stockBalance - Wallet.origBalance) + "   %" + str(100*(Wallet.cashBalance+Wallet.stockBalance - Wallet.origBalance)/Wallet.origBalance)
	print "Total Trades = " + str(Wallet.numTrades)
	#print "Profitting: $" + str(Wallet.CalcProfit(P,Wallet.quantity))
	#print Data.High
	#print Data.PriceList
	#print Data.ROC
	#print Data.ROCROC
	
	
	
	
def Main(ticker,startDate,endDate,num,MLP_S,MLP_B):	

	Stock = Share(ticker)	
	History = getHistory(Stock,startDate,endDate)
	Wallet = Bank(20000)
	length = len(History.values())
	#print np.std(sortPrices(History))
	#print ROCALL(History,startDate,endDate)
	y = []
	x = []
	
	for key in sorted(History):
		y.append(float(History[key]))
		key = key.replace('-','/')
		x.append(dt.datetime.strptime(key, '%Y/%m/%d').date())

	plt.figure(num)
	plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d'))
	plt.gca().xaxis.set_major_locator(mdates.DayLocator())
	plt.plot(x, y,marker='o', color='b')
	plt.gcf().autofmt_xdate()
	plt.xlabel('Time')
	plt.ylabel('Price')
	Trade(History,Wallet,Stock,MLP_S,MLP_B)
	plt.title('Stock Price of ' + ticker + " Total Profit: $" + str(round(Wallet.cashBalance+Wallet.stockBalance - Wallet.origBalance, 2)) + ", %" + str(round(100*(Wallet.cashBalance+Wallet.stockBalance - Wallet.origBalance)/Wallet.origBalance, 0)))
	plt.show()

#Initialize and train NN
NumInputs = 3
Layers =1#1
HiddenN = 10#11
MLP_S = MLP(NumInputs,Layers,HiddenN)
MLP_B = MLP(NumInputs,Layers,HiddenN)
#Sell NN
In_S = [[1,0.03,-0.05],[0,-0.03,0],[1,0.1,-0.12],[0,-0.1,0.2],[0.5,0.01,0.2],[0.95,-0.03,-0.1],[1,0.2,-0.2]]
Out_S = [1,0,1,0,0,1,1]
#In2 = [[0.8,0.03,0],[0.3,-0.03,0],[0.9,0.1,0.12],[0.4,-0.1,-0.2]]

In_B = [[0,-0.07,0.2],[0,-0.03,0.2],[1,0.1,0.12],[0,-0.1,0.1],[0.1,0.03,0.1],[1,0.03,-0.05],[0.1,-0.02,0.1], [0.5,-0.02,0.1]]
OutB = [1,1,0,1,1,0,1,0]
#In2_B = [[0.8,0.03,0],[0,-0.03,0.1],[0.9,0.1,0.12],[0.4,-0.1,-0.2],[0.4,0.01,0.2]]


epochs = 1000#1000
#runTraining(MLP_S, 'training_data', 20, 2)
#runTraining(MLP_B, 'training_data', 20, 1)
MLP_S.Train(In_S,Out_S,epochs)
MLP_B.Train(In_B,OutB,epochs)
#MLP_S.Run(In2)
#--------------------------------------	
startDate = '2014-01-01'
endDate = '2016-08-10' 	
tickers = ['GOOG','TSLA','AAPL','LULU','IBM','GM','BA']
num = 1
for ticker in tickers:
	Main(ticker,startDate,endDate,num,MLP_S,MLP_B)
	num +=1
