import pySI as SI
import pandas as pd
import os

data = pd.read_csv(os.path.join(os.path.dirname(__file__), "SIMODELTESTDATA1.csv"))
model = SI.calibrate(data=data, origins='Origin', destinations='Destination', trips='Data', sep='Dij', cost='exp', constraints={'production':'Origin', 'attraction':'Destination'})
results = model.mle(initialParams={'beta':0})
print results.results.sumStr

data = pd.read_csv(os.path.join(os.path.dirname(__file__), "SIMODELTESTDATA2.csv"))
model = SI.calibrate(data=data, origins='Origin', destinations='Destination', trips='Data', sep='Dij', cost='exp', factors ={'destinations':['Pop']}, constraints={'production':'Origin'})
results = model.mle(initialParams={'beta':0, 'Pop':1})
print results.results.sumStr