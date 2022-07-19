
import pandas as pd    
from ECM import ECM_CLUST
import matplotlib.pyplot as plt

dataframe = pd.read_csv('GasFurnace205pts.csv', usecols = [1,2,3] )
ts = dataframe.values
cc = ECM_CLUST(ts = ts, Dthr = 0.2)
lineObjects = plt.plot(cc) 
plt.legend(iter(lineObjects), ('Error', 'Original', 'Predicted'))

