# EvolvingClusteringMethod
EvolvingClusteringMethod
@Example :

"""
import pandas as pd    
dataframe = pd.read_csv('GasFurnace205pts.csv', usecols = [1,2,3] )
ts = dataframe.values
cc = ECM_CLUST(ts = ts, Dthr = 0.2)
"""
