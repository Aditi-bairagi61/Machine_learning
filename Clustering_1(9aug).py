# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 08:52:25 2024

@author: HP
"""

#9 august 2024
#clustering

import pandas as pd
import matplotlib.pylab as plt
#now import file from data set and create a dataframe
univ1=pd.read_excel("D:/DS/7-Clustering/University_Clustering.xlsx")
a=univ1.describe()
#we have one column 'State' Which really not useful we will drop it
univ = univ1.drop("State", axis=1)
#we know that there is scale differnces among the columns
#which we have to remove
#either by using normalization or standardization
#whenever there is mixed data apply normalization 
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return x
#now apply this normalization function to univ dataframe
#for all the rows and columns from 1 untile end
#since 0th column has university name hence skipped
df_norm=norm_func(univ.iloc[:,1:])
#you can check the df_norm dataframe which is scaled
#between values from 0 to 1
#you can apply describe function to new dataframe
b=df_norm.describe()
#before you apply clustering , you need to plot dendogram first
#now to create dendogram,we need to measure distance,
#we have to import linkage
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch
#linkage function gives us hierarchical or aglomerative clustering
#ref the help for linkage
z = linkage(df_norm, method="complete", metric="euclidean")
plt.figure(figsize=(15,8));
plt.title("Hirarchical clustering dendogram")
plt.xlabel("Index")
plt.ylabel("Distance")
#ref help of dendogram
#sch.dendogram(z)
sch.dendrogram(z, leaf_rotation=0, leaf_font_size=10)
plt.show()


#Dendrogram()
#applying agglomerative clustering choosing 5 as clusters from dendrogram
#whenever has been displayed in dendrogram is not clustering
#it is just showing number of possible clusters
from sklearn.cluster import AgglomerativeClustering
h_complete=AgglomerativeClustering(n_clusters=3,linkage='complete',affinity='euclidean').fit(df_norm)
#apply labels to the clusters
h_complete .labels_
cluster_labels=pd.Series(h_complete.labels_)
#assign this series to univ dataframe as column and name of column
univ['clust']=cluster_labels
#we want to relocate the column 7 to 0 th position
univ1=univ.iloc[:,[7,1,2,3,4,5,6]] 
#now check the univ1 dataframe 
univ1.iloc[:,2:].groupby(univ1.clust).mean()
#from the output cluster 2 has got highest to 10
#lowest accept ratio , best faculty ratio and highest expenses
#highest graduates ratio
univ1.to_csv
