import pandas as pd
from kmodes.kmodes import KModes
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("less developed countries 1990 to 1995 clustering.csv")

df_discrete_values = df[["attacktype1", "targtype1", "weaptype1"]] 



'''
# elbow method to find optimal number of clusters 

cost = []
for i in range(1,11):
    km = KModes(n_clusters=i, init = "Huang", n_init = 10)
    km.fit_predict(df_discrete_values)
    cost.append(km.cost_)
    
x = np.array([i for i in range(1,11)])

plt.plot(x,cost)

plt.show()

'''




km = KModes(n_clusters=3, init = "Huang", n_init = 10)

fit_clusters = km.fit_predict(df_discrete_values)

df = df.reset_index()

df_clusters = pd.DataFrame(fit_clusters)

df_clusters.columns = ["cluster"]

df_with_clusters = pd.concat([df, df_clusters], axis = 1).reset_index()





cluster_0 = df_with_clusters[df_with_clusters["cluster"]==0] 

cluster_0 = cluster_0["attacktype1_txt"] + ", " + cluster_0["targtype1_txt"] + ", " + cluster_0["weaptype1_txt"] 

cluster_0 = cluster_0.groupby(cluster_0).count() 

cluster_0 = cluster_0.sort_values(ascending = False)  

cluster_0.to_csv("cluster_0.csv") 




cluster_1 = df_with_clusters[df_with_clusters["cluster"]==1] 

cluster_1 = cluster_1["attacktype1_txt"] + ", " + cluster_1["targtype1_txt"] + ", " + cluster_1["weaptype1_txt"] 

cluster_1 = cluster_1.groupby(cluster_1).count() 

cluster_1 = cluster_1.sort_values(ascending = False)  

cluster_1.to_csv("cluster_1.csv") 




cluster_2 = df_with_clusters[df_with_clusters["cluster"]==2] 

cluster_2 = cluster_2["attacktype1_txt"] + ", " + cluster_2["targtype1_txt"] + ", " + cluster_2["weaptype1_txt"] 

cluster_2 = cluster_2.groupby(cluster_2).count() 

cluster_2 = cluster_2.sort_values(ascending = False)  

cluster_2.to_csv("cluster_2.csv") 


'''
cluster_3 = df_with_clusters[df_with_clusters["cluster"]==3] 

cluster_3 = cluster_3["attacktype1_txt"] + ", " + cluster_3["targtype1_txt"] + ", " + cluster_3["weaptype1_txt"] 

cluster_3 = cluster_3.groupby(cluster_3).count() 

cluster_3 = cluster_3.sort_values(ascending = False)  

cluster_3.to_csv("cluster_3.csv") 



cluster_4 = df_with_clusters[df_with_clusters["cluster"]==4] 

cluster_4 = cluster_4["attacktype1_txt"] + ", " + cluster_4["targtype1_txt"] + ", " + cluster_4["weaptype1_txt"] 

cluster_4 = cluster_4.groupby(cluster_4).count() 

cluster_4 = cluster_4.sort_values(ascending = False)  

cluster_4.to_csv("cluster_4.csv") 



cluster_5 = df_with_clusters[df_with_clusters["cluster"]==5] 

cluster_5 = cluster_5["attacktype1_txt"] + ", " + cluster_5["targtype1_txt"] + ", " + cluster_5["weaptype1_txt"] 

cluster_5 = cluster_5.groupby(cluster_5).count() 

cluster_5 = cluster_5.sort_values(ascending = False)  

cluster_5.to_csv("cluster_5.csv") 




cluster_6 = df_with_clusters[df_with_clusters["cluster"]==6] 

cluster_6 = cluster_6["attacktype1_txt"] + ", " + cluster_6["targtype1_txt"] + ", " + cluster_6["weaptype1_txt"] 

cluster_6 = cluster_6.groupby(cluster_6).count() 

cluster_6 = cluster_6.sort_values(ascending = False)  

cluster_6.to_csv("cluster_6.csv") 




cluster_7 = df_with_clusters[df_with_clusters["cluster"]==7] 

cluster_7 = cluster_7["attacktype1_txt"] + ", " + cluster_7["targtype1_txt"] + ", " + cluster_7["weaptype1_txt"] 

cluster_7 = cluster_7.groupby(cluster_7).count() 

cluster_7 = cluster_7.sort_values(ascending = False)  

cluster_7.to_csv("cluster_7.csv") 




cluster_8 = df_with_clusters[df_with_clusters["cluster"]==8] 

cluster_8 = cluster_8["attacktype1_txt"] + ", " + cluster_8["targtype1_txt"] + ", " + cluster_8["weaptype1_txt"] 

cluster_8 = cluster_8.groupby(cluster_8).count() 

cluster_8 = cluster_8.sort_values(ascending = False)  

cluster_8.to_csv("cluster_8.csv") 




cluster_9 = df_with_clusters[df_with_clusters["cluster"]==9] 

cluster_9 = cluster_9["attacktype1_txt"] + ", " + cluster_9["targtype1_txt"] + ", " + cluster_9["weaptype1_txt"] 

cluster_9 = cluster_9.groupby(cluster_9).count() 

cluster_9 = cluster_9.sort_values(ascending = False)  

cluster_9.to_csv("cluster_9.csv") 
'''
