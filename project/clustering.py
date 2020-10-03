import pandas as pd
from kmodes.kmodes import KModes
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("developed countries recent five years clustering.csv")

df_discrete_values = df[["attacktype1", "targtype1", "weaptype1"]] 


'''
cost = []
for i in range(1,11):
    km = KModes(n_clusters=i, init = "Huang", n_init = 10)
    km.fit_predict(df_discrete_values)
    cost.append(km.cost_)
    
x = np.array([i for i in range(1,11)])

plt.plot(x,cost)

plt.show()
'''




km = KModes(n_clusters=6, init = "Huang", n_init = 10)

fit_clusters = km.fit_predict(df_discrete_values)

df = df.reset_index()

df_clusters = pd.DataFrame(fit_clusters)

df_clusters.columns = ["cluster"]

df_with_clusters = pd.concat([df, df_clusters], axis = 1).reset_index()




df_cluster_0 = df_with_clusters[df_with_clusters["cluster"]==0]

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

ax.scatter(df_cluster_0.attacktype1, df_cluster_0.weaptype1, df_cluster_0.targtype1, c="red", marker='o')

ax.set_xlabel("attack type")
ax.set_ylabel("weapon type")
ax.set_zlabel("target type")

plt.show()




