# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 18:25:19 2022

@author: Gabriel
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


"Cargar Datos"

url= "bbdd_grupo_GDV.csv"
# load dataset into Pandas DataFrame
df = pd.read_csv(url, header=0)
print(df)
print(df.shape)


"Normalizar Datos"

# Separating out the features
x = df.iloc[:, 1:].values
# Separating out the target
y = df.iloc[:,0].values
# Standardizing the features
x = StandardScaler().fit_transform(x)

"PCA y proyecciones bidimensionales"

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
finalDf = pd.concat([principalDf, df[['Region']]], axis = 1)

"Graficar"

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('PCA Bidimensional', fontsize = 20)
regiones = ['AP','TA','AN','AT','CO','VA','RM','LI','ML','NB','BI','AR','LR','LL','AI','MA']
i = 0
for region in regiones:
    indicesToKeep = i
    
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                , finalDf.loc[indicesToKeep, 'principal component 2']
                , s = 50)
    ax.text(finalDf.loc[indicesToKeep, 'principal component 1']
            , finalDf.loc[indicesToKeep, 'principal component 2']
            , regiones[i]
            )
            
    i = i+1
ax.legend(regiones)
ax.grid()
plt.savefig('bbdd_7', dpi='figure', format=None, metadata=None,
        bbox_inches=None, pad_inches=0.1,
        facecolor='auto', edgecolor='auto',
        backend=None)
plt.show()

print(pca.explained_variance_ratio_)

b = pd.DataFrame(df.columns[1:])
b = pd.concat([b, pd.DataFrame(pca.components_[0,:])], axis = 1)
b = pd.concat([b, pd.DataFrame(pca.components_[1,:])], axis = 1)
b.columns = ['Variable','PC1','PC2']



print(b)

