import pandas as pd
from sklearn.model_selection import train_test_split
from mealpy.swarm_based.BFO import ABFO
from objective_function import *
from savepkl import *
import numpy as np


#data_gen
data=pd.read_csv('C:\\Users\\ASUS\\Desktop\\ISO-NE_case2.csv')
data=data.drop([0,1,2])
data=data.dropna(axis=1)
data=data.astype(float)
columns_to_drop = [67,68,69,70,86,87,88,89,125]
data=data.drop(data.columns[columns_to_drop],axis=1)
data['label']=data.iloc[:,-1].apply(lambda x:1 if x>-1 else 0)
label=data.iloc[:,-1]
data=data.drop(columns=['label','Sub:12:Ln:28.4'])
label=np.array(label)



#feature_selecation
data['kurtosis']=data.kurtosis(axis=1)
data['skew']=data.skew(axis=1)
column = ['skew','kurtosis']
data['corr']=data[column].corr().mean().mean()
feat=np.array(data)



X_train,X_test,Y_train,Y_test=train_test_split(feat,label,test_size=0.3)

save('X_train',X_train)
save('X_test',X_test)
save('Y_train',Y_train)
save('Y_test',Y_test)



lb = (np.zeros([1, X_train.shape[1]]).astype('int16'))
ub = (np.ones([1, X_train.shape[1]]).astype('int16'))
problem_dict1 = {
         "fit_func": obj_fun1 ,
         "lb": lb,
         "ub": ub,
         "minmax": "min",
     }

epoch = 5
pop_size = 50
C_s=0.1
C_e=0.001
Ped = 0.01
Ns = 4
N_adapt = 2
N_split = 40
model = ABFO(epoch, pop_size, C_s, C_e, Ped, Ns, N_adapt, N_split)
best_position, best_fitness = model.solve(problem_dict1)
print(f"Solution: {best_position}, Fitness: {best_fitness}")






