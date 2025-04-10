import pandas as pd
df=pd.read_csv('Clean_loyal.csv')
from sklearn.model_selection import train_test_split
x=df.drop(["satisfaction"],axis=1)
y=df[["satisfaction"]]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
from sklearn.ensemble import RandomForestClassifier
obj=RandomForestClassifier(n_estimators=100,random_state=42)
modal1=obj.fit(x_train,y_train)
import joblib as jb
jb.dump(modal1,'voting_modal.lib')
import pickle as pkl
pkl.load('voting_modal.lib')