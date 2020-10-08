import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import	KMeans
import matplotlib.pylab as plt
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import linkage 
import scipy.cluster.hierarchy as sch
from sklearn.cluster import	AgglomerativeClustering 
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import classification_report,f1_score,precision_score,recall_score,accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression,LinearRegression,BayesianRidge,Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,accuracy_score,mean_squared_error,mean_absolute_error
from sklearn.metrics import roc_auc_score
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier,AdaBoostClassifier,RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import RFE
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix

ser1 = pd.read_csv('C:\\Users\\ACER\\Desktop\\project\\P3_1.csv')

# convert opened_time column into two column opened_date & opened_time
ser1['opened_time'] = pd.to_datetime(ser1['opened_time'])
ser1['opened_Date'] = ser1['opened_time'].dt.strftime('%d%m%Y')
ser1['opened_Time'] = ser1['opened_time'].dt.strftime('%H%M')

# convert created_at column into two column created_date & created_time
ser1['created_at'] = pd.to_datetime(ser1['created_at'])
ser1['created_Date'] = ser1['created_at'].dt.strftime('%d%m%Y')
ser1['created_Time'] = ser1['created_at'].dt.strftime('%H%M')

# convert updated_at column into two column updated_date & updated_time
ser1['updated_at'] = pd.to_datetime(ser1['updated_at'])
ser1['updated_Date'] = ser1['updated_at'].dt.strftime('%d%m%Y')
ser1['updated_Time'] = ser1['updated_at'].dt.strftime('%H%M')

indexser1 = ser1.columns
print(indexser1)
# select necessary columns & make new dataframe
ser2 = ser1.iloc[:,[0,1,2,3,4,5,6,7,9,11,13,14,15,16,17,18,19,20,21,22,23,24,25,26]]
ser2.head()
ser2.shape                     
indexser2 = ser2.columns
print(indexser2) 

# convert string into unique integer value. 
idstaser=ser2['ID_status'].unique()
idstaser1={}
for i in range(9):
    idstaser1[i]=idstaser[i]
idstaser2=ser2['ID_status']
for i in range(len(idstaser2)): 
    for j in range(9):
        if idstaser2[i]==idstaser1[j]:
            idstaser2[i]=j
 
            
ser2.loc[ser2['active']==True,'active']=1
ser2.loc[ser2['active']==False,'active']=0

ser2.loc[ser2['Doc_knowledge']==True,'Doc_knowledge']=1
ser2.loc[ser2['Doc_knowledge']==False,'Doc_knowledge']=0

ser2.loc[ser2['confirmation_check']==True,'confirmation_check']=1
ser2.loc[ser2['confirmation_check']==False,'confirmation_check']=0

# function for split string & print only integer.
####
 def abcser(m):
    for col in m:
        updatser1=[]
        updatser=ser2[col]
        for i in range(len(ser2[col])):
            j = updatser[i]
            l = j.split()
            r = l[1]
            updatser1.append(r)
        ser2[col]=updatser1
ser2.columns
m=['ID_caller','location','category_ID','user_symptom','Support_group','support_incharge']
abcser(m)
#####
def abser(n):
    for col in n:
        updatser1=[]
        updatser=ser2[col]
        for i in range(len(ser2[col])):
            j = updatser[i]
            l = j.split()
            r = l[2]
            updatser1.append(r)
        ser2[col]=updatser1
n=['opened_by','Created_by','updated_by']
abser(n)
#######
def aser(g):
    for col in g:
        updatser1=[]
        updatser=ser2[col]
        for i in range(len(ser2[col])):
            j = updatser[i]
            l = j.split()
            r = l[0]
            updatser1.append(r)
        ser2[col]=updatser1
g=['impact']
aser(g)

#####
# convertr string into integer.
ser2['ID_status'] = ser2['ID_status'].astype(int)
ser2['active'] = ser2['active'].astype(int)
ser2['count_reassign'] = ser2['count_reassign'].astype(int)
ser2['count_opening'] = ser2['count_opening'].astype(int)
ser2['count_updated'] = ser2['count_updated'].astype(int)
ser2['ID_caller'] = ser2['ID_caller'].astype(int)
ser2['opened_by'] = ser2['opened_by'].astype(int)
ser2['Created_by'] = ser2['Created_by'].astype(int)
ser2['updated_by'] = ser2['updated_by'].astype(int)
ser2['location'] = ser2['location'].astype(int)
ser2['category_ID'] = ser2['category_ID'].astype(int)
ser2['user_symptom'] = ser2['user_symptom'].astype(int)
ser2['impact'] = ser2['impact'].astype(int)
ser2['Support_group'] = ser2['Support_group'].astype(int)
ser2['support_incharge'] = ser2['support_incharge'].astype(int)
ser2['Doc_knowledge'] = ser2['Doc_knowledge'].astype(int)
ser2['confirmation_check'] = ser2['confirmation_check'].astype(int)
ser2['opened_Date'] = ser2['opened_Date'].astype(int)
ser2['opened_Time'] = ser2['opened_Time'].astype(int)
ser2['created_Date'] = ser2['created_Date'].astype(int)
ser2['created_Time'] = ser2['created_Time'].astype(int)
ser2['updated_Date'] = ser2['updated_Date'].astype(int)
ser2['updated_Time'] = ser2['updated_Time'].astype(int)

#################################################################

se2 = ser2.columns
print(se2)
axz1 = ser2.drop_duplicates()

# remove INC from ID column & print only integer.
acbser1 = ser2['ID']
acbser_1 = acbser1.tolist()
acbser2 = []
for i in range(len(acbser_1)):
    vg = acbser_1[i]
    vg_new = vg.strip("INC")
    acbser2.append(vg_new)
ser2['ID'] = acbser2                            # store in dataframe.
ser2['ID'] = ser2['ID'].astype(int)
####################################################

# function for features importance

def modelfit(alg, dtrain, predictors, performCV=True, printFeatureImportance=True, cv_folds=5):
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain['impact'])
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    #dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
    
    #Perform cross-validation:
    if performCV:
        cv_score = cross_val_score(alg, dtrain[predictors], dtrain['impact'], cv=cv_folds, scoring='roc_auc_ovr')
    
    #Print model report:
    print ("\nModel Report")
    print ("Accuracy : %.4g" % metrics.accuracy_score(dtrain['impact'].values, dtrain_predictions))
    #print ("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['impact'], dtrain_predprob))
    
    if performCV:
        print ("CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))
        
    #Print Feature Importance:
    if printFeatureImportance:
        feat_imp = pd.Series(alg.feature_importances_, predictors).sort_values(ascending=False)
        feat_imp.plot(kind='bar', title='Feature Importances')
        plt.ylabel('Feature Importance Score')

# check features importance 

ser4 = ser2.drop(['impact'],axis=1)
predictors1 = [x for x in ser4.columns]
rafc1 = RandomForestClassifier(random_state=42)
modelfit(rafc1, ser2, predictors1)

##################################################
xz = ser2.columns
se2 = ser2.iloc[:,[0,1,3,5,6,7,8,9,10,11,12,13,14,15]]
se2.columns
se2index = se2.columns

##########################################################################
#################### model  3 #####################################
########## 3 decission tree classifier  ###################
import lightgbm as lgb
y23 = se2['impact']
x23 = se2.drop(['impact'],axis=1)
X23_train, X23_test, Y23_train, Y23_test = train_test_split(x23, y23, test_size=0.33, random_state=42)
fr1 =  DecisionTreeClassifier(class_weight='balanced')
fr1.fit(X23_train,Y23_train)
pred23 = fr1.predict(X23_test)
accuracy_score(Y23_test,pred23)
confusion_matrix(Y23_test,pred23)

X23_test.columns

pd.DataFrame(X23_test, columns=['ID', 'ID_status', 'count_reassign', 'count_updated', 'ID_caller',
       'opened_by', 'Created_by', 'updated_by', 'location', 'category_ID',
       'user_symptom', 'Support_group', 'support_incharge']).to_csv('test.csv',index=False)


l=list(X23_test['ID'])
data=[]
for row in pred23:
    data.append(row)
    
o=pd.DataFrame(list(zip(l,data)),columns=['ID','res'])

rdd=[]
for i in range(len(o)):
    print(o.iloc[i,:])
    break
import csv
k='C:\\Users\\ACER\\Desktop\\project\\P3_1.csv'
with open(k) as file:
    csvfile=csv.reader()

for i in data:
    print(i)
    break


import pickle
# Saving model to disk
#pickle.dump(fr1, open('model.pkl','wb'))

# Saving model to disk
pickle.dump(fr1, open('model.pkl','wb'))

#hj=open("model.pickle",'wb')
#pickle.dump(fr1,hj)
#hj.close()

#######################################################################
jhhj = pickle.load(open('model.pickle', 'rb'))

hk=[6896,4,1,5,4375,108,52,21,108,57,10,70,111]
hk=np.reshape(hk,(1,13))

l=jhhj.predict(hk)
print(l)

round(l[0], 2)























































