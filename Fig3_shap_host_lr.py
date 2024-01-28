##########################################################################################
#                                    1.Import Libraries                                  #
##########################################################################################
import pandas as pd
import os
import numpy as np 
######----------For Visualisation
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import figure
#sns.set(rc={'figure.figsize':(15,5)})
#plt.style.use('fivethirtyeight')
######----------For Feature Selection and Modeling
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
######---------For SHAP/Model Explainations
import shap

##########################################################################################
#                                      2.Load Data                                       #
##########################################################################################
DATAPATH = "/data/p1_HK_GBS/01data/shapdata"
train= pd.read_csv(os.path.join(DATAPATH,"host_model_trainData.csv"),index_col=0)
test= pd.read_csv(os.path.join(DATAPATH,"host_model_testData.csv"),index_col=0)
train.head()
# Check columns 
train.info()
# Unique Values in each column
#train.nunique()


##########################################################################################
#                                   4.Model Building                                     #
##########################################################################################
######---------4.1 Ensemble Linear Model

#Seperating target
X_train = train.drop(columns='label',axis=1)
Y_train = train.label
X_test = test.drop(columns='label',axis=1)
Y_test = test.label
#Train Test Split
#X_train,X_test,Y_train,Y_test = train_test_split(Xtrain,Ytrain,test_size=0.3,random_state=1200,stratify = Ytrain)
print(X_train.shape , X_test.shape,Y_train.shape,Y_test.shape)

######--------- 4.1.1 Logistic Regression ---------######
clf_lr = LogisticRegression(random_state=1200, multi_class='multinomial',max_iter=10000) #Set dual = True if number of features > number of examples and vice versa.
clf_lr.fit(X_train,Y_train)


##########################################################################################
#                5.Feature Selection Using Feature Importance                            #
##########################################################################################

######--------- 5.1 Feature selectiong for Random Forest Classifier Fitted Model
model0 = SelectFromModel(clf_lr, prefit=True) #we set prefit true since we are passing a fitted model as parameter
X_train_new = model0.transform(X_train)   #drops unimportant features
print('Shape of transformed Train set:',X_train_new.shape)
feature_names = X_train.columns[model0.get_support()]
print('Feature Names \n',feature_names)
X_test_new = X_test[feature_names] #dropping unnecessary features
#clf_lr_modified.score(X_test,Y_test)
clf_lr_modified= RandomForestClassifier(n_estimators=100,random_state=1200)
clf_lr_modified.fit(X_train_new,Y_train)
clf_lr_modified.score(X_test_new,Y_test)


##########################################################################################
#                 6. SHAP:Linear Explainer-Logistic Regression                           #
##########################################################################################

######--------- 6.1 LinearExplainer---------######
# creating an explainer for our model
masker1 = shap.maskers.Independent(data = X_train)
explainer = shap.LinearExplainer(clf_lr,masker=masker1) 

# finding out the shap values using the explainer
shap_values = explainer.shap_values(X_train)

# Expected/Base/Reference value = the value that would be predicted if we didn’t know any features of the current output”
print('Expected Value:', explainer.expected_value)

# Shap Values for class =0 that is  class
print("Shap Values for 'host 0' class")
pd.DataFrame(shap_values[0],columns=X_train.columns).head()
# Shap Values for class =1 that is  class
print("Shap Values for 'host 3' class")
pd.DataFrame(shap_values[3],columns=X_train.columns).head()


######--------- 6.2 Plot of LinearExplainer---------######


######--------- 6.2.2 SHAP Summary Plot
# Summary plot for all classes
# shap.initjs()
# shap.summary_plot(shap_values, X_train,class_names=clf_lr.classes_)
# plt.savefig('host_train_summaryPlot_allcls_lr.pdf')
# plt.close()

# # Summary for Single Class - A summary plot for single class gives us densit, 
# shap.initjs()
# shap.summary_plot(shap_values[0], X_train,class_names=clf_lr.classes_[0])
# plt.savefig('host_train_summaryPlot_cls0_lr.pdf')
# plt.close()
# shap.initjs()
# shap.summary_plot(shap_values[1], X_train,class_names=clf_lr.classes_[1])
# plt.savefig('host_train_summaryPlot_cls1_lr.pdf')
# plt.close()
# shap.initjs()
# shap.summary_plot(shap_values[2], X_train,class_names=clf_lr.classes_[2])
# plt.savefig('host_train_summaryPlot_cls2_lr.pdf')
# plt.close()
# shap.initjs()
# shap.summary_plot(shap_values[3], X_train,class_names=clf_lr.classes_[3])
# plt.savefig('host_train_summaryPlot_cls3_lr.pdf')
# plt.close()
# As we can see low values of ram drives up the prob of belonging to class 0 .Similarly low values of battery power,px_width,px_height drives up the prob of belonging to class 0. 


######--------- 6.2.3 SHAP Decision Plot
## 3.1 Decision Plot for single observation for a single class 0
shap.decision_plot(explainer.expected_value[0],shap_values[0][0,:], X_train.iloc[0,:],class_names=clf_lr.classes_[0])
plt.savefig('host_train_decisionPlot_cls0_lr.pdf')
plt.close()


######--------------------------------------------------------------------- 6.3 SHAP Values for Test Set -------------------------------------------------------------------------######
# creating an explainer for our model
masker2 = shap.maskers.Independent(data = X_test)
explainer2 = shap.LinearExplainer(clf_lr,masker=masker2)
# finding out the shap values using the explainer
shap_values_test = explainer2.shap_values(X_test)

y_pred=clf_lr.predict(X_test)
misclassified=Y_test!=y_pred
sum(misclassified) #total misclassified test observations

pred_true_tbl=pd.DataFrame({'True':Y_test,'Pred':y_pred,'Misclassified':misclassified})
pred_true_tbl[pred_true_tbl.Misclassified==True]
print(np.where(pred_true_tbl.Misclassified==True))  # Index location of misclassified predictions
# (array([ 21,  86,  91, 135, 161]),)
y_pred[[21,  86,  91, 135, 161]]
#array(['Human', 'Pig', 'Human', 'Human', 'Human', 'Human'], dtype=object)
Y_test[[21,  86,  91, 135, 161]]
# CUHK_fGBS802A_18      Fish
# CUHK_GBS554A_17      Human
# CUHK_pGBS19A_18        Pig
# GCA_000323085.1     Bovine
# GCA_001017935.1     Bovine



######--------- 6.3.1 SHAP Summary Plot ---------######
# Summary plot for all classes
shap.initjs()
shap.summary_plot(shap_values_test, X_test,class_names=clf_lr.classes_)
plt.tight_layout()
plt.savefig('host_test_summaryPlot_allcls_lr.pdf')
plt.close()

# Summary for Single Class - A summary plot for single class gives us densit, 
shap.initjs()
#plt.figure(figsize=(3,6))
shap.summary_plot(shap_values_test[0], X_test,class_names=clf_lr.classes_[0])
plt.tight_layout()
plt.savefig('host_test_summaryPlot_cls0_lr.pdf')
plt.close()
shap.initjs()
shap.summary_plot(shap_values_test[1], X_test,class_names=clf_lr.classes_[1])
plt.tight_layout()
plt.savefig('host_test_summaryPlot_cls1_lr.pdf')
plt.close()
shap.initjs()
shap.summary_plot(shap_values_test[2], X_test,class_names=clf_lr.classes_[2])
plt.tight_layout()
plt.savefig('host_test_summaryPlot_cls2_lr.pdf')
plt.close()
shap.initjs()
shap.summary_plot(shap_values_test[3], X_test,class_names=clf_lr.classes_[3])
plt.tight_layout()
plt.savefig('host_test_summaryPlot_cls3_lr.pdf')
plt.close()


# ######--------- 6.3.2 SHAP Force Plot ---------######

# X_test.iloc[21], 
# Y_test.iloc[21]  #the  observation has prica range of class 0


# ######--------- 6.3.3 SHAP Decision Plot ---------######

# shap.initjs()
# shap.decision_plot(explainer.expected_value[0],shap_values_test[0][21,:], X_test.iloc[21,:])
# plt.tight_layout()
# plt.savefig('host_test_decision_cls0_false21_lr.pdf')
# plt.close()

# shap.initjs()
# shap.decision_plot(explainer.expected_value[1],shap_values_test[1][21,:], X_test.iloc[21,:])
# plt.tight_layout()
# plt.savefig('host_test_decision_cls1_false21_lr.pdf')
# plt.close()


# shap.initjs()
# shap.decision_plot(explainer.expected_value[2],shap_values_test[2][21,:], X_test.iloc[21,:])
# plt.tight_layout()
# plt.savefig('host_test_decision_cls2_false21_lr.pdf')
# plt.close()


# shap.initjs()
# shap.decision_plot(explainer.expected_value[3],shap_values_test[3][21,:], X_test.iloc[21,:])
# plt.tight_layout()
# plt.savefig('host_test_decision_cls3_false21_lr.pdf')
# plt.close()


### multi-output decision plot
# color: https://matplotlib.org/stable/tutorials/colors/colormaps.html
### multi-output decision plot
# color: https://matplotlib.org/stable/tutorials/colors/colormaps.html

feature_names = X_test.columns
labels = ["Bovine", "Fish", "Human","Pig"]
shap.initjs()
row_index = 21
shap.multioutput_decision_plot(explainer.expected_value.tolist(),shap_values_test,row_index=row_index,
                               feature_names=feature_names.to_list(),legend_labels=labels,
                               legend_location='lower right')
plt.tight_layout()
plt.savefig('host_test_decision_allcls_false21_lr.pdf')
plt.close()

feature_names = X_test.columns
labels = ["Bovine", "Fish", "Human","Pig"]
shap.initjs()
row_index = 86
shap.multioutput_decision_plot(explainer.expected_value.tolist(),shap_values_test,row_index=row_index,
                               feature_names=feature_names.to_list(),legend_labels=labels,
                               legend_location='lower right')
plt.tight_layout()
plt.savefig('host_test_decision_allcls_false86_lr.pdf')
plt.close()


feature_names = X_test.columns
labels = ["Bovine", "Fish", "Human","Pig"]
shap.initjs()
row_index = 91
shap.multioutput_decision_plot(explainer.expected_value.tolist(),shap_values_test,row_index=row_index,
                               feature_names=feature_names.to_list(),legend_labels=labels,
                               legend_location='lower right')
plt.tight_layout()
plt.savefig('fig_host_test_decision_allcls_false91_lr.pdf')
plt.close()

feature_names = X_test.columns
labels = ["Bovine", "Fish", "Human","Pig"]
shap.initjs()
row_index = 135
shap.multioutput_decision_plot(explainer.expected_value.tolist(),shap_values_test,row_index=row_index,
                               feature_names=feature_names.to_list(),legend_labels=labels,
                               legend_location='lower right')
plt.tight_layout()
plt.savefig('fig_host_test_decision_allcls_false135_lr.pdf')
plt.close()

feature_names = X_test.columns
labels = ["Bovine", "Fish", "Human","Pig"]
shap.initjs()
row_index = 161
shap.multioutput_decision_plot(explainer.expected_value.tolist(),shap_values_test,row_index=row_index,
                               feature_names=feature_names.to_list(),legend_labels=labels,
                               legend_location='lower right')
plt.tight_layout()
plt.savefig('fig_host_test_decision_allcls_false161_lr.pdf')
plt.close()

