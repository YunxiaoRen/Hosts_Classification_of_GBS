# Load the necessary libraries 
from sklearn.datasets import make_classification 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression 
from sklearn.calibration import calibration_curve 
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd

rf_prob = pd.read_csv("rf_predicteds_probs.csv",index_col=0)
rf_prob_ar = rf_prob.to_numpy()
test_label = pd.read_csv("testData_label_num.csv",index_col=0)
test_label_ar = test_label.to_numpy()
my_y_test = test_label_ar.reshape(test_label_ar.shape[0],)

# Compute the calibration curve for each class 
calibration_curve_values = [] 
for i in range(4): 
    curve = calibration_curve(my_y_test == i,  
                              rf_prob_ar[:, i],  
                              n_bins=5,  
                              pos_label=True) 
    calibration_curve_values.append(curve) 
  
# Plot the calibration curves 
fig, axs = plt.subplots(1, 4, figsize=(17,5)) 
for i in range(4): 
    axs[i].plot(calibration_curve_values[i][1],  
                calibration_curve_values[i][0],  
                marker='o') 
    axs[i].plot([0, 1], [0, 1], linestyle='--') 
    axs[i].set_xlim([0, 1]) 
    axs[i].set_ylim([0, 1]) 
    axs[i].set_title(f"Class {i}", fontsize = 17) 
    axs[i].set_xlabel("Predicted probability", fontsize = 15) 
    axs[i].set_ylabel("True probability", fontsize = 15) 
plt.tight_layout() 
plt.savefig("RF_calibration_out_nbin5.pdf")
plt.show()


## LR
lr_prob = pd.read_csv("lr_predicteds_probs.csv",index_col=0)
lr_prob_ar = lr_prob.to_numpy()
test_label = pd.read_csv("testData_label_num.csv",index_col=0)
test_label_ar = test_label.to_numpy()
my_y_test = test_label_ar.reshape(test_label_ar.shape[0],)

# Compute the calibration curve for each class 
calibration_curve_values = [] 
for i in range(4): 
    curve = calibration_curve(my_y_test == i,  
                              lr_prob_ar[:, i],  
                              n_bins=5,  
                              pos_label=True) 
    calibration_curve_values.append(curve) 
  
# Plot the calibration curves 
fig, axs = plt.subplots(1, 4, figsize=(17,5)) 
for i in range(4): 
    axs[i].plot(calibration_curve_values[i][1],  
                calibration_curve_values[i][0],  
                marker='o') 
    axs[i].plot([0, 1], [0, 1], linestyle='--') 
    axs[i].set_xlim([0, 1]) 
    axs[i].set_ylim([0, 1]) 
    axs[i].set_title(f"Class {i}", fontsize = 17) 
    axs[i].set_xlabel("Predicted probability", fontsize = 15) 
    axs[i].set_ylabel("True probability", fontsize = 15) 
plt.tight_layout() 
plt.savefig("LR_calibration_out_nbin5.pdf")
plt.show()


## SVM
svm_prob = pd.read_csv("svm_predicteds_probs_poly.csv",index_col=0)
svm_prob_ar = svm_prob.to_numpy()
test_label = pd.read_csv("testData_label_num.csv",index_col=0)
test_label_ar = test_label.to_numpy()
my_y_test = test_label_ar.reshape(test_label_ar.shape[0],)

# Compute the calibration curve for each class 
calibration_curve_values = [] 
for i in range(4): 
    curve = calibration_curve(my_y_test == i,  
                              svm_prob_ar[:, i],  
                              n_bins=5,  
                              pos_label=True) 
    calibration_curve_values.append(curve) 
  
# Plot the calibration curves 
fig, axs = plt.subplots(1, 4, figsize=(17,5)) 
for i in range(4): 
    axs[i].plot(calibration_curve_values[i][1],  
                calibration_curve_values[i][0],  
                marker='o') 
    axs[i].plot([0, 1], [0, 1], linestyle='--') 
    axs[i].set_xlim([0, 1]) 
    axs[i].set_ylim([0, 1]) 
    axs[i].set_title(f"Class {i}", fontsize = 17) 
    axs[i].set_xlabel("Predicted probability", fontsize = 15) 
    axs[i].set_ylabel("True probability", fontsize = 15) 
plt.tight_layout() 
plt.savefig("SVM_calibration_out_nbin5.pdf")
plt.show()