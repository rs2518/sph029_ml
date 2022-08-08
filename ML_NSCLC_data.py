#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 16:40:39 2019

@author: raphaelsinclair
"""



########## DATA PRE-PROCESSING

# Dataframe
import pandas as pd
# Linear algebra
import numpy as np

# Load dataframes
path = '/Users/raphaelsinclair/Desktop/MSc Health Data Analytics - IC/HDA/SPH029 - Machine Learning/Project'

df_clin_info = pd.read_table(path+'/Clinicalinformation.txt', index_col = 0)
df_clin_info.head()

df = pd.read_table(path+'/Combat_filtered_exprs.txt').T    # Transpose pre-processed data (rows = subjects, columns = gene)
print(df.head())
print(df.shape)    # Size of data (n,p) = (1118, 10077)
print(df.index)    # Row names are sample IDs


# RECODING VARIABLES

# Rename inconvenient columns
df_clin_info.columns     # Look at column names
df_clin_info.columns = ['Dataset', 'Disease Status', 'Gender', 'Race', 'Age', 'Stage', 'Histology', 'Overall survival (month)', 'Death', 'Smoking', 'TNM stage (T)', 'TNM stage (N)', 'TNM stage (M)', 'Recurrence', 'Others']


# Tabulate disease status to observe missing data
print(df_clin_info['Disease Status'].value_counts() )     # View counts of disease status
print(12*'-')
print(df_clin_info['Histology'].value_counts(dropna = False) )     # View counts of disease subtype
print(12*'-')
print(df_clin_info.groupby('Disease Status')['Histology'].value_counts(dropna = False) )     # Counts of subtype by status
# Note: Controls with missing subtype are actually 'Healthy' subtype
# and low numbers of some subtypes may be problematic for subtype analysis


# Recode 'Healthy' samples with missing subtype
df_clin_info.loc[((df_clin_info['Histology'].isnull()) & (df_clin_info['Disease Status'] == 'Normal')) ,'Histology'] = 'Healthy' # Replace 'false' missing values
print(df_clin_info.groupby('Disease Status')['Histology'].value_counts(dropna = False) )    # Normal samples with missing subtypes should be Healthy now


# Recode subtypes with low number of samples as missing
df_clin_info['Histology 2'] = df_clin_info['Histology']
low_sample_subtypes = ['Large cell Neuroendocrine carcinoma', 'Adenosquamous carcinoma', 'Other', 'NSCLC-favor adenocarcinoma', 'NSClarge cell carcinoma-mixed']     # list of subtypes to remove
for subtype in low_sample_subtypes:
    df_clin_info.loc[(df_clin_info['Histology 2'] == subtype), 'Histology 2'] = np.NaN
print(df_clin_info.groupby('Disease Status')['Histology 2'].value_counts(dropna = False) )     # Listed subtypes should be missing now


# Recode disease status labels (0 = Normal, 1 = NSCLC)
df_clin_info['Disease status labels'] = np.where(df_clin_info['Disease Status'] == 'NSCLC', 1, 0)


# MERGE DATAFRAMES
 
# Create merged dataframe with covariates and labels. Then create data arrays
clin_info_columns = ['Disease Status', 'Disease status labels', 'Histology', 'Histology 2']     # Create list of column headers from df_clin_info that we want to merge
df_merged = df.merge(df_clin_info[clin_info_columns], how = 'inner', left_index = True, right_index = True)


# PREPARING FEATURES AND LABELS

# Train_test_split function
from sklearn.model_selection import train_test_split

# Create arrays for covariates and disease status labels
y = df_merged['Disease status labels'].values
X = df_merged.drop(['Disease Status', 'Disease status labels', 'Histology', 'Histology 2'], axis = 1).values


# Split data into training set and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42, stratify = y)     # Splits ALL data into train, test sets according to labels

# View counts of each class in test and train set
print(np.array(np.unique(y_test, return_counts=True)))
print(np.array(np.unique(y_train, return_counts=True)))




########## SUPERVISED LEARNING

# IMPORT LIBRARIES

# Normalise data
from sklearn.preprocessing import StandardScaler
# Pipeline object
from sklearn.pipeline import Pipeline
# Classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
# Cross-validation
from sklearn.model_selection import GridSearchCV
# Metrics
from sklearn.metrics import confusion_matrix, classification_report, roc_curve
# Visualisation
import matplotlib.pyplot as plt


###########################################

# KNN CLASSIFIER

# Create pipeline
steps = [('scaler', StandardScaler()),('knn', KNeighborsClassifier())]
knn_pipeline = Pipeline(steps)

# Cross-validation
k = [1] + list(range(10,121,10))
parameter = {'knn__n_neighbors': k}
folds = 4
knn_cv = GridSearchCV(knn_pipeline, parameter, cv = folds)

# Fit model, compute accuracy of the best score and return the best parameter
knn_cv.fit(X_train, y_train)
print(12*'-')
print("Best score is {}".format(knn_cv.best_score_))
print(12*'-')
print("Best choice of k is {}".format(knn_cv.best_params_))
# k=120 best choice


# Try different k
k = list(range(100,151,10))
knn_cv.fit(X_train, y_train)
print(12*'-')
print("Best score is {}".format(knn_cv.best_score_))
print(12*'-')
print("Best choice of k is {}".format(knn_cv.best_params_))
# k=120 best choice

# Try different k
k = list(range(115,126,1))
knn_cv.fit(X_train, y_train)
print(12*'-')
print("Best score is {}".format(knn_cv.best_score_))
print(12*'-')
print("Best choice of k is {}".format(knn_cv.best_params_))
# k=120 best choice


# Predict tuned hyperparameter on test set and view classification report
y_pred_knn = knn_cv.predict(X_test)
print(12*'-')
print("Accuracy: {}".format(knn_cv.score(X_test, y_test)))
print(12*'-')
print(confusion_matrix(y_test, y_pred_knn))
print(12*'-')
print(classification_report(y_test, y_pred_knn))
# Poor classifier. Just assumes all subjects are cases (because of disproportionate cases vs controls)



# KNN now weighted to make closer points have higher influence on classification by assigning heavier weights

steps = [('scaler', StandardScaler()),('knn', KNeighborsClassifier(weights = 'distance'))]
weighted_knn_pipeline = Pipeline(steps)
k = [1] + list(range(10,121,10))
parameter = {'knn__n_neighbors': k}
folds = 3
weighted_knn_cv = GridSearchCV(weighted_knn_pipeline, parameter, cv = folds)
weighted_knn_cv.fit(X_train, y_train)
print(12*'-')
print("Best score is {}".format(weighted_knn_cv.best_score_))
print(12*'-')
print("Best choice of k is {}".format(weighted_knn_cv.best_params_))

k = list(range(110,131,10))
weighted_knn_cv.fit(X_train, y_train)
print(12*'-')
print("Best score is {}".format(weighted_knn_cv.best_score_))
print(12*'-')
print("Best choice of k is {}".format(weighted_knn_cv.best_params_))

k = list(range(117,123,1))
weighted_knn_cv.fit(X_train, y_train)
print(12*'-')
print("Best accuracy score is {}".format(weighted_knn_cv.best_score_))
print(12*'-')
print("Best choice of k is {}".format(weighted_knn_cv.best_params_))

y_pred_w_knn = weighted_knn_cv.predict(X_test)
print(12*'-')
print("Accuracy: {}".format(weighted_knn_cv.score(X_test, y_test)))
print(12*'-')
print(classification_report(y_test, y_pred_w_knn))
# No improvement. Distance metric still distorted by p >> n


# Dataframe of weighted knn test accuracies for each k
weighted_knn_cv_results = pd.DataFrame({**weighted_knn_cv.param_grid, **{"test_score": weighted_knn_cv.cv_results_["mean_test_score"]}})


###########################################

# SVM CLASSIFIER

# Recode 0 to -1 for consistency with mathematical notation (i.e. +ve and -ve class for SVM)
y_train[y_train == 0] = -1
y_test[y_test == 0] = -1

# Create pipeline
steps = [('scaler', StandardScaler()),('SVM', SVC())]
svm_pipeline = Pipeline(steps)

# Cross-validation
c_space = np.logspace(-1,2,4)
gamma_space = np.logspace(-2,1,4)
parameter = {'SVM__C': c_space, 'SVM__gamma': gamma_space}
svm_cv = GridSearchCV(svm_pipeline, parameter, cv = folds)

# Fit model, compute accuracy of the best score and return the best hyperparameters
svm_cv.fit(X_train, y_train)
print(12*'-')
print("Best score is {}".format(svm_cv.best_score_))
print(12*'-')
print("Best choice of parameters is {}".format(svm_cv.best_params_))

y_pred_svm = svm_cv.predict(X_test)
print(12*'-')
print("Accuracy: {}".format(svm_cv.score(X_test, y_test)))
print(12*'-')
print(classification_report(y_test, y_pred_svm))
# No improvement. Possibly because of p >> n. Needs penalisation.
# Unpenalised distance-based classifier not effective


###########################################

# LOGISTIC REGRESSION

# Recode back to 0
y_train[y_train == -1] = 0
y_test[y_test == -1] = 0

# Train logistic regression classifier
steps = [('scaler', StandardScaler()),('logistic', LogisticRegression(class_weight = 'balanced'))]
logreg_pipeline = Pipeline(steps)
c_space = np.logspace(-10, 10, 11)
parameter = {'logistic__C': c_space, 'logistic__penalty': ['l1', 'l2']}
logreg_cv = GridSearchCV(logreg_pipeline, parameter, cv = folds, scoring = 'roc_auc')
logreg_cv.fit(X_train, y_train)

# Print the tuned hyperparameters and training score
print("Tuned Logistic Regression Parameters: {}".format(logreg_cv.best_params_)) 
print("Best AUC score is {}".format(logreg_cv.best_score_))

# Test set results
y_pred_lr = logreg_cv.predict(X_test)
print(12*'-')
print("AUC score on test set: {}".format(logreg_cv.score(X_test, y_test)))
print(12*'-')
print(classification_report(y_test, y_pred_lr))
# Far better!!!

# Compute predicted probabilities: y_pred_prob
y_pred_prob = logreg_cv.predict_proba(X_test)[:,1]

# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()


# Save results as a dataframe
lr_cv_results = pd.DataFrame({**logreg_cv.param_grid, **{"test_score": logreg_cv.cv_results_["mean_test_score"]}})

# Extract coefficients and view non-zero coefficients for gene expression levels
lr_cv_coef = pd.DataFrame(logreg_cv.best_estimator_.named_steps['logistic'].coef_, columns = df.columns).T
lr_cv_coef.columns = ['Coefficients']
len(lr_cv_coef[lr_cv_coef['Coefficients'] != 0])


########## UNSUPERVISED LEARNING

# IMPORT LIBRARIES
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

###########################################

# PREPARE DATA FOR SUBTYPE ANALYSIS

# Missing subtypes omitted
df_na_omitted = df_merged.dropna(axis = 0)

# Convert subtype into numerical labels
y_subtype = df_na_omitted['Histology 2'].astype('category')
subtype_labels = dict(enumerate(y_subtype.cat.categories))     # Dictionary of labels against name of subtype 
y_subtype = y_subtype.cat.codes

X_subtype = df_na_omitted.drop(['Disease Status', 'Disease status labels', 'Histology', 'Histology 2'], axis = 1).values


###########################################

# PCA

# Standard PCA on unnormalised data
pca = PCA(random_state = 42)
pca_features = pca.fit_transform(X_subtype)

pc1 = pca_features[:,0]
pc2 = pca_features[:,1]

plt.scatter(pc1, pc2)
plt.axis('equal')
plt.show()


# View PCA on normalised data by using pipeline
pca_pipeline = Pipeline([('scaler', StandardScaler()), ('pca', PCA(n_components = 2, random_state = 42))])
pca_pipeline.fit_transform(X_subtype)
pca.explained_variance_ratio_[0]
pca.explained_variance_ratio_[1]
# Variance explained is far too small for PC1 and PC2
# Can produce plot to show this

features = range(pca.n_components_)


###########################################

# T-SNE

# t-SNE to visualise non-linear projection of subtypes in 2D
tsne_model = TSNE(learning_rate = 10, perplexity = 50)   
tsne_features = tsne_model.fit_transform(X_subtype)   
xs = tsne_features[:,0]
ys = tsne_features[:,1]

# Scatter plot, coloring by variety_numbers (list of variety numbers for each grain)
plt.scatter(xs, ys, c=list(y_subtype))
plt.legend()
plt.show()
