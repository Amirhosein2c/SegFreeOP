
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sklearn.linear_model import Lasso, Ridge
from sklearn.linear_model import LassoCV, RidgeCV
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sklearn.neural_network import MLPRegressor
from sksurv.metrics import concordance_index_censored
from sklearn.feature_selection import SelectKBest, mutual_info_regression, f_regression
from sklearn.pipeline import Pipeline
from sksurv.datasets import load_whas500
from sksurv.preprocessing import OneHotEncoder
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest
from sksurv.svm import FastSurvivalSVM
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA, NMF, FastICA
from sklearn.manifold import TSNE
from sklearn.manifold import LocallyLinearEmbedding
from sksurv.linear_model import CoxPHSurvivalAnalysis
import csv

import warnings
warnings.filterwarnings("ignore")

# Load data
# NETWORK = 'InceptionResNetV2'
# NETWORK = 'InceptionV3'
# NETWORK = 'Resnet_FC'
# NETWORK = 'Resnet'
# NETWORK = 'Resnet_res2c'
# NETWORK = 'Resnet_res3d'
# NETWORK = 'Resnet_res4f'
# NETWORK = 'VGG16'
# NETWORK = 'VGG19'
# NETWORK = 'Xception'



NETWORKS = [
            # 'InceptionResNetV2',
            # 'InceptionV3',
            # 'Resnet_FC',
            # 'Resnet',
            # 'Resnet_res2c',
            # 'Resnet_res3d',
            # 'Resnet_res4f',
            # 'VGG16',
            # 'VGG19',
            # 'Xception',
            # 'DenseNet201',
            # 'EfficientNetB1',
            # 'EfficientNetB4',
            # 'EfficientNetB7',
            # 'EfficientNetV2M',
            # 'NASNetLarge',
            # 'ResNet152V2'
            'ConvNeXtBase'
            ]



METHOD = "Max-Median"

RESULTS_DIRECTORY = "/home/amirhosein/HECKTOR2022/WHOLEIMAGE_MAMIP/Results"
Results_filename = f"SA_Result_Single-Network_MA-MIP_WHOLE-IMAGE_{METHOD}_Series-3.csv"

csv_file = open(os.path.join(RESULTS_DIRECTORY, Results_filename), mode='w')

field_names = ['Network', 
               'Fold_Number', 
               
               'KBest_CoxPHSA_CIndex',
               'ICA_CoxPHSA_CIndex',
               'KBest_ICA_CoxPHSA_CIndex',
               
               'KBest_RSF_CIndex',
               'ICA_RSF_CIndex',
               'KBest_ICA_RSF_CIndex',
               
               'KBest_SVMSA_CIndex',
               'ICA_SVMSA_CIndex',
               'KBest_ICA_SVMSA_CIndex']


with open(os.path.join(RESULTS_DIRECTORY,Results_filename), 'w', newline='') as csv_file:
    csv_writer = csv.DictWriter(csv_file, fieldnames=field_names)
    csv_writer.writeheader()

    for NETWORK in NETWORKS:
        print(NETWORK)

        data = pd.read_csv(f"/home/amirhosein/HECKTOR2022/WHOLEIMAGE_MAMIP/ProcessedFeatures/Processed_Features_{NETWORK}_MA-MIP_WHOLE-IMAGE_{METHOD}.csv")

        data['Relapse'] = data['Relapse'].astype('bool')
        # Define features and target
        features = data.drop(["Relapse", "RFS", "Patient_ID"], axis=1)
        # target = data[["recurrence", "time"]]
        target = data[["Relapse", "RFS"]]

        features.drop(features.columns[features.isna().any() | np.isinf(features).any()], axis=1, inplace=True)

        # Compute correlation matrix
        corr_matrix = features.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
        features = features.drop(to_drop, axis=1)

        # Create structured array for target
        y = np.empty(len(target), dtype=[('Relapse', bool), ('RFS', float)])
        y['Relapse'] = data['Relapse']
        y['RFS'] = data['RFS']

        features = features.to_numpy()
        X = features

        # scaler = StandardScaler()
        # scaler.fit(X)
        # X = scaler.transform(X)

        fold = 1
        kf = KFold(n_splits=5, shuffle=True, random_state=1111)
        
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            print("==================================================================")
            print(f"Fold: {fold}")
            print("==================================================================")
        
            CoxPHSA_CIndex_list = list()
            RSF_CIndex_list = list()
            SVMSA_CIndex_list = list()
            
            # Define the parameter grids for each dimensionality reduction method
            dim_red_params_list = list()
            # pca_params = {'pca__n_components': [5, 10, 15, 20, 25, 30, 35, 40],
            #               'select_features__k': [5, 10, 15, 20, 25, 30]}
            kbest_params = {'select_features__k': [5, 10, 15, 20, 25, 30]}
            dim_red_params_list.append(kbest_params)
            ica_params = {'ica__n_components': [5, 10, 15, 20, 25, 30, 35, 40]}
            dim_red_params_list.append(ica_params)
            ica_kbest_params = {'ica__n_components': [5, 10, 15, 20, 25, 30, 35, 40],
                        'select_features__k': [5, 10, 15, 20, 25, 30]}
            dim_red_params_list.append(ica_kbest_params)
            # Define the pipeline for each dimensionality reduction method
            pipelines_list = list()
            # pca_pipe = Pipeline([('pca', PCA()), 
            #                      ('select_features', SelectKBest()), 
            #                      ('coxph', CoxPHSurvivalAnalysis())])
            kbest_pipe = Pipeline([('select_features', SelectKBest()), 
                                ('coxph', CoxPHSurvivalAnalysis())])
            pipelines_list.append(kbest_pipe)
            ica_pipe = Pipeline([('ica', FastICA()),
                                ('coxph', CoxPHSurvivalAnalysis())])
            pipelines_list.append(ica_pipe)
            kbest_ica_pipe = Pipeline([('ica', FastICA()), 
                                ('select_features', SelectKBest()), 
                                ('coxph', CoxPHSurvivalAnalysis())])
            pipelines_list.append(kbest_ica_pipe)
            methods_list = list()
            methods_list.append('kbest')
            methods_list.append('ica')
            methods_list.append('ica + kbest')


            for i in range(len(pipelines_list)):

                # Define the GridSearchCV object
                # cv = RepeatedKFold(n_splits=5, n_repeats=5)
                cv = KFold(n_splits=5)
                grid = GridSearchCV(estimator=pipelines_list[i], 
                                    param_grid=dim_red_params_list[i], 
                                    cv=cv)
                # Fit the model and find the best hyperparameters
                grid.fit(X_train, y_train)
                # Print the best hyperparameters and score
                print("--------------------------------------------")
                print(f"Method: {methods_list[i]} + CoxPHSurvivalAnalysis")
                # print("--------------------------------------------")
                # print('Best hyperparameters: ', grid.best_params_)
                # print('Best score: ', grid.best_score_)
                # Extract the best model from the grid search
                best_model = grid.best_estimator_
                # Evaluate the best model on the test set
                test_score = best_model.score(X_test, y_test)
                print('Test score: ', test_score)
                
                CoxPHSA_CIndex_list.append(test_score)
                
                
                
                
            # Define the parameter grids for each dimensionality reduction method
            dim_red_params_list = list()
            # pca_params = {'pca__n_components': [5, 10, 15, 20, 25, 30, 35, 40],
            #               'select_features__k': [5, 10, 15, 20, 25, 30]}
            kbest_params = {'select_features__k': [5, 10, 15, 20, 25, 30]}
            dim_red_params_list.append(kbest_params)

            ica_params = {'ica__n_components': [5, 10, 15, 20, 25, 30, 35, 40]}
            dim_red_params_list.append(ica_params)
            ica_kbest_params = {'ica__n_components': [5, 10, 15, 20, 25, 30, 35, 40],
                        'select_features__k': [5, 10, 15, 20, 25, 30]}
            dim_red_params_list.append(ica_kbest_params)
            # Define the pipeline for each dimensionality reduction method
            pipelines_list = list()
            # pca_pipe = Pipeline([('pca', PCA()), 
            #                      ('select_features', SelectKBest()), 
            #                      ('rsf', RandomSurvivalForest())])
            kbest_pipe = Pipeline([('select_features', SelectKBest()), 
                                ('rsf', RandomSurvivalForest())])
            pipelines_list.append(kbest_pipe)
            ica_pipe = Pipeline([('ica', FastICA()), 
                                ('rsf', RandomSurvivalForest())])
            pipelines_list.append(ica_pipe)
            kbest_ica_pipe = Pipeline([('ica', FastICA()), 
                                ('select_features', SelectKBest()), 
                                ('rsf', RandomSurvivalForest())])
            pipelines_list.append(kbest_ica_pipe)
            methods_list = list()
            methods_list.append('kbest')
            methods_list.append('ica')
            methods_list.append('ica + kbest')


            for i in range(len(pipelines_list)):

                # Define the GridSearchCV object
                # cv = RepeatedKFold(n_splits=5, n_repeats=5)
                cv = KFold(n_splits=5)
                grid = GridSearchCV(estimator=pipelines_list[i], 
                                    param_grid=dim_red_params_list[i], 
                                    cv=cv)
                # Fit the model and find the best hyperparameters
                grid.fit(X_train, y_train)
                # Print the best hyperparameters and score
                print("--------------------------------------------")
                print(f"Method: {methods_list[i]} + RandomSurvivalForest")
                # print("--------------------------------------------")
                # print('Best hyperparameters: ', grid.best_params_)
                # print('Best score: ', grid.best_score_)
                # Extract the best model from the grid search
                best_model = grid.best_estimator_
                # Evaluate the best model on the test set
                test_score = best_model.score(X_test, y_test)
                print('Test score: ', test_score)
                
                
                RSF_CIndex_list.append(test_score)

                

            # Define the parameter grids for each dimensionality reduction method
            dim_red_params_list = list()
            # pca_params = {'pca__n_components': [5, 10, 15, 20, 25, 30, 35, 40],
            #               'select_features__k': [5, 10, 15, 20, 25, 30]}
            # dim_red_params_list.append(pca_params)
            kbest_params = {'select_features__k': [5, 10, 15, 20, 25, 30]}
            dim_red_params_list.append(kbest_params)
            ica_params = {'ica__n_components': [5, 10, 15, 20, 25, 30, 35, 40]}
            dim_red_params_list.append(ica_params)
            ica_kbest_params = {'ica__n_components': [5, 10, 15, 20, 25, 30, 35, 40],
                        'select_features__k': [5, 10, 15, 20, 25, 30]}
            dim_red_params_list.append(ica_kbest_params)
            # Define the pipeline for each dimensionality reduction method
            pipelines_list = list()
            # pca_pipe = Pipeline([('pca', PCA()), 
            #                      ('select_features', SelectKBest()), 
            #                      ('fssvm', FastSurvivalSVM())])
            kbest_pipe = Pipeline([('select_features', SelectKBest()), 
                                ('fssvm', FastSurvivalSVM())])
            pipelines_list.append(kbest_pipe)
            ica_pipe = Pipeline([('ica', FastICA()), 
                                ('fssvm', FastSurvivalSVM())])
            pipelines_list.append(ica_pipe)
            kbest_ica_pipe = Pipeline([('ica', FastICA()), 
                                ('select_features', SelectKBest()), 
                                ('fssvm', FastSurvivalSVM())])
            pipelines_list.append(kbest_ica_pipe)
            methods_list = list()
            methods_list.append('kbest')
            methods_list.append('ica')
            methods_list.append('ica + kbest')


            for i in range(len(pipelines_list)):
                # Define the GridSearchCV object
                # cv = RepeatedKFold(n_splits=5, n_repeats=5)
                cv = KFold(n_splits=5)
                grid = GridSearchCV(estimator=pipelines_list[i], 
                                    param_grid=dim_red_params_list[i], 
                                    cv=cv)
                # Fit the model and find the best hyperparameters
                grid.fit(X_train, y_train)
                # Print the best hyperparameters and score
                print("--------------------------------------------")
                print(f"Method: {methods_list[i]} + FastSurvivalSVM")
                # print("--------------------------------------------")
                # print('Best hyperparameters: ', grid.best_params_)
                # print('Best score: ', grid.best_score_)
                # Extract the best model from the grid search
                best_model = grid.best_estimator_
                # Evaluate the best model on the test set
                test_score = best_model.score(X_test, y_test)
                print('Test score: ', test_score)
                
                SVMSA_CIndex_list.append(test_score)

            
            csv_writer.writerow({'Network': NETWORK,
                                'Fold_Number': fold,
                                
                                'KBest_CoxPHSA_CIndex': CoxPHSA_CIndex_list[0],
                                'ICA_CoxPHSA_CIndex': CoxPHSA_CIndex_list[1],
                                'KBest_ICA_CoxPHSA_CIndex': CoxPHSA_CIndex_list[2],
                                
                                'KBest_RSF_CIndex': RSF_CIndex_list[0],
                                'ICA_RSF_CIndex': RSF_CIndex_list[1],
                                'KBest_ICA_RSF_CIndex': RSF_CIndex_list[2],
                                
                                'KBest_SVMSA_CIndex': SVMSA_CIndex_list[0],
                                'ICA_SVMSA_CIndex': SVMSA_CIndex_list[1],
                                'KBest_ICA_SVMSA_CIndex': SVMSA_CIndex_list[2]
                                })
            csv_file.flush()
            fold += 1
csv_file.close()