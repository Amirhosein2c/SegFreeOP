
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

NETWORKS = [
            'InceptionResNetV2',
            'InceptionV3',
            'Resnet',
            'VGG16',
            'VGG19',
            'Xception',
            'DenseNet201',
            'EfficientNetB1',
            'EfficientNetB4',
            'EfficientNetB7',
            # 'NASNetLarge',
            # 'ConvNeXtBase'
            ]


METHOD = "MED-ICA"

RESULTS_DIRECTORY = "/datassd/WHOLEIMAGE_MAMIP/Results"
Results_filename = f"SA_Result_Single_{METHOD}.csv"

csv_file = open(os.path.join(RESULTS_DIRECTORY, Results_filename), mode='w')

field_names = ['Network', 
               'Fold_Number', 
               'ICA_CoxPHSA']


with open(os.path.join(RESULTS_DIRECTORY,Results_filename), 'w', newline='') as csv_file:
    csv_writer = csv.DictWriter(csv_file, fieldnames=field_names)
    csv_writer.writeheader()

    for NETWORK in NETWORKS:
        print(NETWORK)

        data = pd.read_csv(f"/datassd/WHOLEIMAGE_MAMIP/PROCESSED_FEATURES/ProcessedFeatures_{METHOD}/Processed_Features_{NETWORK}_MA-MIP_WHOLE-IMAGE.csv")

        data['Relapse'] = data['Relapse'].astype('bool')
        features = data.drop(["Relapse", "RFS", "Patient_ID"], axis=1)
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

        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)

        fold = 1
        kf = KFold(n_splits=5, shuffle=True, random_state=1111)
        
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            print("================")
            print(f"Fold: {fold}")
            print("================")
        
            CoxPHSA_CIndex_list = list()
            
            # Define the parameter grids for each dimensionality reduction method
            dim_red_params_list = list()
            ica_params = {'ica__n_components': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]}
            dim_red_params_list.append(ica_params)
            pipelines_list = list()
            ica_pipe = Pipeline([('ica', FastICA()),
                                ('coxph', CoxPHSurvivalAnalysis())])
            pipelines_list.append(ica_pipe)
            methods_list = list()
            methods_list.append('ica')


            for i in range(len(pipelines_list)):

                # Define the GridSearchCV object
                # cv = RepeatedKFold(n_splits=5, n_repeats=5)
                cv = KFold(n_splits=5)
                grid = GridSearchCV(estimator=pipelines_list[i], 
                                    param_grid=dim_red_params_list[i], 
                                    cv=cv)
                grid.fit(X_train, y_train)
                best_model = grid.best_estimator_
                test_score = best_model.score(X_test, y_test)
                print('Test score: ', test_score)
                
                CoxPHSA_CIndex_list.append(test_score)

            csv_writer.writerow({'Network': NETWORK,
                                'Fold_Number': fold,
                                'ICA_CoxPHSA': CoxPHSA_CIndex_list[0],
                                })
            csv_file.flush()
            fold += 1
csv_file.close()