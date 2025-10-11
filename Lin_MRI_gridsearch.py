import numpy as np # type: ignore
import pandas as pd # type: ignore
from sklearn.svm import SVR # type: ignore
from sklearn.pipeline import make_pipeline # type: ignore
from sklearn.preprocessing import StandardScaler, MinMaxScaler # type: ignore
from sklearn.model_selection import * # type: ignore
from sklearn.metrics import mean_squared_error, make_scorer # type: ignore
from sklearn.impute import SimpleImputer # type: ignore
from sklearn.linear_model import ElasticNet # type: ignore

# Locate your data files in the folder
mri_file = r'/MRI_data.csv'
activity_file = r'/ActivityandMeta_data.csv'

mri_data = pd.read_csv(mri_file)
activity_data = pd.read_csv(activity_file)

# Define Target and Kernel
target = 'combtotMVPA'

#Drop MRI data not used in models
drop_cols_mri = [
    'lhCortexVol',
    'rhCortexVol',
    'CortexVol',
    'SubCortGrayVol',
    'TotalGrayVol',
    'MaskVol',
    'EstimatedTotalIntraCranialVol',
    'rh_MeanThickness_thickness',
    'lh_MeanThickness_thickness'
]

mri_data = mri_data.drop(drop_cols_mri, axis=1)

# Target labels: 'combviiva', 'combtotMVPA'
targets_all = activity_data[['ID_AFIS','combviiva','combtotMVPA']]
# Add Gender and Age Metadata
meta_data = activity_data[['ID_AFIS','Gender','Age']]
meta_all = pd.merge(meta_data, targets_all, on = 'ID_AFIS', how = 'left')
meta_all['GENDER'] = np.where(meta_all['Gender'] == 'M', 1, -1)
meta_all = meta_all.drop('Gender', axis=1)

# Drop all subjects missing MRI data or targets
# mri_data = pd.merge(mri_data, targets_all, on = 'ID_AFIS', how = 'left')
mri_data = pd.merge(mri_data, meta_all, on = 'ID_AFIS', how = 'left')
mri_data.dropna(axis = 'index', how = 'any', inplace = True)
mri_data = mri_data.reset_index(drop=True)

print('Number of Subjects in MRI dataset: %i' % len(mri_data))
mri_data.to_csv('MRI+Meta_participants.csv')

y_mri = mri_data[target].to_numpy()

#log transform MVPA data
if 'MVPA' in target:
 y_mri = np.log(y_mri)
 print('MVPA data has been log-transformed')

target_labels = targets_all.columns.values
X_mri = mri_data.drop(target_labels, axis=1)

#Remove categorical data for imputing and scaling
cat_labels = ['GENDER']
cat_data = X_mri[cat_labels]
X_mri = X_mri.drop(cat_labels, axis=1)

# Rescale all features (except categorical)
X_mri_cols = X_mri.columns
X_mri_scaler = StandardScaler()
X_mri = pd.DataFrame(X_mri_scaler.fit_transform(X_mri), columns=X_mri_cols)

# Add Categorical Data (Gender)
X_mri[cat_labels] = cat_data

# Rescale targets
# y_scaler = StandardScaler()
# y_mri = y_scaler.fit_transform(y_mri.reshape(-1, 1)).flatten()

feat_count_mri = len(X_mri.columns)
print('Number of MRI features: %i' % (feat_count_mri-1))

grid_params = {
'alpha': [1, 0.01, 0.1, 10, 100],
'l1_ratio': [0.5, 0, 0.25, 0.75, 1],
'max_iter': [10000],
}

scoring = {"R-squared": "r2", "MAE": "neg_mean_absolute_error", "RMSE": "neg_root_mean_squared_error"}

grid_meg = GridSearchCV(ElasticNet(), param_grid = grid_params, verbose = 0, cv = 5, refit = "RMSE", scoring = scoring)
grid_meg.fit(X_mri, y_mri)
results = pd.DataFrame(grid_meg.cv_results_)


results_file = "Lasso_MRI+Meta_cv_results_Rawlog_" + target +".csv"
results.to_csv(results_file)

meg_params = grid_meg.best_params_
print("best params by RMSE: ", meg_params)
print("best score (RMSE): ", 0-grid_meg.best_score_)

X_train, X_test, y_train, y_test = train_test_split(X_mri, y_mri, test_size=0.2, random_state=0)
svr_meg = ElasticNet(**meg_params)
svr_meg.fit(X_train, y_train)
train_pred = svr_meg.predict(X_train)
test_pred = svr_meg.predict(X_test)
print('train score (RMSE): ', mean_squared_error(y_train, train_pred, squared= False))
print("test score (RMSE): ", mean_squared_error(y_test, test_pred, squared= False))
print('GridSearchCV complete')
