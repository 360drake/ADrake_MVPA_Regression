import numpy as np # type: ignore
import pandas as pd # type: ignore
from sklearn.svm import SVR # type: ignore
from sklearn.pipeline import make_pipeline # type: ignore
from sklearn.preprocessing import StandardScaler, MinMaxScaler # type: ignore
from sklearn.model_selection import * # type: ignore
from sklearn.metrics import mean_squared_error, make_scorer # type: ignore
from sklearn.impute import SimpleImputer # type: ignore


# Locate your data files in the folder
meg_file = r'/MEG_data.csv'
activity_file = r'/ActivityandMeta_data.csv'

meg_data = pd.read_csv(meg_file)
activity_data = pd.read_csv(activity_file)

## Define Target and Kernel
kernel = 'linear'
target = 'combviiva'

#Drop MEG data not used in models
drop_cols_meg = [

]

# Target labels: 'combviiva', 'combtotMVPA'
targets_all = activity_data[['ID_AFIS','combviiva','combtotMVPA']]
# Add Gender and Age Metadata
meta_data = activity_data[['ID_AFIS','Gender','Age']]
meta_all = pd.merge(targets_all, meta_data, on = 'ID_AFIS', how = 'left')
meta_all['GENDER'] = np.where(meta_all['Gender'] == 'M', 1, -1)
meta_all = meta_all.drop('Gender', axis=1)

# Drop all subjects missing MEG data or targets
meg_data = pd.merge(meg_data, meta_all, on = 'ID_AFIS', how = 'left')
# meg_data = pd.merge(meg_data, targets_all, on = 'ID_AFIS', how = 'left')
meg_data.dropna(axis = 'index', how = 'any', inplace = True)
meg_data = meg_data.reset_index(drop=True)

print('Number of Subjects in MEG dataset: %i' % len(meg_data))
meg_data.to_csv('MEG+Meta_participants.csv')

y = meg_data[target].to_numpy()
#log transform MVPA data
# if 'MVPA' in target:
#   y = np.log(y)
#   print('MVPA data has been log-transformed')

target_labels = targets_all.columns.values
X = meg_data.drop(target_labels, axis=1)

#Remove categorical data for imputing and scaling
cat_labels = ['GENDER']
cat_data = X[cat_labels]
X = X.drop(cat_labels, axis=1)

# Impute missing data with the mean: this will become 0 after scaling
X_cols = X.columns
X = X.replace(to_replace = 0, value= np.nan)
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=X_cols)

# Rescale all features (except categorical)
X_scaler = StandardScaler()
X = pd.DataFrame(X_scaler.fit_transform(X), columns=X_cols)

# Add Categorical Data (Gender)
X[cat_labels] = cat_data

# Rescale targets
# y_scaler = StandardScaler()
# y = y_scaler.fit_transform(y.reshape(-1, 1)).flatten()

# Keep only alpha frequency bands
filter_col = [col for col in X if 'alpha' in col]
X = X[filter_col]

feat_count_meg = len(X.columns)
print('Number of MEG features: %i' % (feat_count_meg))

if kernel == 'linear':
  grid_params = {
    'C': [1, 0.001, 0.01, 0.1, 10, 100],
    'epsilon': [1, 0.001, 0.01, 0.1, 10, 100, 1000]
    }

if kernel == 'rbf':
  grid_params = {
    'C': [1, 0.001, 0.01, 0.1, 10, 100],
    'gamma': [1/100, 1/10000, 1/1000, 1/10, 1, 10],
    'epsilon': [1, 0.001, 0.01, 0.1, 10, 100, 1000]
  }

if kernel == 'poly':
  grid_params = {
    'C': [1, 0.001, 0.01, 0.1, 10, 100],
    'gamma': [1/100, 1/10000, 1/1000, 1/10, 1, 10],
    'degree':[2, 3, 4],
    'coef0': [0, -10, -1, 1, 10],
    'epsilon': [1, 0.001, 0.01, 0.1, 10, 100, 1000]
  }

if kernel == 'sigmoid':
  grid_params = {
    'C': [1, 0.001, 0.01, 0.1, 10, 100],
    'gamma': [1/100, 1/10000, 1/1000, 1/10, 1, 10],
    'coef0': [0, -10, -1, 1, 10],
    'epsilon': [1, 0.001, 0.01, 0.1, 10, 100, 1000]
  }
    
scoring = {"R-squared": "r2", "MAE": "neg_mean_absolute_error", "RMSE": "neg_root_mean_squared_error"}

grid_meg = GridSearchCV(SVR(), param_grid = grid_params, verbose = 0, cv = 5, refit = "RMSE", scoring = scoring)
grid_meg.fit(X, y)
results = pd.DataFrame(grid_meg.cv_results_)


results_file = "SVR_MEG+Meta_cv_results_Rawlog_" + target + "_" + kernel +".csv"
results.to_csv(results_file)

meg_params = grid_meg.best_params_
print("best params by RMSE: ", meg_params)
print("best score (RMSE): ", 0-grid_meg.best_score_)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
svr_meg = SVR(kernel=kernel, **meg_params)
svr_meg.fit(X_train, y_train)
train_pred = svr_meg.predict(X_train)
test_pred = svr_meg.predict(X_test)
print('train score (RMSE): ', mean_squared_error(y_train, train_pred, squared= False))
print("test score (RMSE): ", mean_squared_error(y_test, test_pred, squared= False))
print('GridSearchCV complete')
