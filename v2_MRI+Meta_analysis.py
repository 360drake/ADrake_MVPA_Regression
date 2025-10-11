import numpy as np # type: ignore
import pandas as pd # type: ignore
pd.options.mode.chained_assignment = None
from sklearn.svm import SVR # type: ignore
from sklearn.pipeline import make_pipeline # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold, LeaveOneOut # type: ignore
from sklearn.metrics import mean_squared_error as MSE, mean_absolute_error as MAE, r2_score as R2 # type: ignore
from sklearn.inspection import permutation_importance # type: ignore
from sklearn.linear_model import ElasticNet # type: ignore


from austin_utility import loo_scoring

# Locate your data files in the folder
mri_file = r'/MRI_data.csv'
activity_file = r'/ActivityandMeta_data.csv'

mri_data = pd.read_csv(mri_file)
activity_data = pd.read_csv(activity_file)

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
feat_count_mri = len(mri_data.columns)
print('Number of MRI features: %i' % (feat_count_mri-1)) # subtract 1 for ID_AFIS

# Target labels: 'combviiva', 'combtotMVPA'
meta_all = activity_data[['ID_AFIS','combviiva','combtotMVPA','Gender','Age']]

# one-hot encode gender data
gender = np.where(meta_all['Gender'] == 'M', 1, -1)
meta_all['GENDER'] = gender
meta_all = meta_all.drop('Gender', axis=1)

# log-transform MVPA data
logmvpa = np.log(meta_all['combtotMVPA'])
meta_all['logMVPA'] = logmvpa

# Drop all subjects missing MRI data, meta data, or targets
all_data = pd.merge(mri_data, meta_all, on = 'ID_AFIS', how = 'left')
all_data.dropna(axis = 'index', how = 'any', inplace = True)
all_data = all_data.reset_index(drop=True)
all_data = all_data.drop('ID_AFIS', axis=1)

subj_count = len(all_data)
print('Number of Subjects in MRI dataset: %i' % subj_count)

#Remove categorical data for imputing and scaling
cat_labels = ['GENDER']
cat_data = all_data[cat_labels]
all_data = all_data.drop(cat_labels, axis=1)

#Remove targets
target_data = all_data.loc[:,['combviiva','combtotMVPA', 'logMVPA']]
X = all_data.drop(['combviiva','combtotMVPA', 'logMVPA'], axis=1)

# Rescale all features (except categorical)
X_cols = X.columns
X_mri_scaler = StandardScaler()
X = pd.DataFrame(X_mri_scaler.fit_transform(X), columns=X_cols)

# Add Categorical Data (Gender)
X[cat_labels] = cat_data

# Rescale Targets
# target_cols = target_data.columns
# y_mri_scaler = StandardScaler()
# target_data = pd.DataFrame(y_mri_scaler.fit_transform(target_data), columns=target_cols)

# Define gridsearch parameters
grid_params = [
        {
        'C': [1, 0.1, 10],
        'kernel': ['linear'],
        'epsilon': [0.1, 0.0001, 0.001, 0.01, 1, 10]
        },
        {
        'C': [1, 0.1, 10, 100],
        'kernel': ['rbf'],
        'gamma': [1/100, 1/1000, 1/10, 1],
        'epsilon': [0.1, 0.01, 1, 10]
        }
]

y_AF    = target_data['combviiva']
y_PA    = target_data['combtotMVPA']
y_logPA = target_data['logMVPA']

X_mri   = X.drop(['GENDER', 'Age'], axis=1)
X_meta  = X.loc[:,['GENDER', 'Age']]

AF_scores, AF_params       = loo_scoring(inputs=X, targets=y_AF, param_grid=grid_params, model=SVR())
PA_scores, PA_params       = loo_scoring(inputs=X, targets=y_PA, param_grid=grid_params, model=SVR())
logPA_scores, logPA_params = loo_scoring(inputs=X, targets=y_logPA, param_grid=grid_params, model=SVR())

print('predictions complete')

# Permutation testing

perm_count = 100

AF_allperm    = pd.DataFrame(columns = ['R2', 'MAE', 'MSE'])
PA_allperm    = pd.DataFrame(columns = ['R2', 'MAE', 'MSE'])
logPA_allperm = pd.DataFrame(columns = ['R2', 'MAE', 'MSE'])

# Generate unique seed for each permutation
for j in range(perm_count):
    perm_gen = np.random.default_rng(seed=j)
    perm_idx = perm_gen.permutation(len(X_mri))

    X_mri_perm = X_mri.loc[perm_idx].reset_index(drop=True)
    X_perm = pd.concat((X_mri_perm, X_meta), axis = 1, ignore_index = True)

    AF_perm, _    = loo_scoring(inputs=X_perm, targets=y_AF, param_grid=grid_params, model=SVR())
    PA_perm, _    = loo_scoring(inputs=X_perm, targets=y_PA, param_grid=grid_params, model=SVR())
    logPA_perm, _ = loo_scoring(inputs=X_perm, targets=y_logPA, param_grid=grid_params, model=SVR())

    AF_allperm.loc[j]    = AF_perm
    PA_allperm.loc[j]    = PA_perm
    logPA_allperm.loc[j] = logPA_perm

    print('permutation %i complete' % (j+1))

print('permutations complete')

p_AF    = (np.sum(AF_allperm['MSE'] >= AF_scores[2]) + 1) / (perm_count+1)
p_PA    = (np.sum(PA_allperm['MSE'] >= PA_scores[2]) + 1) / (perm_count+1)
p_logPA = (np.sum(logPA_allperm['MSE'] >= logPA_scores[2]) + 1) / (perm_count+1)

AF_file = "MSE_MRI+Meta+Raw_AF_perm_scores.csv"
AF_allperm.to_csv(AF_file)
PA_file = "MSE_MRI+Meta+Raw_PA_perm_scores.csv"
PA_allperm.to_csv(PA_file)
logPA_file = "MSE_MRI+Meta+Raw_logPA_perm_scores.csv"
logPA_allperm.to_csv(logPA_file)

with open("MSE_MRI+Meta+Raw_Report.txt", "w") as f:
    f.write('Results of Permutation Test on MRI Data \n\n')
    f.write('Permutations: %i \n' % (perm_count))
    f.write('Subjects: %i \n' % (subj_count))
    f.write('MRI Features: %i \n\n' % (feat_count_mri-1))

    f.write('AF Score:\nR2: %.4f' % AF_scores[0])
    f.write('\nMAE: %.4f' % AF_scores[1])
    f.write('\nMSE: %.4f \n\n' % AF_scores[2])

    f.write('PA Score:\nR2: %.4f' % PA_scores[0])
    f.write('\nMAE: %.4f' % PA_scores[1])
    f.write('\nMSE: %.4f \n\n' % PA_scores[2])

    f.write('logPA Score:\nR2: %.4f' % logPA_scores[0])
    f.write('\nMAE: %.4f' % logPA_scores[1])
    f.write('\nMSE: %.4f \n\n' % logPA_scores[2])

    f.write('\np-values of permutation tests (MSE) \n')
    f.write('AF: %.4f \n' % p_AF)
    f.write('PA: %.4f \n' % p_PA)
    f.write('logPA: %.4f \n\n' % p_logPA)

    f.write('Best parameter sets')
    f.write('\nAF: ')
    f.write(str(AF_params))
    f.write('\nPA: ')
    f.write(str(PA_params))
    f.write('\nlogPA: ')
    f.write(str(logPA_params))
