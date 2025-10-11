import numpy as np # type: ignore
import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore


file_template= r'/SVR_cv_results_'

# Score to be plotted
score = 'MAE'

# List all kernels to be compared
kernels = [
    'linear',
    'rbf',
    'poly',
    'sigmoid'
]

# One target at a time
targets = [
    'combtotMVPA',
    'combviiva'
]

for j, target in enumerate(targets):
    for i, kernel in enumerate(kernels):
        file_path = file_template + target + '_' + kernel + '.csv'
        df_ = pd.read_csv(file_path)

        rank_label = 'rank_test_' + score
        df_.sort_values(by=[rank_label])
        best_model = df_.iloc[0]

        test_labels = ['split0_test_'+score, 'split1_test_'+score, 'split2_test_'+score, 'split3_test_'+score, 'split4_test_'+score]
        test_scores = best_model.loc[test_labels]

        if 'M' in score:
            test_scores = 0 - test_scores

        err = test_scores.std()/np.sqrt(5)

        colors = ['b','r','g','m']
        plt.bar(x=j+(0.2*i)-0.4, height=test_scores.mean(), yerr=err, width=0.2, label=kernel, color=colors[i])

plt.xticks(ticks = np.arange(2), labels=targets)
plt.ylim((-1,1))
plt.grid(visible=True)
plt.title(score + ' from MRI Data')
plt.legend(kernels)
plt.savefig(score + '_MRI.png')
