import lightgbm as lgb
import pandas as pd
import glob
import matplotlib.pyplot as plt

print('Loading data...')
path = r'./small/' # use your path
all_files = glob.glob(path + "/*.csv")

li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)

df_train = pd.concat(li, axis=0, ignore_index=True)

y_train = df_train["target"]
X_train = df_train.drop("target", axis=1)

# create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, y_train)

# specify your configurations as a dict
params = {
    'boosting_type': 'rf',
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_leaves': 2000000,
    'max_depth':1,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0,
    'num_iterations':10
}

print('Starting training...')

#gbm = lgb.train(params, lgb_train, num_boost_round=20,)

model = lgb.LGBMClassifier(**params)
model.fit(X_train, y_train)

print('Plotting a tree...')  # one tree use categorical feature to split
ax = lgb.plot_tree(model, tree_index=2, figsize=(15, 15), show_info=['split_gain'])
plt.show()

