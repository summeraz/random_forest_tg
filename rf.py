from collections import OrderedDict
import json

from atools_ml.dataio import df_setup
from atools_ml.descriptors import rdkit_descriptors
from atools_ml.prep import dimensionality_reduction, train_test_split
import numpy as np
import pandas as pd
import signac
import scipy
from sklearn import ensemble, linear_model, metrics, model_selection

"""
INPUT


SMILES FOR TOP AND BOTTOM TERMINAL GROUPS

`SMILES1` corresponds to the SMILES for the terminal group of the bottom
monolayer, terminated by a hydrogen. For example, a hydroxyl group this
would simply be 'O'.

`SMILES2` corresponds to the SMILES for the terminal group of the top
monolayer, terminated by a hydrogen. For example, a methyl group this
would simply be 'C'.


RANDOM SEED

This is the random number generator seed used for test/train splits for
generating the random forest model. In the manuscript, the following
seeds were used:

    Model       Seed
    -----       ----
    1           43
    2           0
    3           1
    4           2
    5           3

As the manuscript reports on the results of Model 1, a seed of 43 is used
by default.

`path_to_data` is the relative path to the MD screening data. If the
installation instructions were followed exactly, these two directories
('terminal_group_screening' and 'terminal_groups_mixed') should be located
one above the current working directory. If these were placed elsewhere,
the `path_to_data` string should be updated accordingly.
"""

SMILES1 = 'C(=O)N'
SMILES2 = 'O'
random_seed = 43

path_to_data = ".."

"""

The code below will use the seed to generate the random forest model and
predict coefficient of friction and adhesion for the SMILES combination
provided.

Features are obtained using RDKit and have manually been classified into
describing either shape, size, charge distribution, or complexity. The
corresponding clusters are provided in the `feature-cluster.json` file.

"""

def predict(smiles1, smiles2, random_seed, path_to_data):

    ch3_SMILES1 = 'C{}'.format(SMILES1)
    ch3_SMILES2 = 'C{}'.format(SMILES2)

    with open('feature-clusters.json', 'r') as f:
        clusters = json.load(f)
    shape_features = clusters['shape']

    """
    Because Gasteiger charges are assigned, molecules aren't guarenteed to be
    charge neutral. However, the total positive and negative charge are not
    very helpful in predictive modeling, so those features are removed here.

    The "min" and "mean" indicate features describing the minimum and mean
    values between the two terminal groups respectively.
    """
    to_drop = ['pc+-mean', 'pc+-min', 'pc--mean', 'pc--min']

    # Descriptors for H-terminated SMILES
    desc_h_tg1 = rdkit_descriptors(SMILES1)
    desc_h_tg2 = rdkit_descriptors(SMILES2)

    # Descriptors for CH3-terminated SMILES
    desc_ch3_tg1 = rdkit_descriptors(ch3_SMILES1, include_h_bond=True,
                                     ch3_smiles=ch3_SMILES1)
    desc_ch3_tg2 = rdkit_descriptors(ch3_SMILES2, include_h_bond=True,
                                     ch3_smiles=ch3_SMILES2)

    desc_h_df = pd.DataFrame([desc_h_tg1, desc_h_tg2])
    desc_ch3_df = pd.DataFrame([desc_ch3_tg1, desc_ch3_tg2])

    desc_df = []
    for i, df in enumerate([desc_h_df, desc_ch3_df]):
        if i == 1:
            hbond_tb = max(df['hdonors'][0], df['hacceptors'][1]) \
                       if all((df['hdonors'][0], df['hacceptors'][1])) \
                       else 0
            hbond_bt = max(df['hdonors'][1], df['hacceptors'][0]) \
                       if all((df['hdonors'][1], df['hacceptors'][0])) \
                       else 0
            hbonds = hbond_tb + hbond_bt
            df.drop(['hdonors', 'hacceptors'], 'columns', inplace=True)
        else:
            hbonds = 0
        means = df.mean()
        mins = df.min()
        means = means.rename({label: '{}-mean'.format(label)
                              for label in means.index})
        mins = mins.rename({label: '{}-min'.format(label)
                            for label in mins.index})
        desc_tmp = pd.concat([means, mins])
        desc_tmp['hbonds'] = hbonds
        desc_tmp.drop(labels=to_drop, inplace=True)
        desc_df.append(desc_tmp)

    df_h_predict = desc_df[0]
    df_ch3_predict = desc_df[1]
    df_h_predict = pd.concat([
        df_h_predict.filter(like=feature) for feature in shape_features], axis=0)
    df_ch3_predict.drop(labels=df_h_predict.keys(), inplace=True)

    df_h_predict_mean = df_h_predict.filter(like='-mean')
    df_h_predict_min = df_h_predict.filter(like='-min')
    df_ch3_predict_mean = df_ch3_predict.filter(like='-mean')
    df_ch3_predict_min = df_ch3_predict.filter(like='-min')

    df_predict = pd.concat([df_h_predict_mean, df_h_predict_min,
                            df_ch3_predict_mean, df_ch3_predict_min,
                            df_ch3_predict[['hbonds']]])

    """
    Load data from MD screening
    """
    root_dir_same = ('{}/terminal_group_screening'.format(path_to_data))
    proj_same = signac.get_project(root=root_dir_same)

    root_dir_mixed = ('{}/terminal_groups_mixed'.format(path_to_data))
    proj_mixed = signac.get_project(root=root_dir_mixed)

    # Define chemistry identifiers and target variable
    identifiers = ['terminal_group_1', 'terminal_group_2']
    targets= ['COF', 'intercept']

    df_h = df_setup([proj_same, proj_mixed], mean=True,
                    descriptors_filename='descriptors-h.json',
                    smiles_only=True)
    df_ch3 = df_setup([proj_same, proj_mixed], mean=True,
                      descriptors_filename='descriptors-ch3.json',
                      smiles_only=True)

    to_drop = ['pc+-mean', 'pc+-diff', 'pc+-min', 'pc+-max',
               'pc--mean', 'pc--diff', 'pc--min', 'pc--max']

    df_h.drop(labels=to_drop, axis=1, inplace=True)
    df_ch3.drop(labels=to_drop, axis=1, inplace=True)

    shape_features = clusters['shape']
    df_h = pd.concat([
                df_h.filter(like=feature) for feature in shape_features],
                axis=1)
    df_ch3.drop(labels=df_h.columns, axis=1, inplace=True)

    df_h_mean = df_h.filter(like='-mean')
    df_h_min = df_h.filter(like='-min')
    df_ch3_mean = df_ch3.filter(like='-mean')
    df_ch3_min = df_ch3.filter(like='-min')

    df = pd.concat([df_h_mean, df_h_min, df_ch3_mean, df_ch3_min,
                    df_ch3[identifiers + targets + ['hbonds']]], axis=1)

    # Reduce the number of features by running them through various filters
    features = list(df.drop(identifiers + targets, axis=1))
    df_red = dimensionality_reduction(df, features, filter_missing=True,
                                      filter_var=True, filter_corr=True,
                                      missing_threshold=0.4,
                                      var_threshold=0.02,
                                      corr_threshold=0.9)
    df = df_red
    features = list(df.drop(identifiers + targets, axis=1))
    df_predict = df_predict.filter(features)

    for target in ['COF', 'intercept']:
        X_train, X_test, y_train, y_test = model_selection.train_test_split(
                df[features], df[target], test_size=0.2,
                random_state=random_seed)

        regr = ensemble.RandomForestRegressor(n_estimators=1000,
                                              oob_score=True,
                                              random_state=random_seed)
        regr.fit(X_train, y_train)

        predicted = regr.predict(np.array(df_predict).reshape(1, -1))
        print('{} (predicted): {:.4f}'.format(target, predicted[0]))

predict(SMILES1, SMILES2, random_seed, path_to_data)
