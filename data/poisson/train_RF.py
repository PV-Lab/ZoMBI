# Author's Note:
# Data was last pulled from Materials Project and used to train the RF on 30 June 2022.
# The dataset may have changed/updated since last pull.

from mp_api import MPRester
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib

props = ['nelements', 'composition', 'density', 'energy_per_atom', 'efermi', 'energy_above_hull',
         'band_gap', 'homogeneous_poisson'] # pull the target properties for the X and Y datasets
api_key = '<YOUR API KEY HERE>' # *ENTER YOUR API FROM MATERIALS PROJECT*
# API keys: https://docs.materialsproject.org/downloading-data/using-the-api/querying-data

with MPRester(api_key) as mpr:
    docs = mpr.summary.search(fields = props)

# build full dataset pulled from materials project
chem_props = pd.DataFrame()
nelements = []
composition = []
energy_per_atom = []
density = []
efermi = []
band_gap = []
energy_above_hull = []
homogeneous_poisson = []
for n in range(len(docs)):
    nelements.append(docs[n].nelements)
    composition.append(docs[n].composition)
    energy_per_atom.append(docs[n].energy_per_atom)
    density.append(docs[n].density)
    efermi.append(docs[n].efermi)
    band_gap.append(docs[n].band_gap)
    energy_above_hull.append(docs[n].energy_above_hull)
    homogeneous_poisson.append(docs[n].homogeneous_poisson)
for p in props:
    chem_props[p] = globals()[p]
chem_props = chem_props[~np.isnan(chem_props.homogeneous_poisson)]  # remove nan
chem_props = chem_props[~np.isnan(chem_props.efermi)] # remove nan

# build training dataset
X_train = chem_props.iloc[:,2:-1]
dim = X_train.shape[1]
# normalize each column
X_train_norm = []
for d in range(dim):
    dcol = np.abs(X_train.iloc[:,d])
    X_train_norm.append(np.array(dcol / np.max(dcol))) # normalize
X_train_norm = np.array(X_train_norm).T
y_train = chem_props.iloc[:,-1]
regr_RF = RandomForestRegressor(n_estimators = 500)
regr_RF.fit(X_train_norm, y_train)

# save trained model as pickled file
filename = 'poisson_RF_trained.pkl'
joblib.dump(regr_RF, filename, compress = 3) # compress the file. compress = 3 is optimal