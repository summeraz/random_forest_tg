## Random forest models for predicting the coefficient of friction and adhesion for systems of two contacting functionalized monolayers.

### Installation

Use of this model requires several Python packages to be installed, as well
as data obtained from molecular dynamics screening. Most of the required
packages are located in the `req.txt` file. It is recommended to
use the Anaconda package manager to create a new environment, as the
packages can be pulled from this file directly.

The recommended installation instructions are as follows:

#### Clone this repository

```
>> git clone https://github.com/summeraz/random_forest_tg.git
```

#### Create a new Anaconda environment

`>> conda create --name myconda --file random_forest_tg/req.txt -c conda-forge python=3.5`

#### Activate the environment

`>> source activate myconda`

#### Download data from MD screening

```
git clone https://github.com/summeraz/terminal_group_screening.git
git clone https://github.com/summeraz/terminal_groups_mixed.git
```

#### Install atools-ml package

```
git clone https://github.com/summeraz/atools_ml.git
cd atools_ml
pip install .
cd ..
```

### Using the models
The random forest models can be regenerated in a few seconds.
Thus, rather than providing these models in a form already generated (such
as a serialized form like pickle), the script herein re-creates the models
on the spot.
The script `rf.py` is used to regenerate the models and generate
predictions for user-specified terminal group chemistries. These can be
changed by opening the file and altering the `"SMILES1"` and `"SMILES2"`
variables.
Further instructions can be found inside the `rf.py` file.
