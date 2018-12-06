# random_forest_tg
Random forest models for predicting the coefficient of friction and adhesion for systems of two contacting functionalized monolayers.

### Installation

Use of this model requires several Python packages to be installed, as well
as data obtained from molecular dynamics screening. It is recommended to
use the Anaconda package manager to create a new environment for using these
models.

`>> conda create --name myconda python=3.5`

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
pip install .
```

#### Install the Signac analysis package
`conda install signac -c glotzer`

#### Install other required packages

```
conda install numpy pandas scipy scikit-learn
```
