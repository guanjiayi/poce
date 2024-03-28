## Install
### Create a conda environment and install
```
cd poce/
conda env create --file conda-recipe.yaml
conda activate poce
pip install -e .
```
### Load the environment 
Please download the compressed environment package "safety-gymnasium-poce.zip" from the Google Drive address provided in the TXT file and extract it into the "envs/" directory.
```
cd poce/envs/
```

### Install the environment 
```
cd poce/envs/safety-gymnasium-poce/
pip install -e .
```

### Load the dataset
Please download the dataset from the Google Drive address provided in the TXT file and place it in the current directory.
```
cd poce/dataset/
```
# Training and testing the model
```
cd poce/
python main/train_policy.py
```

