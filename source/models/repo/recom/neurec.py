# -*- coding: utf-8 -*-
"""NeuRec.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1MHcoIDyMjdcIsC3alBzIh_rCIo0vPDxJ

# Repo
https://github.com/wubinzzu/NeuRec
## tutorial
https://github.com/wubinzzu/NeuRec/blob/master/tutorial.ipynb
"""

from google.colab import drive
drive.mount('/content/drive')

#### Working directory
import os
dirdrive = "/content/drive/My Drive/Colab Notebooks/shared_session/recsys/"
os.chdir( dirdrive)

! pwd



"""## link conda with colab"""

"""

https://serverfault.com/questions/43383/caching-preloading-files-on-linux-into-ram

"""


##### Access to conda environnment package  ################################################
import os, sys
! ls "/content/drive/MyDrive/Colab Notebooks/shared_session/bin/conda"
# from util_conda import conda_link_gdrive, conda_create_env, conda_install_pkg

def conda_link_gdrive():
  prefix = "/usr/local/"
  ### Linking to Conda on Google Drive
  ### conda_commands = r"/content/drive/My\ Drive/Colab\ Notebooks/shared_session/recsys/bin/conda/bin"
  conda_folder = r"/content/drive/My\ Drive/Colab\ Notebooks/shared_session//bin/conda"

  # Sym link conda with colab
  !ln -s  $conda_folder "/usr/local" #create symlink
  !ls /usr/local/conda/  &&  ls /usr/local/conda/envs/

  # give colab permission to use conda 
  !sudo chmod -R 777 /usr/local/conda/bin

  # add conda commands to system path
  os.environ['PATH'] += ":/usr/local/conda/bin"


def conda_create_env(env_name = "py36", python_version='3.6.9'):
    conda_envs = !conda env list
    res = [i for i in conda_envs if env_name in i]
    if (len(res) == 0):
        print('not found ' + env_name + ' env', len(res))
        !conda create -y -q --name $env_name python=$python_version 
    else:
        print('ALready found ' + env_name + ' env', len(res))

def conda_install_pkg(env_name = "py36",  install='', give_access=True):
    conda_envs = !conda env list
    res = [i for i in conda_envs if env_name in i]
    if (len(res) == 0):
        print('not found ' + env_name + ' env', len(res))
    else:
        print('found ' + env_name + ' env', len(res))
        if (give_access):
            !sudo chmod -R 755 /usr/local/conda/envs/$env_name/bin/

        if (len(install) > 0):
            if (install.find('.txt') > -1):
                # install files like requirments.txt
                !source activate $env_name  && pip install -r $install
            else:
                !source activate $env_name && pip install $install

def conda_change_env(env_name = "py36"):
    prefix = "/usr/local/"
    conda_envs = !conda env list
    res = [i for i in conda_envs if env_name in i]
    if (len(res) == 0):
        print('not found ' + env_name + ' env', len(res))
    else:
        print('found ' + env_name + ' env', len(env_name))
        # romoves all enviroment packeges to add new env
        for path in sys.path:
            if 'envs' in path:
                sys.path.remove(path)
        packages_path = f'{prefix}/conda/envs/{env_name}/lib/python3.6/site-packages'
        if packages_path not in sys.path:
            print("packages path not found")
            print("updating packages path....")
            sys.path.append( f'{prefix}/conda/envs/{env_name}/lib/python3.6/site-packages')
            # print(sys.path)
            sys.__plen = len(sys.path) - 1
            new=sys.path[sys.__plen:] 
            del sys.path[sys.__plen:] 
            p=getattr(sys,'__egginsert',0)
            p=0
            sys.path[p:p]=new
            sys.__egginsert = p+len(new)
            print(sys.path)
            print("packages path updated.")
            # ! pip install arrow
        else:
            print("packages path existed")

#### List conda conda envs from the Google Drive conda
conda_link_gdrive()

# Commented out IPython magic to ensure Python compatibility.
# %%bash
# conda env list
#

##### Install package  ####################################
# env_name = 'py36rec'
# conda_install_pkg(env_name, 'tensorflow==1.12.3')


# %%bash
# source activate py36rec
# conda list
# conda install tensorflow==1.12.3 -n py36rec 
# pip install

# Commented out IPython magic to ensure Python compatibility.
# ##### Check package  ####################################
# %%bash
# source activate py36rec
# conda list
# #conda install tensorflow==1.12.3 -n py36rec 
# # pip install 
# 
#

env_name = 'py36rec'
conda_change_env(env_name)
print(sys.path)

##### Caching into RAM the folder

! find /usr/local/conda/envs/py36rec/lib/python3.6/site-packages/ -type f -exec cat {} \;  &>/dev/null









# Commented out IPython magic to ensure Python compatibility.
# """
# 
# find /usr/local/conda/envs/py36rec/lib/python3.6/site-packages/ -type f -exec cat {} \;  &>/dev/null
# 
# 
# command &>/dev/null
# 
# 
# %%bash
# 
# mkdir /usr/local/conda
# sudo mount -t tmpfs -o size=4G tmpfs '/usr/local/conda'
# 
# df
# 
# 
# !rsync -avh "/content/drive/MyDrive/Colab Notebooks/shared_session/bin/conda" /usr/local/
# 
# ! ls  '/content/drive/My Drive/colabs/'
# 
# 
# ! mkdir '/content/drive/My Drive/colabs/test2'
# 
# 
# 
# ! ls  '/content/drive/MyDrive/Colab Notebooks/shared_session/bin/conda/'
# 
# ! ls /dev/shm/
# 
# 
# %%bash
# 
# sudo mount -t tmpfs -o nonempty -o size=4G tmpfs "/content/drive/My Drive/Colab\ Notebooks/shared_session/bin/conda/"
# 
# df
# 
# """
# 
#





import importlib

import tensorflow as tf
# importlib.reload(tf)
print(tf.__version__)





"""## Install repo"""

# !git clone https://github.com/wubinzzu/NeuRec

cd NeuRec/

# cd "/content/drive/My Drive/Colab Notebooks/shared_session/recsys/NeuRec"

# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 1.12.3

!python setup.py build_ext --inplace





"""# Sequential Recommender

## Caser model
"""

# Commented out IPython magic to ensure Python compatibility.
# confogration check repo readme file and toutrial link above. 
model='Caser'

#### Data
splitter = 'loo';    ### How the data is split
dataset_name = 'ml-100k' # 'ml-100k' #'custmized_dataset' #custmized_time_dataset
format='UIRT'

### Train
epochs=2
time = False
# --data.convert.separator=';'




# Training model
# %run main.py --recommender=$model --learning_rate=0.001 --batch_size=128   --data.input.dataset=$dataset_name   --slitter=$splitter  --data.column.format=$format --epochs=$epochs --by_time=$time







"""## SRGNN"""

# Commented out IPython magic to ensure Python compatibility.
# confogration check repo readme file and toutrial link above. 
model='SRGNN'

#### Data
splitter = 'loo';    ### How the data is split
dataset_name = 'ml-100k'  # 'Dataset name in dataset folder
format='UIRT'   #Stands for User Item Rating Time

### Train
epochs=2
time = False
# --data.convert.separator=';'




# Training model
# %run main.py --recommender=$model --learning_rate=0.001 --batch_size=128   --data.input.dataset=$dataset_name   --slitter=$splitter  --data.column.format=$format --epochs=$epochs --by_time=$time









"""##  NPE"""

# Commented out IPython magic to ensure Python compatibility.
# confogration check repo readme file and toutrial link above. 
model='NPE'

#### Data
splitter = 'loo';    ### How the data is split
dataset_name = 'ml-100k'  # 'Dataset name in dataset folder
format='UIRT'   #Stands for User Item Rating Time

### Train
epochs=2
time = False
# --data.convert.separator=';'




# Training model
# %run main.py --recommender=$model --learning_rate=0.001 --batch_size=128   --data.input.dataset=$dataset_name   --slitter=$splitter  --data.column.format=$format --epochs=$epochs --by_time=$time













"""##  TransRec"""

# Commented out IPython magic to ensure Python compatibility.
# confogration check repo readme file and toutrial link above. 
model='TransRec'

#### Data
splitter = 'loo';    ### How the data is split
dataset_name = 'ml-100k'  # 'Dataset name in dataset folder
format='UIRT'   #Stands for User Item Rating Time

### Train
epochs=2
time = False
# --data.convert.separator=';'




# Training model
# %run main.py --recommender=$model --learning_rate=0.001 --batch_size=128   --data.input.dataset=$dataset_name   --slitter=$splitter  --data.column.format=$format --epochs=$epochs --by_time=$time











"""## SASRec"""

# Commented out IPython magic to ensure Python compatibility.
# confogration check repo readme file and toutrial link above. 
model='SASRec'

#### Data
splitter = 'loo';    ### How the data is split
dataset_name = 'ml-100k'  # 'Dataset name in dataset folder
format='UIRT'   #Stands for User Item Rating Time

### Train
epochs=2
time = False
# --data.convert.separator=';'




# Training model
# %run main.py --recommender=$model --learning_rate=0.001 --batch_size=128   --data.input.dataset=$dataset_name   --slitter=$splitter  --data.column.format=$format --epochs=$epochs --by_time=$time











"""## HRM"""

# Commented out IPython magic to ensure Python compatibility.
# confogration check repo readme file and toutrial link above. 
model='HRM'

#### Data
splitter = 'loo';    ### How the data is split
dataset_name = 'ml-100k'  # 'Dataset name in dataset folder
format='UIRT'   #Stands for User Item Rating Time

### Train
epochs=2
time = False
# --data.convert.separator=';'




# Training model
# %run main.py --recommender=$model --learning_rate=0.001 --batch_size=128   --data.input.dataset=$dataset_name   --slitter=$splitter  --data.column.format=$format --epochs=$epochs --by_time=$time











"""## FPMCplus"""

# Commented out IPython magic to ensure Python compatibility.
# confogration check repo readme file and toutrial link above. 
model='FPMCplus'

#### Data
splitter = 'loo';    ### How the data is split
dataset_name = 'ml-100k'  # 'Dataset name in dataset folder
format='UIRT'   #Stands for User Item Rating Time

### Train
epochs=2
time = False
# --data.convert.separator=';'




# Training model
# %run main.py --recommender=$model --learning_rate=0.001 --batch_size=128   --data.input.dataset=$dataset_name   --slitter=$splitter  --data.column.format=$format --epochs=$epochs --by_time=$time

















"""# General Recommender

## MLP
"""

# Commented out IPython magic to ensure Python compatibility.
# confogration check repo readme file and toutrial link above. 
model='MLP'

#### Data
splitter = 'loo';    ### How the data is split
dataset_name = 'ml-100k' # 'ml-100k' #'custmized_dataset' #custmized_time_dataset
format='UIRT'

### Train
epochs=2
time = False
# --data.convert.separator=';'




# Training model
# %run main.py --recommender=$model --learning_rate=0.001 --batch_size=128   --data.input.dataset=$dataset_name   --slitter=$splitter  --data.column.format=$format --epochs=$epochs --by_time=$time







"""## FISM"""

# Commented out IPython magic to ensure Python compatibility.
# confogration check repo readme file and toutrial link above. 
model='FISM'

#### Data
splitter = 'loo';    ### How the data is split
dataset_name = 'ml-100k'  # 'Dataset name in dataset folder
format='UIRT'   #Stands for User Item Rating Time

### Train
epochs=2
time = False
# --data.convert.separator=';'




# Training model
# %run main.py --recommender=$model --learning_rate=0.001 --batch_size=128   --data.input.dataset=$dataset_name   --slitter=$splitter  --data.column.format=$format --epochs=$epochs --by_time=$time









"""##  NAIS"""

# Commented out IPython magic to ensure Python compatibility.
# confogration check repo readme file and toutrial link above. 
model='NAIS'

#### Data
splitter = 'loo';    ### How the data is split
dataset_name = 'ml-100k'  # 'Dataset name in dataset folder
format='UIRT'   #Stands for User Item Rating Time

### Train
epochs=2
time = False
# --data.convert.separator=';'




# Training model
# %run main.py --recommender=$model --learning_rate=0.001 --batch_size=128   --data.input.dataset=$dataset_name   --slitter=$splitter  --data.column.format=$format --epochs=$epochs --by_time=$time













"""##  DeepICF"""

# Commented out IPython magic to ensure Python compatibility.
# confogration check repo readme file and toutrial link above. 
model='DeepICF'

#### Data
splitter = 'loo';    ### How the data is split
dataset_name = 'ml-100k'  # 'Dataset name in dataset folder
format='UIRT'   #Stands for User Item Rating Time

### Train
epochs=2
time = False
# --data.convert.separator=';'




# Training model
# %run main.py --recommender=$model --learning_rate=0.001 --batch_size=128   --data.input.dataset=$dataset_name   --slitter=$splitter  --data.column.format=$format --epochs=$epochs --by_time=$time











"""## MF"""

# Commented out IPython magic to ensure Python compatibility.
# confogration check repo readme file and toutrial link above. 
model='MF'

#### Data
splitter = 'loo';    ### How the data is split
dataset_name = 'ml-100k'  # 'Dataset name in dataset folder
format='UIRT'   #Stands for User Item Rating Time

### Train
epochs=2
time = False
# --data.convert.separator=';'




# Training model
# %run main.py --recommender=$model --learning_rate=0.001 --batch_size=128   --data.input.dataset=$dataset_name   --slitter=$splitter  --data.column.format=$format --epochs=$epochs --by_time=$time











"""## LightGCN"""

# Commented out IPython magic to ensure Python compatibility.
# confogration check repo readme file and toutrial link above. 
model='LightGCN'

#### Data
splitter = 'loo';    ### How the data is split
dataset_name = 'ml-100k'  # 'Dataset name in dataset folder
format='UIRT'   #Stands for User Item Rating Time

### Train
epochs=2
time = False
# --data.convert.separator=';'




# Training model
# %run main.py --recommender=$model --learning_rate=0.001 --batch_size=128   --data.input.dataset=$dataset_name   --slitter=$splitter  --data.column.format=$format --epochs=$epochs --by_time=$time











"""# Social Recommender

## SBPR
"""

# Commented out IPython magic to ensure Python compatibility.
# confogration check repo readme file and toutrial link above. 
model='SBPR'

#### Data
splitter = 'loo';    ### How the data is split
dataset_name = 'ml-100k' # 'ml-100k' #'custmized_dataset' #custmized_time_dataset
format='UIRT'

### Train
epochs=2
time = False
# --data.convert.separator=';'




# Training model
# %run main.py --recommender=$model --learning_rate=0.001 --batch_size=128   --data.input.dataset=$dataset_name   --slitter=$splitter  --data.column.format=$format --epochs=$epochs --by_time=$time







"""# prediction"""

user0 = dataset.test_matrix[0]
print(user0)

output = model.predict(0)
output

"""# Create fake Data"""

import pandas as pd

def read_dataset(name ='custmized_dataset' ):
    '''
    Reading train and test csv file with extention (train and test)

    name ==> is dataset name included in dataset folder
    return 
        Two dataframes one for train data and other for test
        
    '''
    test_data  = pd.read_csv('dataset/'+name+'.test')
    train_data = pd.read_csv('dataset/'+name+'.train')
    for data in [train_data, test_data]:
        for name in data.columns:
            if name == 'Unnamed: 0':
                del data['Unnamed: 0']
            if name == 'Unnamed: 0.1':
                del data['Unnamed: 0.1']
    return train_data, test_data
train_data, test_data = read_dataset('custmized_time_dataset')

train_data.head(5)

# Generate Fak dates
import pandas as pd
last30 = pd.datetime.now().replace(microsecond=0) - pd.Timedelta('30H')
print (last30)

dates = pd.date_range(last30, periods = 30 * 60 * 60, freq='S')
print (dates)

# Fake data with dates 
import numpy as np
N = len(train_data)
train_data['0\t0'] = train_data['0\t0'] + '\t' + str(np.random.choice(dates, size=N))

N = len(test_data)
test_data['0\t43886'] = test_data['0\t43886'] + '\t' + str(np.random.choice(dates, size=N))

# train_data['user'] = train_data.apply(lambda x: x['0\t0'].split('\t')[0], axis=1)
# train_data['item'] = train_data.apply(lambda x: x['0\t0'].split('\t')[1], axis=1)

# del train_data['0\t0']

# test_data['user'] = test_data.apply(lambda x: x['0\t43886'].split('\t')[0], axis=1)
# test_data['item'] = test_data.apply(lambda x: x['0\t43886'].split('\t')[1], axis=1)

# del test_data['0\t43886']

# t_data = pd.DataFrame()
# for col in ["user", "item", "rating", "time"]:
#     t_data[col] = test_data[col]
# test_data = t_data

train_data.head(5)

def save_dataset(train_data,test_data, name='custmized_dataset'):
    train_data.to_csv('dataset/'+name+'.train')
    test_data.to_csv('dataset/'+name+'.test')

save_dataset(train_data,test_data,'custmized_time_dataset')

# del test_data['Unnamed: 0']
# del test_data['Unnamed: 0.1'] 
# test_data

# test_data['0\t0\t43886'] = test_data.apply(lambda x: str(x['time']) + '\t' + x['0\t43886'], axis = 1)

# train_data['0\t0\t0'] = train_data.apply(lambda x: str(x['time']) + '\t' + x['0\t0'], axis = 1)

test_data.columns

# del test_data['0\t43886']
# del test_data['time']



"""# Data processing / Check"""










