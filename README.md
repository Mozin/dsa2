### Install

     pip install pyro-ppl lightgbm pandas scikit-learn scipy matplotlib



### Command line usage
    cd dsa2
    source activate py36 
    python source/run_train.py  run_train   --n_sample 100  --model_name lightgbm  --path_config_model source/config_model.py  --path_output /data/output/a01_test/     --path_data /data/input/train/    


    source activate py36 
    python source/run_inference.py  run_predict  --n_sample 1000  --model_name lightgbm  --path_model /data/output/a01_test/   --path_output /data/output/a01_test_pred/     --path_data /data/input/train/




### data/input  : Input data format

    data/input/titanic/raw/  : the raw files
    data/input/titanic/raw2/ : the raw files  split manually


    File names Are FIXED, please create sub-folder  
        data/input/titanic/train/ :   features.zip ,  target.zip, cols_group.json  names are FIXED
             cols_group.json : name of columns per feature category.
             features.zip :    csv file of the inputs
             target.zip :      csv file of the label to predict.

        data/input/titanic/test/ :   features.zip  , used for predictions






### source/  : code source CLI to train/predict.
```
   run_feature_profile.py : CLI Pandas profiling
   run_train.py :           CLI to train any model, any data (model  data agnostic )
   run_inference.py :       CLI to predict with any model, any data (model  data agnostic )


   config_model.py   :  file containing custom parameter for each specific model.
                        Please insert your model config there :
                           titanic_lightgbm


```



### source/models/  : Generic API to access models.
```
   One file python file per model.

   models/model_sklearn.py      :   generic module as class, which wraps any sklearn API type model.
   models/model_bayesian_pyro.py :  generic model as class, which wraps Bayesian regression in Pyro/Pytorch.

   Method of the moddule/class
       .init
       .fit()
       .predict()


```



