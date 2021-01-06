from sklearn import preprocessing, metrics
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold, GroupKFold, GridSearchCV, train_test_split, TimeSeriesSplit
from datetime import datetime
import copy
import os


##### import all Feature engineering functions
from util_feat_m5 import *


def features_to_category(df):
	nan_features = ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']
	for feature in nan_features:
		df[feature].fillna('unknown', inplace = True)

	categorical_cols = ['dept_id', 'cat_id', 'store_id', 'state_id', 'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']
	for feature in categorical_cols:
		encoder     = preprocessing.LabelEncoder()
		df[feature] = encoder.fit_transform(df[feature].astype(str))

	return df


def update_meta_csv(featnames, filename, cat_cols):
	meta_csv = pd.DataFrame(columns = ['featname', 'filename', 'feattype'])
	if os.path.exists('meta_features.csv'):
		meta_csv = pd.read_csv('meta_features.csv')
	append_data_dict = {'featname' : [], 'filename' : [], 'feattype' : []}
	for feat in featnames:
		if feat not in meta_csv['featname'].unique():
			append_data_dict['filename'].append(filename)
			append_data_dict['featname'].append(feat)
			feat_type = "numeric" if feat not in cat_cols else "categorical"
			append_data_dict['feattype'].append(feat_type)
		else:
			meta_csv.loc[meta_csv['featname'] == feat, 'filename'] = filename
	append_df = pd.DataFrame.from_dict(append_data_dict)
	meta_csv = meta_csv.append(append_df)
	meta_csv.to_csv('meta_features.csv', index = False)


def get_cat_num_features_from_meta_csv():
	meta_csv = pd.read_csv('meta_features.csv')
	num_feats = [ x for x in meta_csv[meta_csv["feattype"] == "numeric"]['featname'].tolist()  if x not in ["demand", "date"]]
	cat_feats = [ x for x in meta_csv[meta_csv["feattype"] == "categorical"]['featname'].tolist() if x not in ["item_id"]]
	return cat_feats, num_feats


def get_file_feat_from_meta_csv(selected_cols):
	meta_csv = pd.read_csv('meta_features.csv')
	file_feat_mapping = {k:['date', 'item_id'] for k in meta_csv['filename'].unique().tolist()}
	for selected_col in selected_cols:
		selected_col_meta_df = meta_csv[meta_csv["featname"] == selected_col]
		file_feat_mapping[selected_col_meta_df['filename'].tolist()[0]].append(selected_col)
	return {k:list(set(v)) for k,v in file_feat_mapping.items()}


def features_generate_file(dir_in, dir_out, my_fun_features, features_group_name) :

    # from util_feat_m5  import lag_featrues
    # features_generate_file(dir_in, dir_out, lag_featrues)

	merged_df = pd.read_parquet(dir_in + "/raw_merged.df.parquet")
	dfnew, cat_cols= my_fun_features(merged_df)
	dfnew.to_parquet(f'{dir_out}/{features_group_name}.parquet')
	# num_cols = list(set(dfnew._get_numeric_data().columns))
	update_meta_csv(dfnew.columns, f'{features_group_name}.parquet', cat_cols)


def feature_merge_df(df_list, cols_join):
	dfall = None
	for dfi in df_list :
		cols_joini = [ t for t in cols_join if t in dfi.columns ]
		dfall      = dfall.join(dfi.set_index(cols_joini), on = cols_joini, how="left") if dfall is not None else dfi
	return dfall


def raw_merged_df(input_path = "", fname='raw_merged.df.parquet', max_rows=10):
	df_sales_train            = pd.read_csv("data/sales_train_eval.csv")
	df_calendar               = pd.read_csv("data/calendar.csv")
	df_sales_val              = pd.read_csv("data/sales_train_val.csv")
	df_sell_price             = pd.read_csv("data/sell_prices.csv")
	# df_submi                  = pd.read_csv("data/sample_submi.csv")

	df_sales_val_melt         = pd.melt(df_sales_val[0:max_rows], id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], var_name = 'day', value_name = 'demand')
	# val_rows                  = [row for row in df_submi['id'] if 'val' in row]
	# eval_rows                 = [row for row in df_submi['id'] if 'eval' in row]
	# df_submi_val              = df_submi[df_submi['id'].isin(val_rows)][0:max_rows]
	# df_submi_eval             = df_submi[df_submi['id'].isin(eval_rows)][0:max_rows]

	# df_submi_val              = df_submi_val.merge(df_product, how = 'left', on = 'id')
	# df_submi_eval             = df_submi_eval.merge(df_product, how = 'left', on = 'id')

	# df_submi_val              = pd.melt(df_submi_val, id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], var_name = 'day', value_name = 'demand')
	# df_submi_eval             = pd.melt(df_submi_eval, id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], var_name = 'day', value_name = 'demand')

	# df_sales_val_melt['part'] = 'train'
	# df_submi_val['part']      = 'test1'
	# df_submi_eval['part']     = 'test2'

	# merged_df = pd.concat([df_sales_val_melt, df_submi_val, df_submi_eval], axis = 0)
	merged_df = df_sales_val_melt
	df_calendar.drop(['weekday', 'wday', 'month', 'year'], inplace = True, axis = 1)
	merged_df = pd.merge(merged_df, df_calendar, how = 'left', left_on = ['day'], right_on = ['d'])
	merged_df = merged_df.merge(df_sell_price, on = ['store_id', 'item_id', 'wm_yr_wk'], how = 'left')

	merged_df = features_to_category(merged_df)
	# merged_df = add_time_features(merged_df)

	merged_df.to_parquet(fname)
	# return merged_df




def features_get_cols(mode = "random"):
	# categorical_cols = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2' ]
	# numerical_cols = ['snap_TX',  'sell_price', 'week', 'snap_CA', 'month', 'snap_WI', 'dayofweek', 'year']

	categorical_cols, numerical_cols = get_cat_num_features_from_meta_csv()

	cols_cat = []
	cols_num = []

	if mode == "random":
		cols_cat = [categorical_cols[i] for i in np.random.choice(len(categorical_cols), 3, replace = False)]
		cols_num = [numerical_cols[i] for i in np.random.choice(len(numerical_cols), 5, replace = False) ]

	if mode == "all":
		cols_cat = categorical_cols
		cols_num = numerical_cols

	if mode == "smartway":
		cols_cat = categorical_cols
		cols_num = numerical_cols
		# TODO: Need to update

	return cols_cat, cols_num



def load_data(path, selected_cols):
	selected_cols = ['demand', 'date'] + selected_cols

	file_col_mapping = get_file_feat_from_meta_csv(selected_cols)
	# merged_df = pd.DataFrame()
	# for file_name,file_cols in file_col_mapping.items():
	# 	file_df = pd.read_parquet(path + file_name, columns = file_cols)
	# 	merged_df = pd.concat([merged_df, file_df], axis = 0)

	# for file_name,file_cols in file_col_mapping.items():
	# 	print(file_name)
	# 	print(file_cols)
	# 	pd.read_parquet(f'{path}/{file_name}', columns = file_cols)

	feature_dfs = [pd.read_parquet(f'{path}/{file_name}', columns = file_cols) for file_name,file_cols in file_col_mapping.items() if len(file_cols) > 0]
	merged_df = feature_merge_df(feature_dfs, ['date', 'item_id'])

	merged_df = merged_df.sort_values('date')
	merged_df.drop(['item_id'], inplace = True, axis = 1)
	return merged_df


def X_transform(df, selected_cols):
	X_df = df[  selected_cols  ]    #.drop(['demand', 'date'], axis =1)
	return X_df


def Y_transform(df, selected_col):
	Y_df= df[selected_col]
	return Y_df


def run_eval(max_rows = None, n_experiments = 3):
	model_params = {'num_leaves': 555,
          'min_child_weight' : 0.034,
          'feature_fraction' : 0.379,
          'bagging_fraction' : 0.418,
          'min_data_in_leaf' : 106,
          'objective'        : 'regression',
          'max_depth'        : -1,
          'learning_rate'    : 0.005,
          "boosting_type"    : "gbdt",
          "bagging_seed"     : 11,
          "metric"           : 'rmse',
          "verbosity"        : -1,
          'reg_alpha'        : 0.3899,
          'reg_lambda'       : 0.648,
          'random_state'     : 222,
         }

	dict_metrics = {'run_id' : [], 'cols' : [], 'metric_name': [], 'model_params': [], 'metrics_val' : []}

	for ii in range(n_experiments):
		cols_cat, cols_num   = features_get_cols()
		df          		 = load_data('data/output', cols_cat + cols_num)

		# df_output            = load_data('data/output', cols_cat + cols_num, 'test1')
		print('Features loaded')

		X 	               = X_transform(df, cols_cat + cols_num)
		y            	   = Y_transform(df, 'demand')
		# X_output 		   = X_transform(df_output, cols_cat + cols_num)
		# Y_test             = Y_transform(df_test, 'demand')

		# preparing split
		X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.33, random_state=42)
		print(Y_test)

		# Y_test = np.zeros(X_test.shape[0])

		dtrain                     = lgb.Dataset(X_train, label=Y_train)
		dtest                      = lgb.Dataset(X_test, label=Y_test)
		clf                        = lgb.train(model_params, dtrain, 2500,
			                         valid_sets = [dtrain, dtest],
			                         early_stopping_rounds = 50, verbose_eval=100)
		Y_test_pred          	   = clf.predict(X_test,num_iteration=clf.best_iteration)
		val_score                  = np.sqrt(metrics.mean_squared_error(Y_test_pred, Y_test))
		#print(f'val rmse score is {val_score}')

		# Y_test += clf.predict(X_test, num_iteration=clf.best_iteration)/n_fold

		dict_metrics['run_id'].append(datetime.now())
		dict_metrics['cols'].append(";".join(X_train.columns.tolist()))
		dict_metrics['model_params'].append(model_params)
		dict_metrics['metric_name'].append('rmse')
		dict_metrics['metrics_val'].append(val_score)

	df_metrics = pd.DataFrame.from_dict(dict_metrics)
	print("        DF metrics          ")
	print(df_metrics)
	df_metrics.to_csv("df_metrics.csv")





if __name__ == "__main__":
	# create_and_save_features(100, ["set1", "set2"])
	#run_eval(100)

	# To be run once
	raw_merged_df(input_path = "data/input/m5")

	# Generating features
	features_generate_file(".", "data/output/m5", basic_time_features, "basic_time")
	features_generate_file(".", "data/output/m5", features_rolling, "rolling")
	features_generate_file(".", "data/output/m5", features_lag, "lag")
	features_generate_file(".", "data/output/m5", features_tsfresh, "tsfresh")
	features_generate_file(".", "data/output/m5", identity_features, "identity")

	run_eval()




"""


import util_feat_m5



# df_meta=   col_name, col_type, file_path



def features_generate_file(dir_in, dir_out, my_fun_features) :

    # from util_feat_m5  import lag_featrues
    # features_generate_file(dir_in, dir_out, lag_featrues)

	train_df = pd.read_csv( dir_in  + "/sales_train_val.csv.zip")
	calendar_df = pd.read_csv(dir_in  + "/calendar.csv")
	price_df = pd.read_csv(dir_in  + "/sell_prices.csv")


	dfnew = my_fun_features(train_df, calendar_df, price_df) :

    dfnew.to_parquet( dir_out +"/mfeaturesXXXX.parquet")



 def features_get_cols(mode="random") :
    cols_cat0 = [  "feat1", "fewat2" ]

    if mode == "random" :
		### Random selection
	    cols_cat = cols_cat0[ np.random.c = hoice( 5, len(cols_cat)  ) ]
	    cols_num = cols_num0[ np.random.c = hoice( 5, len(cols_num)  ) ]
        return cols_cat, col_num

    if mode == "all" :
      pass

    if mode == "smartway" :
       pass



def run_eval(model, pars={} ) :

    data_pars = {}
    model_pars = {}

    for ii in range(n_experiments) :
		cols_cat, cols_num = features_get_cols()

        df     = load_data(path, cols_cat + cols_num, "train")
        dftest = load_data(path, cols_cat + cols_num, 'test')

		X_train = X_transform( df, cols_num, cols_cat, pars) # select sri
		y_train  = y_transform(df, coly)

		X_test = X_transform( dftest, cols_num, cols_cat, pars) # select variables
		y_test  = y_transform(dftest, coly)



		lgbm = lgb.LGBMRegressor()
        lgbm.fit( X_train, y_train)

		# prediction + metrics
		y_test_pred = lgbm.predict(X_test)
        metric_val = metrics_calc(y_test, y_test_pred)


	    ### Store in metrics :

	    # run_id, feat_name, feat_name_long, feat_type, model_params, metric_name, metric_val
	    # 3,roling_demand,Mean of the variable estimates,lag_features,params = {"objective" : "poisson","metric" :"rmse","force_row_wise" : True,"learning_rate" : 0.075,
	"sub_row" : 0.75,"bagging_freq" : 1,"lambda_l2" : 0.1,"metric": ["rmse"],'verbosity': 1,'num_iterations' : 250,
	},rmse,1.16548

	    df_metrics['run_id'] = time()
	    df_metrics['cols'].append(  ",".join( cols_num + cols_cat ))
	    df_metrics['metrics_val'].append(metric_val)





def test_old():
	from util_feat_m5  import lag_featrues
	train_df = pd.read_csv("sales_train_val.csv")
	calendar_df = pd.read_csv("calendar.csv")
	price_df = pd.read_csv("sell_prices.csv")
	sample = pd.read_csv("sample_submi.csv")
	calendar_df["date_dt"] = pd.to_datetime(calendar_df["date"])
	train  = train_df.copy()
	price = price_df.copy()
	calendar = calendar_df.copy()
	Train_data = train.iloc[:,:-56]
	Val_data = train.iloc[:,:-28]

	X_train = lag_featrues(Train_data).iloc[:,5:] # select variables
	y_train = train.iloc[:,-56]
	X_test = lag_featrues(Val_data).iloc[:,5:]
	y_test = train.iloc[:,-28]

	# Create instance
	lgbm = lgb.LGBMRegressor()

	# Training and score
	learning_rate = [0.15, 0.2, 0.25]
	max_depth = [15, 20, 25]

	param_grid = {'learning_rate': learning_rate, 'max_depth': max_depth}

	# Fitting
	cv_lgbm = GridSearchCV(lgbm, param_grid, cv=10, n_jobs =1)
	cv_lgbm.fit(X_train, y_train)

	print("Best params:{}".format(cv_lgbm.best_params_))

	# best params
	best_lg = cv_lgbm.best_estimator_

	# prediction
	y_train_pred_lg = best_lg.predict(X_train)
	y_test_pred_lg = best_lg.predict(X_test)

	print("MSE train:{}".format(mean_squared_error(y_train, y_train_pred_lg)))
	print("MSE test;{}".format(mean_squared_error(y_test, y_test_pred_lg)))

	print("R2 score train:{}".format(r2_score(y_train, y_train_pred_lg)))
	print("R2 score test:{}".format(r2_score(y_test, y_test_pred_lg)))
	#Predict using only variables with an importance of 1 or higher.
	importance = best_lg.feature_importances_

	indices = np.argsort(importance)[::-1]

	# print importance
	importance_df = pd.DataFrame({})
	columns = []
	importance_ = []
	for f in range(X_train.shape[1]):
	    print("%2d) %-*s %.2f" %(f+1, 30, X_train.columns[indices[f]], importance[indices[f]]))
	    col = X_train.columns[indices[f]]
	    imp = importance[indices[f]]
	    columns.append(col)
	    importance_.append(imp)
	importance_df["col_name"] = columns
	importance_df["importance"] = importance_
	importance = best_lg.feature_importances_

	indices = np.argsort(importance)[::-1]

	# importance columns (>0)
	imp_col = importance_df[importance_df["importance"]>0]["col_name"].values

	# Train test split, select by imp_col

	X_train = lag_featrues(Train_data).iloc[:,5:][imp_col] # select variables
	y_train = train.iloc[:,-56]
	X_test = lag_featrues(Val_data).iloc[:,5:][imp_col]
	y_test = train.iloc[:,-28]

	# Create instance
	lgbm = lgb.LGBMRegressor()

	# Training and score
	learning_rate = [0.15, 0.2, 0.25]
	max_depth = [15, 20, 25]

	param_grid = {'learning_rate': learning_rate, 'max_depth': max_depth}

	# Fitting
	cv_lgbm = GridSearchCV(lgbm, param_grid, cv=10, n_jobs =1)
	cv_lgbm.fit(X_train, y_train)

	print("Best params:{}".format(cv_lgbm.best_params_))

	# best params
	best_lg = cv_lgbm.best_estimator_

	# prediction
	y_train_pred_lg = best_lg.predict(X_train)
	y_test_pred_lg = best_lg.predict(X_test)

	print("MSE train:{}".format(mean_squared_error(y_train, y_train_pred_lg)))
	print("MSE test;{}".format(mean_squared_error(y_test, y_test_pred_lg)))

	print("R2 score train:{}".format(r2_score(y_train, y_train_pred_lg)))
	print("R2 score test:{}".format(r2_score(y_test, y_test_pred_lg)))
	run_id=list(range(300))
	df_metrics=pd.DataFrame(run_id,columns=['run_id'])
	df_metrics['feat_name'] = pd.Series(X_train.columns[0:300], index=dataframe.index)
	df_metrics['feat_type']=df_metrics.feat_name
	df_metrics.replace({'feat_type': r'^lag_.*'}, {'feat_type': 'lag'}, regex=True,inplace=True)
	df_metrics.replace({'feat_type': r'^rolling.*'}, {'feat_type': 'rolling'}, regex=True,inplace=True)
	df_metrics['parameter'] = pd.Series(best_lg, index=dataframe.index)
	df_metrics['metric_name'] ="MSE"
	df_metrics['metric_val'] = pd.Series(pred_mse[:300], index=dataframe.index)
	df_metrics.to_csv("run_eval.csv")
"""
