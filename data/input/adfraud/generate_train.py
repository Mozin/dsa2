import pandas as pd, numpy as np, random, copy, os, sys
random.seed(100)




########################################################################################################
def generate_train(df, col=None, pars=None):
   df = copy.deepcopy(df)

   df = df.drop(columns=["attributed_time"])

   ## Let's see on which hour the click was happend
   df["click_time"]   = pd.to_datetime(df["click_time"])

   df["hour"]         = df["click_time"].dt.hour.astype("uint8")
   df["minute"]       = df["click_time"].dt.minute.astype("uint8")
   # df["second"]       = df["click_time"].dt.second.astype("uint8")
   df["day"]          = df["click_time"].dt.day.astype("uint8")
   df["day_of_week"]  = df["click_time"].dt.dayofweek.astype("uint8")


   ##Let's divide the day in four section ,See in which section click has happend ")
   day_section = 0
   for start_time, end_time in zip([0, 6, 12, 18], [6, 12, 18, 24]):
              df.loc[(df['hour'] >= start_time) & (df['hour'] < end_time), 'day_section'] = day_section
              day_section += 1

   print( "clicks count features")
   df["n_ip_clicks"]  = df[['ip', 'channel']].groupby(by=["ip"])[["channel"]].transform("count").astype("uint8")



   ##Computing the number of clicks associated with a given app per hour...')
   df["n_app_clicks"] = df[['app', 'day', 'hour', 'channel']].groupby(by=['app', 'day', 'hour'])[['channel']].transform("count").astype("uint8")

   ##Computing the number of channels associated with a given IP address within each hour...')
   df["n_channels"]   = df[['ip', 'day', 'hour', 'channel']].groupby(by=['ip', 'day', 'hour'])[['channel']].transform("count").astype("uint8")


   ##Let's divide the day in four section ,See in which section click has happend ")
   day_section              = 0
   for start_time, end_time in zip([0, 6, 12, 18], [6, 12, 18, 24]):
     df.loc[(df['hour'] >= start_time) & (df['hour'] < end_time), 'day_section'] = day_section
     day_section             += 1

   ##Computing the number of channels associated with ')
   df['ip_app_count']       = df[['ip', 'app', 'channel']].groupby(by=['ip', 'app'])[['channel']].transform("count").astype("uint8")
   print( "new clicks count features")
   df["n_ip_clicks"]        = df[['ip', 'channel']].groupby(by=["ip"])[["channel"]].transform("count").astype("uint8")

   ##Computing the number of channels associated with ')
   df["ip_app_os_count"]    = df[['ip', 'app', 'os', 'channel']].groupby(by=['ip', 'app', 'os'])[['channel']].transform("count").astype("uint8")

   df['n_ip_os_day_hh']     = df[['ip', 'os', 'day', 'hour', 'channel']].groupby(by=['ip', 'os', 'day', 'hour'])[['channel']].transform("count").astype("uint8")
   ##Computing the number of clicks associated with a given app per hour...')
   df["n_app_clicks"]       = df[['app', 'day', 'hour', 'channel']].groupby(by=['app', 'day', 'hour'])[['channel']].transform("count").astype("uint8")

   df['n_ip_app_day_hh']    = df[['ip', 'app', 'day', 'hour', 'channel']].groupby(by=['ip', 'app', 'day', 'hour'])[['channel']].transform("count").astype("uint8")
   ##Computing the number of channels associated with a given IP address within each hour...')
   df["n_channels"]         = df[['ip', 'day', 'hour', 'channel']].groupby(by=['ip', 'day', 'hour'])[['channel']].transform("count").astype("uint8")

   df['n_ip_app_os_day_hh'] = df[['ip', 'app', 'os', 'day', 'hour', 'channel']].groupby(by=['ip', 'app', 'os', 'day', 'hour'])[['channel']].transform("count").astype("uint8")
   ##Computing the number of channels associated with ')
   df['ip_app_count']       = df[['ip', 'app', 'channel']].groupby(by=['ip', 'app'])[['channel']].transform("count").astype("uint8")

   df['n_ip_app_dev_os']    = df[['ip', 'app', 'device', 'os', 'channel']].groupby(by=['ip', 'app', 'device', 'os'])[['channel']].transform("count").astype("uint8")
   ##Computing the number of channels associated with ')
   df["ip_app_os_count"]    = df[['ip', 'app', 'os', 'channel']].groupby(by=['ip', 'app', 'os'])[['channel']].transform("count").astype("uint8")

   df['n_ip_dev_os']        = df[['ip', 'device', 'os', 'channel']].groupby(by=['ip', 'device', 'os'])[['channel']].transform("count").astype("uint8")


   GROUPBY_AGGREGATIONS = [
        # Count, for ip-day-hour
        {'groupby': ['ip','day','hour'], 'select': 'channel', 'agg': 'count'},
        # Count, for ip-app
        {'groupby': ['ip', 'app'], 'select': 'channel', 'agg': 'count'},
        # Count, for ip-app-os
        {'groupby': ['ip', 'app', 'os'], 'select': 'channel', 'agg': 'count'},
        # Count, for ip-app-day-hour
        {'groupby': ['ip','app','day','hour'], 'select': 'channel', 'agg': 'count'},
        # Mean hour, for ip-app-channel
        {'groupby': ['ip','app','channel'], 'select': 'hour', 'agg': 'mean'},

        ################### V2 - GroupBy Features #
        # Average clicks on app by distinct users; is it an app they return to?
        {'groupby': ['app'],
         'select': 'ip',
         'agg': lambda x: float(len(x)) / len(x.unique()),
         'agg_name': 'AvgViewPerDistinct'
        },
        # How popular is the app or channel?
        {'groupby': ['app'], 'select': 'channel', 'agg': 'count'},
        {'groupby': ['channel'], 'select': 'app', 'agg': 'count'},

        #################### V3 - GroupBy Features                                              #
        # https://www.kaggle.com/bk0000/non-blending-lightgbm-model-lb-0-977 #
        {'groupby': ['ip'],  'select': 'channel', 'agg': 'nunique'},
        {'groupby': ['ip'],  'select': 'app', 'agg': 'nunique'},
        {'groupby': ['ip',   'day'], 'select': 'hour', 'agg': 'nunique'},
        {'groupby': ['ip',   'app'], 'select': 'os', 'agg': 'nunique'},
        {'groupby': ['ip'],  'select': 'device', 'agg': 'nunique'},
        {'groupby': ['app'], 'select': 'channel', 'agg': 'nunique'},
        {'groupby': ['ip',  'device', 'os'], 'select': 'app', 'agg': 'nunique'},
        {'groupby': ['ip',  'device','os'], 'select': 'app', 'agg': 'cumcount'},
        {'groupby': ['ip'], 'select': 'app', 'agg': 'cumcount'},
        {'groupby': ['ip'], 'select': 'os', 'agg': 'cumcount'}
  ]

   # Apply all the groupby transformations
   for spec in GROUPBY_AGGREGATIONS:
      # Name of the aggregation we're applying
      agg_name = spec['agg_name'] if 'agg_name' in spec else spec['agg']

      # Name of new feature
      new_feature = '{}_{}_{}'.format('_'.join(spec['groupby']), agg_name, spec['select'])

      print("Grouping by {}, and aggregating {} with {}".format( spec['groupby'], spec['select'], agg_name ))

      # Unique list of features to select
      all_features = list(set(spec['groupby'] + [spec['select']]))

      # Perform the groupby
      gp = df[all_features]. \
          groupby(spec['groupby'])[spec['select']]. \
          agg(spec['agg']). \
          reset_index(). \
          rename(index=str, columns={spec['select']: new_feature})

      # Merge back to df
      if 'cumcount' == spec['agg']:
          df[new_feature] = gp[0].values
      else:
          df = df.merge(gp, on=spec['groupby'], how='left')

   del gp

   df['n_ip_os_day_hh']     = df[['ip', 'os', 'day', 'hour', 'channel']].groupby(by=['ip', 'os', 'day', 'hour'])[['channel']].transform("count").astype("uint8")
   df['n_ip_app_day_hh']    = df[['ip', 'app', 'day', 'hour', 'channel']].groupby(by=['ip', 'app', 'day', 'hour'])[['channel']].transform("count").astype("uint8")
   df['n_ip_app_os_day_hh'] = df[['ip', 'app', 'os', 'day', 'hour', 'channel']].groupby(by=['ip', 'app', 'os', 'day', 'hour'])[['channel']].transform("count").astype("uint8")
   df['n_ip_app_dev_os']    = df[['ip', 'app', 'device', 'os', 'channel']].groupby(by=['ip', 'app', 'device', 'os'])[['channel']].transform("count").astype("uint8")
   df['n_ip_dev_os']        = df[['ip', 'device', 'os', 'channel']].groupby(by=['ip', 'device', 'os'])[['channel']].transform("count").astype("uint8")


   df.drop(columns=["click_time"], axis=1, inplace=True)





   ###############################################################################
   #### coly  genereate  #########################################################
   df['is_attributed'].value_counts(normalize=True)  ### ?
   df["is_attributed"]=df["is_attributed"].astype("uint8")

   dfnew    = df
   col_pars = {}
   return dfnew, col_pars





########################################################################################################
def generate_train2(df, col=None, pars=None):
   """
     By IP Apress
     channel                                  1011
os                                        544
hour                                      472
app                                       468
ip_app_os_device_day_click_time_next_1     320
app_channel_os_mean_is_attributed         189
ip_app_mean_is_attributed                 124
ip_app_os_device_day_click_time_next_2     120
ip_os_device_count_click_id               113
ip_var_hour                                94
ip_day_hour_count_click_id                 91
ip_mean_is_attributed                      74
ip_count_click_id                          73
ip_app_os_device_day_click_time_lag1       67
app_mean_is_attributed                     67
ip_nunique_os_device                       65
ip_nunique_app                             63
ip_nunique_os                              51
ip_nunique_app_channel                     49
ip_os_device_mean_is_attributed            46
device                                     41
app_channel_os_count_click_id              37
ip_hour_mean_is_attributed                 21


   """
   df = copy.deepcopy(df)

   df = df.drop(columns=["attributed_time"])
   print(df.columns)

   ## Let's see on which hour the click was happend
   df["click_time"]   = pd.to_datetime(df["click_time"])
   df["hour"]         = df["click_time"].dt.hour.astype("uint8")
   #df["minute"]       = df["click_time"].dt.minute.astype("uint8")
   df["day"]          = df["click_time"].dt.day.astype("uint8")
   df["dayweek"]  = df["click_time"].dt.dayofweek.astype("uint8")


   #### By IP Address
   df1 = df.groupby(['ip', 'app', 'os', 'device', 'channel']).agg( {  'is_attributed' : 'max' }).reset_index()     #   'hour' : {'min', 'max'} } )
   # print(df1.head(3).T )

   GROUPBY_AGGREGATIONS = [
        {'groupby': ['ip', 'app'],             'select': 'channel', 'agg': 'count'},
        {'groupby': ['ip', 'app', 'os'],       'select': 'channel', 'agg': 'count'},


        ##### Time Related
        {'groupby': ['ip'],    'select': 'hour', 'agg': 'max'},
        {'groupby': ['ip'],    'select': 'hour', 'agg': 'min'},

        {'groupby': ['ip'],    'select': 'dayweek', 'agg': 'max'},
        {'groupby': ['ip'],    'select': 'dayweek', 'agg': 'min'},

        {'groupby': ['ip'], 'select': 'hour', 'agg': lambda x: float(len(x)) / len(x.unique()), 'agg_name': 'AvgViewPerDistinct'},
        {'groupby': ['ip'], 'select': 'dayweek', 'agg': lambda x: float(len(x)) / len(x.unique()), 'agg_name': 'AvgViewPerDistinct'},


        {'groupby': ['ip','channel'],    'select': 'hour', 'agg': 'max'},
        {'groupby': ['ip','channel'],    'select': 'hour', 'agg': 'min'},

        {'groupby': ['ip','channel'],    'select': 'dayweek', 'agg': 'max'},
        {'groupby': ['ip','channel'],    'select': 'dayweek', 'agg': 'min'},


        {'groupby': ['ip'],          'select': 'channel', 'agg': 'nunique'},
        {'groupby': ['ip'],          'select': 'app',     'agg': 'nunique'},
        {'groupby': ['ip'],          'select': 'device',  'agg': 'nunique'},

        {'groupby': ['ip'], 'select': 'app',     'agg': lambda x: float(len(x)) / len(x.unique()), 'agg_name': 'AvgViewPerDistinct'},
        {'groupby': ['ip'], 'select': 'channel', 'agg': lambda x: float(len(x)) / len(x.unique()), 'agg_name': 'AvgViewPerDistinct'},


        {'groupby': ['ip',   'app'], 'select': 'os', 'agg': 'nunique'},
        {'groupby': ['ip',  'device', 'os'], 'select': 'app', 'agg': 'nunique'},


        ################### V2 - GroupBy Features #
        # Average clicks on app by distinct users; is it an app they return to?
        {'groupby': ['app'], 'select': 'ip', 'agg': lambda x: float(len(x)) / len(x.unique()), 'agg_name': 'AvgViewPerDistinct'},
        {'groupby': ['app'],         'select': 'channel', 'agg': 'count'},
        {'groupby': ['app'],         'select': 'channel', 'agg': 'nunique'},


        {'groupby': ['channel'],     'select': 'app',     'agg': 'count'},


        # {'groupby': ['ip',  'device','os'],  'select': 'app', 'agg': 'cumcount'},
        # {'groupby': ['ip'], 'select': 'app', 'agg': 'cumcount'},
        # {'groupby': ['ip'], 'select': 'os',  'agg': 'cumcount'}
  ]

   # Apply all the groupby transformations
   for spec in GROUPBY_AGGREGATIONS:
      agg_name    = spec['agg_name'] if 'agg_name' in spec else spec['agg']
      new_feature = '{}_{}_{}'.format('_'.join(spec['groupby']), agg_name, spec['select'])

      print("Grouping by {}, and aggregating {} with {}".format( spec['groupby'], spec['select'], agg_name ))

      all_features = list(set(spec['groupby'] + [spec['select']]))

      # Perform the groupby
      gp = df[all_features].groupby(spec['groupby'])[spec['select']]. \
          agg(spec['agg']).reset_index(). \
          rename(index=str, columns={spec['select']: new_feature})
      
      if 'cumcount' == spec['agg']:
          df1[new_feature] = gp[0].values
      else:
          df1 = df1.merge(gp, on=spec['groupby'], how='left')
          # print( df1.head(3).T )
   del gp



   dfnew    = df1
   col_pars = {}
   return dfnew, col_pars








def generatePastClickFeatures(df):
    # Time between past clicks
    df2 = pd.DataFrame([], index= df.index)  
    pastClickAggregateFeatures = [
        {'groupBy': ['ip', 'channel']},
        {'groupBy': ['ip', 'os']}
    ]
    for spec in pastClickAggregateFeatures:
        feature_name = '{}-past-click'.format('_'.join(spec['groupBy']))   
        df[feature_name] = df[spec['groupBy'] + ['click_time']].groupby(['ip']).click_time.transform(lambda x: x.diff().shift(1)).dt.seconds
    return df2





#################################################################################
##### Train sample data #########################################################
coly = "is_attributed"


def a10m():
  ##### Test Data  #############################################
  df = pd.read_csv("raw/raw_10m.zip")
  df, col_pars = generate_train(df)
  df_X = df.drop(coly, axis=1)
  df_y = df[[coly]]


  path = "train_10m/"
  os.makedirs(path, exist_ok=True)
  df_X.to_parquet( f"{path}/features.parquet")
  df_y.to_parquet( f"{path}/target.parquet")



def a100k():
  df   = pd.read_csv("raw/train_100k.csv")

  df, col_pars = generate_train2(df)
  df_X         = df.drop(coly, axis=1)
  df_y         = df[[coly]]
  print( np.sum(df_y))
  print( df_X.shape )
  print(df_X.columns)

  path = "train_100k/"
  os.makedirs(path, exist_ok=True)
  df_X.to_parquet( f"{path}features.parquet")
  df_y.to_parquet( f"{path}/target.parquet")

  df_X.to_csv( f"{path}features.csv")
  df_y.to_csv( f"{path}/target.csv")




def a200m():
  ##### Train data  ############################################
  df           = pd.read_csv("raw/train_200m.zip")
  df, col_pars = generate_train(df)
  df_X         = df.drop(coly, axis=1)
  df_y         = df[[coly]]


  path = "train_200m/"
  os.makedirs(path, exist_ok=True)
  df_X.to_parquet( f"{path}/features.parquet")
  df_y.to_parquet( f"{path}/target.parquet")




###########################################################################################################
###########################################################################################################
if __name__ == "__main__":

    import fire
    fire.Fire()
    








###### Load data  #############################################################
"""
dtypes = {'ip': np.uint32, 'app': np.uint16, 'device': np.uint8, 'os': np.uint8, 'channel': np.uint8, 'is_attributed': np.bool}
df     = pd.read_csv('raw/train_100k.csv', sep=',', dtype=dtypes, parse_dates=['click_time', 'attributed_time'])


df2 = generateAggregateFeatures(df)
df3 = generatePastClickFeatures(df)


df  = df.join(  df2, how='left' )
df  = df.join(  df3, how='left' )


"""



def generateAggregateFeatures(df):
    ### New Agg features
    df2 = pd.DataFrame([], index= df.index)
    aggregateFeatures = [
        # How popular is the app in channel?
        {'name': 'app-popularity', 'groupBy': ['app'], 'select': 'channel', 'agg': 'count'},
        # How popular is the channel in app?
        {'name': 'channel-popularity', 'groupBy': ['channel'], 'select': 'app', 'agg': 'count'},
        # Average clicks on app by distinct users; is it an app they return to?
        {'name': 'avg-clicks-on-app', 'groupBy': ['app'], 'select': 'ip', 'agg': lambda x: float(len(x)) / len(x.unique())}
    ]

    for spec in aggregateFeatures:
        print("Generating aggregate feature {} group by {}, and aggregating {} with {}".format(spec['name'], spec['groupBy'], spec['select'], spec['agg']))
        gp = df[spec['groupBy'] + [spec['select']]] \
            .groupby(by=spec['groupBy'])[spec['select']] \
            .agg(spec['agg']) \
            .reset_index() \
            .rename(index=str, columns={spec['select']: spec['name']})
        df2 = df2.merge(gp, on=spec['groupBy'], how='left')
        del gp
        gc.collect()

    return df2