

import pandas as pd
import time


import json
import re
import os
import numpy as np
import ast
from pprint import pprint
from pandas.io.json import json_normalize
import pickle


# ## Kaggle Google Analytics Google Store
# ### test_v2.csv : visites et transactions de mai à octobre 2018
# nrows = 1000
start_time = time.time()

path= r'C:\Users\utilisateur\Documents\Chef doeuvre\ga-customer-revenue-prediction\train_v2.csv'
data = pd.read_csv(path,
                   dtype={'fullVisitorId': 'str', 'visitId': 'str'},
                   parse_dates = ['date'],
                   nrows = 1000)
 
print('%s seconds' %(time.time() - start_time))


df_device = data['device']
jdevice = df_device.apply(json.loads) 
jdevicelist = pd.DataFrame(jdevice.tolist()) 
jdevicelist
data = pd.concat([data,jdevicelist['browser'], jdevicelist['isMobile'], jdevicelist['deviceCategory'], jdevicelist['operatingSystem']], axis=1) 


jgeo = data['geoNetwork'].apply(json.loads)
jgeolist = pd.DataFrame(jgeo.tolist())
jgeolist
data = pd.concat([data,jgeolist['continent'], jgeolist['subContinent'], jgeolist['country'], jgeolist['region'], jgeolist['city']], axis=1) 


jtot = data['totals'].apply(json.loads)
jtotlist = pd.DataFrame(jtot.tolist())
jtotlist['bounces'] = pd.to_numeric(jtotlist['bounces'], errors='coerce')
jtotlist['newVisits'] = pd.to_numeric(jtotlist['newVisits'], errors='coerce')
jtotlist['timeOnSite'] = pd.to_numeric(jtotlist['timeOnSite'], errors='coerce')
jtotlist['totalTransactionRevenue'] = pd.to_numeric(jtotlist['totalTransactionRevenue'], errors='coerce')
jtotlist['transactions'] = pd.to_numeric(jtotlist['transactions'], errors='coerce')
jtotlist['pageviews'] = pd.to_numeric(jtotlist['pageviews'], errors='coerce')
data = pd.concat([data, jtotlist['visits'].astype('int'), jtotlist['newVisits'], jtotlist['bounces'], jtotlist['hits'].astype('int'), jtotlist['pageviews'], jtotlist['sessionQualityDim'].astype('int'), jtotlist['timeOnSite'], jtotlist['totalTransactionRevenue'], jtotlist['transactions']], axis=1) 



data.drop(['device', 'geoNetwork', 'totals', 'trafficSource', 'customDimensions', 'socialEngagementType'], axis=1, inplace=True)



def session_save(picke_path, objects):
    pickling_on = open(picke_path,"wb")
    try:
        pickle.dump(objects, pickling_on)
        print("session_save success\n")
    except:
        print("session_save failed\n")
    finally:
        pickling_on.close()



session_save(r"GS pickle\dataMhitstodo.pickle", data)

