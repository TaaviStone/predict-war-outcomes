import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import missingno as msno

battles = pd.read_csv('../battlesFiles/battles.csv')
terrain = pd.read_csv('../battlesFiles/terrain.csv')
weather = pd.read_csv('../battlesFiles/weather.csv')

# Let's first look at only WW2 battles
war_list = ['WORLD WAR II (ITALY 1943-1944)', 'WORLD WAR II (ITALY 1944)', 'WORLD WAR II (EUROPEAN THEATER)',
            'WORLD WAR II', 'WORLD WAR II (EASTERN FRONT)', 'WORLD WAR II (OKINAWA)']

# There are only 4 features that we're interested in, so we are gonna merge these tables and have only the features left.
# -weather -terrain -fortification -element of surprise achieved or not

mergedDf = pd.merge(battles, terrain, on="isqno")
mergedDf = pd.merge(mergedDf, weather, on="isqno")
mergedDf.set_index('isqno', inplace=True)
mergedDf = mergedDf[
    ['surpa', 'post1', 'post2', 'wx1', 'wx2', 'wx3', 'wx4', 'wx5', 'terra1', 'terra2', 'terra3', 'aeroa', 'wina']]

# Nan datas in 'wina' are supposed to be -1. Meaning that the attacker lost
mergedDf['wina'] = mergedDf['wina'].fillna(-1)

# Now we'll check the missing data and make the tables easier to analyze
surprise = mergedDf[['surpa', 'wina']]
print("Missing surprise values: ", surprise.isnull().sum())

terrain = mergedDf[['terra1', 'terra2', 'terra3', 'wina']]
print("Missing terrain values: ", terrain.isnull().sum())

weather = mergedDf[['wx1', 'wx2', 'wx3', 'wx4', 'wx5', 'wina']]
print("Missing weather values: ", weather.isnull().sum())

fortification = mergedDf[['post1', 'post2' ,'wina']]
print("Missing fortification values: ", fortification.isnull().sum())

aerialSuper = mergedDf[['aeroa', 'wina']]
print("Missing aerial superiority values: ", aerialSuper.isnull().sum())

surprise.to_csv('../Graphs/surprise_data.csv', index=False)
terrain.to_csv('../Graphs/terrain_data.csv', index=False)
weather.to_csv('../Graphs/weather_data.csv', index=False)
aerialSuper.to_csv('../Graphs/aerialSuper_data.csv', index=False)

# If there are too many missing values for imputation, it will be excluded. (post2, terra3)
combinedDf = mergedDf[['surpa', 'post1', 'wx1', 'wx2', 'wx3', 'wx4', 'wx5', 'terra1', 'terra2', 'aeroa', 'wina']]
print("Missing values: ",combinedDf.isnull().sum())


#since there's too many missing values in the wx2 column we will drop it and then remove all NaNs from the remaining data
column_list = [col for col in combinedDf.columns if col != 'wx2']

combinedDf = combinedDf[column_list].dropna()
print("Missing values after dropping: ",combinedDf.isnull().sum())

#Since we're trying to predict winning or losing, we don't care about draws and are going to remove them from the dataset
df_mask= combinedDf['wina'] != 0
combinedDf = combinedDf[df_mask]

#We're going to craete categories based on the way the Defense formed formations
# DL - D - Delaying action adopted
# WD - W - Withdrwal
# FD - F - Fortified defense
# HD - H - Hasty Defense
# PD - P - Prepared Defense

def check_post1():
    global combinedDf

    combinedDf['post1'] = combinedDf['post1'].map(lambda ca: ca[0])
    # dummy encoding
    post1_dummies = pd.get_dummies(combinedDf['post1'], prefix='post1')
    combinedDf = pd.concat([combinedDf, post1_dummies], axis=1)
    combinedDf.drop('post1', inplace=True, axis=1)
    return combinedDf

combinedDf = check_post1()
print("Delaying actions adopted", combinedDf['post1_D'].sum())
print("Fortified defense", combinedDf['post1_F'].sum())
print("Hasty Defense", combinedDf['post1_H'].sum())
print("Prepared Defense", combinedDf['post1_P'].sum())
print("Withdrwal", combinedDf['post1_W'].sum())


#Create categories based on wx1
#D : Dry -> 0
#W : wet -> 1

def check_wx1():
    global combinedDf
    combinedDf['wx1'] = combinedDf['wx1'].map(lambda s: 1 if s == 'W' else 0)
    return combinedDf
combinedDf = check_wx1()

#Create categories based on wx3 where in every column the value will be 0 or 1
#C : Cold
#H : Hot
#T : Temperate

def check_wx3():
    global combinedDf

    combinedDf['wx3'] = combinedDf['wx3'].map(lambda ca: ca[0])
    # dummy encoding
    wx3_dummies = pd.get_dummies(combinedDf['wx3'], prefix='wx3')
    combinedDf = pd.concat([combinedDf, wx3_dummies], axis=1)
    combinedDf.drop('wx3', inplace=True, axis=1)
    return combinedDf
combinedDf = check_wx3()

#Create categories based on wx4 where in every column the value will be 0 or 1
# S : Summer
# $ : Summer
# W : Winter
# F : Fall
def check_wx4():
    global combinedDf

    combinedDf['wx4'] = combinedDf['wx4'].map(lambda ca: ca[0])
    # dummy encoding
    wx4_dummies = pd.get_dummies(combinedDf['wx4'], prefix='wx4')
    combinedDf = pd.concat([combinedDf, wx4_dummies], axis=1)
    combinedDf.drop('wx4', inplace=True, axis=1)
    return combinedDf
combinedDf = check_wx4()

#Create categories based on wx5 where in every column the value will be 0 or 1
# E : Tropical (i.e., "Equatorial")
# D : Desert
# T : Temperate
def check_wx5():
    global combinedDf

    combinedDf['wx5'] = combinedDf['wx5'].map(lambda ca: ca[0])
    # dummy encoding
    wx5_dummies = pd.get_dummies(combinedDf['wx5'], prefix='wx5')
    combinedDf = pd.concat([combinedDf, wx5_dummies], axis=1)
    combinedDf.drop('wx5', inplace=True, axis=1)
    return combinedDf
combinedDf = check_wx5()

#Create categories based on terrrain1 where in every column the value will be 0 or 1
# R-Rolling
# G-Rugged
# F-Flat

def check_terra1():
    global combinedDf
    combinedDf['terra1'] = combinedDf['terra1'].map(lambda ca: ca[0])
    #dummy encoding
    terra1_dummies = pd.get_dummies(combinedDf['terra1'], prefix='terra1')
    combinedDf = pd.concat([combinedDf, terra1_dummies], axis=1)
    combinedDf.drop('terra1', inplace=True, axis=1)
    return combinedDf
combinedDf = check_terra1()

#Create categories based on terrrain2 where in every column the value will be 0 or 1
# B - Bare
# M - Mixed
# D - Desert
# W - Heavily wooded
def check_terra2():
    global combinedDf


    combinedDf['terra2'] = combinedDf['terra2'].map(lambda ca: ca[0])
    #dummy encoding
    terra2_dummies = pd.get_dummies(combinedDf['terra2'], prefix='terra2')
    combinedDf = pd.concat([combinedDf, terra2_dummies], axis=1)
    combinedDf.drop('terra2', inplace=True, axis=1)
    return combinedDf
combinedDf = check_terra2()

#Time to build and train the models
combinedDf['wina'] = combinedDf['wina'].apply(lambda x: x+1 if x == -1 else x)
combinedDf = combinedDf.astype('int')


combinedDf.to_csv('../Models/combinedData.csv', index=False)
