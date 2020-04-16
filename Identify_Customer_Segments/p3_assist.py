#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 13:58:21 2020

@author: dominickandolino
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, Imputer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans



# Load in the general demographics data.
azdias = pd.read_csv('Udacity_AZDIAS_Subset.csv', sep=';')

# Load in the feature summary file.
feat_info = pd.read_csv('AZDIAS_Feature_Summary.csv', sep=';')
  


# Step 1.1.1 ------------------------------------------------------------------------------------------------------------------------

def get_nan_dict(df):
    '''
    Input: feat_info dictionary
    Output: Nan dictionary. key = attribute. Value = proper list object of nan values
    Values in string form and int form if possible
    '''
    
    nan_dict = {} #initialize as empty
     
    for feat_i, feat_row in df.iterrows():   
        
        nan_lst = feat_row['missing_or_unknown'].strip('][').split(',') #parse string into proper list object
        
        nan_lst_int = []
        for val in nan_lst:
            try:
                nan_lst_int.append(int(val)) # add same value in integer form if possible
            except:
                pass
        
        nan_lst = nan_lst + nan_lst_int
        
        
        nan_dict[feat_row['attribute']] = nan_lst # add to dictinary, use atr as key
    
    
    return(nan_dict)
    
    

nan_dict = get_nan_dict(feat_info)


azdias_clean = azdias.copy()

#replace data with nan where value in nan_dict
azdias_clean = azdias_clean.apply(lambda col : col.replace(nan_dict[col.name], np.nan))


# Step 1.1.2 ------------------------------------------------------------------------------------------------------------------------


nan_cnts = azdias_clean.isna().sum() #gets count of nans per column
feat_info['nan_cnt'] = nan_cnts.values # add nan count to feat info df
nan_cnts.sort_values(ascending=False)[0:10] # get top 10 

nan_cnts.hist(bins=25)


feat_info.groupby(['information_level'])['nan_cnt'].mean().round().sort_values(ascending=False)



nan_q1, nan_q3 = nan_cnts.quantile([0.25,0.75])
nan_out_thrs = int(nan_q3 + (1.5 *(nan_q3-nan_q1)) )
nan_out_cols = list(nan_cnts[lambda x: x >= nan_out_thrs].index)

azdias_clean.drop(nan_out_cols, axis = 1, inplace=True)

'''
The IQR (InterQuartile Range) method was used to determine which columns to consider as outliers with respect to how many missing values they had. This resulted in a threshold of 291,287, so any columns with more empty values then this will be dropped. Translating to percentages, any column with more than 33% missing values will be dropped. 

The columns removed are: ['AGER_TYP', 'GEBURTSJAHR', 'TITEL_KZ', 'ALTER_HH', 'KK_KUNDENTYP', 'KBA05_BAUMAX']

On average, the microcell_rr3 level columns had the most missing values. 
'''


# Step 1.1.3 ------------------------------------------------------------------------------------------------------------------------


nan_cnts_row = azdias_clean.isna().sum(axis=1) #gets count of nans per row
azdias_clean['nan_cnt'] = nan_cnts_row.values #add nan count to df
nan_cnts_row.sort_values(ascending=False)[0:10] # get top 10 

nan_cnts_row.hist(bins=50)


nan_q1_row, nan_q3_row = nan_cnts_row.quantile([0.25,0.75])
nan_out_thrs_row = int(nan_q3_row + (1.5 *(nan_q3_row-nan_q1_row)) ) 
print('Row outlier if # nan >= {}'.format(nan_out_thrs_row))
 

azdias_clean_low_miss = azdias_clean.loc[azdias_clean.loc[:,'nan_cnt'] < nan_out_thrs_row] 
azdias_clean_high_miss = azdias_clean.loc[azdias_clean.loc[:,'nan_cnt'] >= nan_out_thrs_row] 

azdias_clean_low_miss.drop(['nan_cnt'], axis = 1, inplace=True)
azdias_clean_high_miss.drop(['nan_cnt'], axis = 1, inplace=True)



cols_no_miss = list(nan_cnts[lambda x: x ==0].index)[:10]

figure, axs = plt.subplots(nrows=len(cols_no_miss), ncols=2, figsize = (20,30))
figure.subplots_adjust(hspace = 0.3, wspace = 0.2)
for i in range(len(cols_no_miss)):
    sns.countplot(azdias_clean_low_miss[cols_no_miss[i]], ax = axs[i][0])
    axs[i][0].set_title('Low Missing Values')
    sns.countplot(azdias_clean_high_miss[cols_no_miss[i]], ax = axs[i][1])
    axs[i][1].set_title('High Missing Values')


'''
The IQR (InterQuartile Range) method was used to determine which rows to consider as outliers with respect to how many missing values they had. This resulted in a threshold of 7, so any rows with more empty values then this will be treated separetley. Translating to percentages, this is any row with more than 9% if it's columns missing values.  

There appears to be a difference in the financial columns between the data with high and low missing values. The data with high missing values are much more likeley to give a 5 (very low) to the ANLEGER (investor) feature. Similiarly, the other financial features are heavily weighted to either 3 (average) or 4 (low) for the data with high missing values. The data with low missing values is much more evenly spread for these atributes. 
'''



# Step 1.2.1------------------------------------------------------------------------------------------------------------------------


feat_info.loc[feat_info.loc[:,'nan_cnt'] < nan_out_thrs].groupby(['type'])['attribute'].count()


# get all categorical featuress
cat_feats = list(feat_info.loc[ 
                                    (feat_info.loc[:,'nan_cnt'] < nan_out_thrs) 
                                    & (feat_info.loc[:,'type'] == 'categorical') 
                               ]['attribute']
                )


#get unique values per cat feat, set as binary or multi-level
bin_feats = []
multi_lvl_feats = []
for feat in cat_feats:
    
    feat_unique_vals = len(azdias_clean_low_miss[feat].unique())
    
    if( feat_unique_vals == 2 ):
        bin_feats.append(feat)
        
    elif( feat_unique_vals > 2):
        multi_lvl_feats.append(feat)

print('Binary features: {}'.format(bin_feats))
print('Multi-level features: {}'.format(multi_lvl_feats))


#check for non-numeric binary feat
for feat in bin_feats:
    print('Featue: {},  Values: {}'.format(feat, azdias_clean_low_miss[feat].unique()))
    

#remapp binary non-numerical feature
azdias_clean_low_miss = azdias_clean_low_miss.replace({'OST_WEST_KZ': {'W': 0, 'O': 1}})

#drop multi-level cat feats
azdias_clean_low_miss.drop(multi_lvl_feats, axis = 1, inplace=True)

'''
There are 18 categorical features after we dropped columns with lots of missing data. 4 of these features are binary, with only OST_WEST_KZ containing non-numeric data. This feature was remapped to 0 and 1 and all of these features are kept. The 14 multi-level categorical features were dropped to keep things simple for this analysis. If we do not get the desired results, we could revist this and one-hot encode these features. 
'''


# Step 1.2.2------------------------------------------------------------------------------------------------------------------------

#get list of mixed features
mix_feats = list(feat_info.loc[ 
                                    (feat_info.loc[:,'nan_cnt'] < nan_out_thrs) 
                                    & (feat_info.loc[:,'type'] == 'mixed') 
                               ]['attribute']
                )



#initialize 2 new variables with same data
azdias_clean_low_miss['DECADE'] = azdias_clean_low_miss['PRAEGENDE_JUGENDJAHRE']
azdias_clean_low_miss['MOVEMENT'] = azdias_clean_low_miss['PRAEGENDE_JUGENDJAHRE']

#see data dict for mapping
dec_dict = {1:1, 2:1, 3:2, 4:2, 5:3, 6:3, 7:3, 8:4, 9:4, 10:5, 11:5, 12:5, 13:5, 14:6, 15:6}
mvmnt_dict = {1:1, 2:0, 3:1, 4:0, 5:1, 6:0, 7:0, 8:1, 9:0, 10:1, 11:0, 12:1, 13:0, 14:1, 15:0}

#replace with mapping
azdias_clean_low_miss['DECADE'].replace(dec_dict, inplace=True)
azdias_clean_low_miss['MOVEMENT'].replace(mvmnt_dict, inplace=True)



#initialize 2 new variables with same data
azdias_clean_low_miss['WEALTH'] = azdias_clean_low_miss['CAMEO_INTL_2015']
azdias_clean_low_miss['LIFE_STAGE'] = azdias_clean_low_miss['CAMEO_INTL_2015']

#see data dict for mapping
wlth_dict = {'11':1, '12':1, '13':1, '14':1, '15':1, '21':2, '22':2, '23':2,
             '24':2, '25':2, '31':3, '32':3, '33':3, '34':3, '35':3, '41':4, 
             '42':4, '43':4, '44':4, '45':4, '51':5, '52':5, '53':5, '54':5, 
             '55':5
            }


lf_stg_dict = {'11':1, '12':2, '13':3, '14':4, '15':5, '21':1, '22':2, '23':3, 
               '24':4, '25':5, '31':1, '32':2, '33':3, '34':4, '35':5, '41':1, 
               '42':2, '43':3, '44':4, '45':5, '51':1, '52':2, '53':3, '54':4, 
               '55':5
              }

#replace with mapping
azdias_clean_low_miss['WEALTH'].replace(wlth_dict, inplace=True)
azdias_clean_low_miss['LIFE_STAGE'].replace(lf_stg_dict, inplace=True)



azdias_clean_low_miss.drop(mix_feats, axis = 1, inplace=True)


azdias_clean_low_miss.describe()

'''
In total, there are 6 mixed-type features.

The mixed feature PRAEGENDE_JUGENDJAHRE was replaced with 2 new features, DECADE and MOVEMENT. The provided data dictionary was used to determine the correct mapping. Similiarly, CAMEO_INTL_2015 was replaced with 2 new features, WEALTH and LIFE_STAGE.

All other mixed-type features were dropped, including the orginal PRAEGENDE_JUGENDJAHRE and CAMEO_INTL_2015 features.
'''


# Step 1.3 ------------------------------------------------------------------------------------------------------------------------


def clean_data(df):
    """
    Perform feature trimming, re-encoding, and engineering for demographics
    data
    
    INPUT: Demographics DataFrame
    OUTPUT: Trimmed and cleaned demographics DataFrame
    
    Assumptions:
        nan_dict (cleaned list of nan values per col) already created
        nan_out_cols (cols with lots of nans) already created
        nan_out_thrs_row (threshold for dropping row with lots of nans) already created
        multi_lvl_feats already created
        dec_dict and mvmnt_dict already created
        wlth_dict and lf_stg_dict already created
        mix_feats already created
        
    """
    
  
    # convert missing value codes into NaNs, ...
    df_clean = df.copy()
    df_clean = df_clean.apply(lambda col : col.replace(nan_dict[col.name], np.nan))
    
    
    # remove selected columns and rows, ...
    df_clean.drop(nan_out_cols, axis = 1, inplace=True) #drop same cols with lots of nans
    
    nan_cnts_row = df_clean.isna().sum(axis=1) #gets count of nans per row
    df_clean['nan_cnt'] = nan_cnts_row.values #add nan count to df
    df_clean_low_miss = df_clean.loc[df_clean.loc[:,'nan_cnt'] < nan_out_thrs_row] #drop row if lots of cols are nan, same threshold
    df_clean_low_miss.drop(['nan_cnt'], axis = 1, inplace=True) #don't need this col now

    
    # select, re-encode, and engineer column values.
    df_clean_low_miss = df_clean_low_miss.replace({'OST_WEST_KZ': {'W': 0, 'O': 1}}) #remapp binary non-numerical feature
    df_clean_low_miss.drop(multi_lvl_feats, axis = 1, inplace=True) #drop multi-level cat feats
    
    df_clean_low_miss['DECADE'] = df_clean_low_miss['PRAEGENDE_JUGENDJAHRE'] #initialize 2 new variables with same data
    df_clean_low_miss['MOVEMENT'] = df_clean_low_miss['PRAEGENDE_JUGENDJAHRE']
    df_clean_low_miss['DECADE'].replace(dec_dict, inplace=True) #replace with mapping
    df_clean_low_miss['MOVEMENT'].replace(mvmnt_dict, inplace=True)
    
    df_clean_low_miss['WEALTH'] = df_clean_low_miss['CAMEO_INTL_2015'] #initialize 2 new variables with same data
    df_clean_low_miss['LIFE_STAGE'] = df_clean_low_miss['CAMEO_INTL_2015']
    df_clean_low_miss['WEALTH'].replace(wlth_dict, inplace=True) #replace with mapping
    df_clean_low_miss['LIFE_STAGE'].replace(lf_stg_dict, inplace=True)

    df_clean_low_miss.drop(mix_feats, axis = 1, inplace=True) #drop mixed-type features, inc 2 re-engineered above

    
    # Return the cleaned dataframe.
    return(df_clean_low_miss)



#cust_df_test = clean_data(pd.read_csv('Udacity_CUSTOMERS_Subset.csv', sep=';'))
#cust_df_test.describe()


# Step 2.1 ------------------------------------------------------------------------------------------------------------------------



imp = Imputer(strategy='most_frequent')
azdias_clean_low_miss_impute = imp.fit_transform(azdias_clean_low_miss)
azdias_clean_low_miss_impute = pd.DataFrame(azdias_clean_low_miss_impute) #cast as dataframe
azdias_clean_low_miss_impute.columns = azdias_clean_low_miss.columns #reset columns
azdias_clean_low_miss_impute.isna().sum().sort_values(ascending=False)[0:10] #verify no nan left

sclr = StandardScaler()
azdias_clean_low_miss_impute_scaled = sclr.fit_transform(azdias_clean_low_miss_impute)
azdias_clean_low_miss_impute_scaled = pd.DataFrame(azdias_clean_low_miss_impute_scaled)
azdias_clean_low_miss_impute_scaled.columns = azdias_clean_low_miss.columns
azdias_clean_low_miss_impute_scaled.describe() #verify mean 0, std 1

'''
Any remaining missing values were replaced with the most_frequent value for the column. This was done so we can properly Scale the fetures, preventing bias in future steps. 

The data was scaled using the StantardScaler, resulting in each feature having a mean of 0 and stantard deviation of 1. This was validated using the describe function on our dataframe, means of 2.76x10^-15 is essentially zero. 
'''

# Step 2.2 ------------------------------------------------------------------------------------------------------------------------


def scree_plot(pca):
    '''
    Creates a scree plot associated with the principal components 
    
    INPUT: pca - the result of instantian of PCA in scikit learn
            
    OUTPUT:
            None
    '''
    num_components=len(pca.explained_variance_ratio_)
    ind = np.arange(num_components)
    vals = pca.explained_variance_ratio_
 
    plt.figure(figsize=(10, 6))
    ax = plt.subplot(111)
    cumvals = np.cumsum(vals)
    ax.bar(ind, vals)
    ax.plot(ind, cumvals)
    #for i in range(num_components):
        #ax.annotate(r"%s%%" % ((str(vals[i]*100)[:4])), (ind[i]+0.2, vals[i]), va="bottom", ha="center", fontsize=12)
 
    ax.xaxis.set_tick_params(width=0)
    ax.yaxis.set_tick_params(width=2, length=12)
 
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Variance Explained (%)")
    plt.title('Explained Variance Per Principal Component')


pca = PCA()
pca.fit(azdias_clean_low_miss_impute_scaled)
scree_plot(pca)



pca_20 = PCA(n_components = 20)
azdias_pca = pca_20.fit_transform(azdias_clean_low_miss_impute_scaled)


'''
Examining the scree plot when fitting PCA using all features, shows that 80% of the variability can be explained with just 20 features. Therefore, we can reduce the number of features by 68% and still retain much of the information.  
'''



# Step 2.3 ------------------------------------------------------------------------------------------------------------------------

def pc_weights(pca, feat_lst, i):
    
    df = pd.DataFrame(pca.components_) #cast as dataframe, row per PC, col per input feature
    df.columns = feat_lst #set column names
    wghts = df.iloc[i, :].sort_values(ascending=False) # get ith row sorted
    
    return(wghts)


feat_lst = list(azdias_clean_low_miss_impute_scaled.columns)
pc_weights(pca_20, feat_lst, 0)
pc_weights(pca_20, feat_lst, 1)
pc_weights(pca_20, feat_lst, 2)

'''
Examing the weights of the 1st principal component shows correlations related to macro-cell features. PLZ8_ANTG3 (number of 6-10 family homes) is negativley correlated to MOBI_REGIO (movement patterns).

Examing the weights of the 2nd principal component shows ALTERSKATEGORIE_GROB (estimated age) is negativley correlated with SEMIO_REL (relgious) personalities.

Examing the weights of the 3rd principal component shows interesting correlations related to personality type. SEMIO_VERT (dreamful) and SEMIO_SOZ (socially-minded) are positvely correlated. SEMIO_KAEM (combative) is negativley correlated to dreamful. ANREDE_KZ (gender) is positvely correlated with combative.
'''

# Step 3.1 ------------------------------------------------------------------------------------------------------------------------


def get_kmeans_score(data, center):

    kmeans = KMeans(n_clusters = center)
    model = kmeans.fit(data)
    score = np.abs(model.score(data))
    
    return(score)


scores = []
centers = list(range(1,25))

for center in centers:
    scores.append(get_kmeans_score(azdias_pca, center))
    
plt.plot(centers, scores, linestyle='--', marker='o', color='b')
plt.xlabel('K')
plt.ylabel('SSE')
plt.title('SSE vs. K')


kmeans_7 = KMeans(n_clusters = 7)
azdias_model = kmeans_7.fit(azdias_pca)
azdias_labels = azdias_model.predict(azdias_pca)

'''
The above scree plot does not have any clear "elbow". It starts by decreasing exponentially until about k=7, and is then mostly decreasing lineraly. For this reason, I will be moving forward with k=7.
'''

# Step 3.2 ------------------------------------------------------------------------------------------------------------------------


customers = clean_data(pd.read_csv('Udacity_CUSTOMERS_Subset.csv', sep=';'))

#impute nans
customers_impute = imp.transform(customers)
customers_impute = pd.DataFrame(customers_impute) #cast as dataframe
customers_impute.columns = customers.columns #reset columns

#apply StandardScaler
customers_impute_scaled = sclr.transform(customers_impute)
customers_impute_scaled = pd.DataFrame(customers_impute_scaled)
customers_impute_scaled.columns = customers.columns

#apply pca_20
customers_pca = pca_20.transform(customers_impute_scaled)
customers_labels = azdias_model.predict(customers_pca)


# Step 3.3 ------------------------------------------------------------------------------------------------------------------------


azdias_plot_df = pd.DataFrame(pd.DataFrame(azdias_labels).groupby([0])[0].count())
azdias_plot_df.columns = ['cnts']
azdias_plot_df['percent'] = azdias_plot_df/azdias_plot_df.sum()
azdias_plot_df['cluster']  = azdias_plot_df.index

customers_plot_df = pd.DataFrame(pd.DataFrame(customers_labels).groupby([0])[0].count())
customers_plot_df['percent'] = customers_plot_df/customers_plot_df.sum()
customers_plot_df.columns = ['cnts', 'percent']
customers_plot_df['cluster']  = customers_plot_df.index


figure, axs = plt.subplots(nrows=1, ncols=2, figsize = (10,5))
figure.subplots_adjust(hspace = 1, wspace=.3)
sns.barplot(x = 'cluster', y = 'percent', data = azdias_plot_df, ax = axs[0])
axs[0].set_title('Azdias Labels')
sns.barplot(x = 'cluster', y = 'percent', data = customers_plot_df, ax = axs[1])
axs[1].set_title('Customer Labels')


#biggest delta between azdias and customer data
(customers_plot_df['percent'] - azdias_plot_df['percent']).sort_values(ascending=False)



pc_weights(pca_20, feat_lst, 0)


cluster_3 = sclr.inverse_transform(pca_20.inverse_transform(azdias_model.cluster_centers_[3]))
cluster_3 = pd.DataFrame(cluster_3)
cluster_3.index = feat_lst
cluster_3.loc[['GREEN_AVANTGARDE','EWDICHTE','ORTSGR_KLS9', 'MOVEMENT','KKK','HH_EINKOMMEN_SCORE']]

'''
Cluster 3 is a clear leader in overrepresenting the customer base, it represnets 50% of the customers and only 17.5% of the general population. People in this cluster are charactersied as living in communities with 90-149 household per square kilometer (EWDICHTE) and 10,000 - 20,0000 inhabitants (ORTSGR_KLS9). They also have slighlty above average income (HH_EINKOMMEN_SCORE).

Cluster 5 was the most underrpresented cluster. It represents 3% of customers but is 14% of the general popuilation. Charactersistics of this cluster are they have almost no 10+ family houses in their microcell (KBA05_ANTG4) and almost no academeic title holders in thier building (ANZ_HH_TITEL).
'''

#------------------------------------------------------------------------------------------------------------------------












