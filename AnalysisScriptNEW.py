import AnalysisFunctions as af

import pandas as pd  
#defaultdict to use nested dictionaries
from collections import defaultdict

import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np

"""
-------------------------------------------------------------------------------------------------------------------------------------------------
PRE-PROCESSING: Import data, organize them, set observations subsets, calculate the quantiles:
-------------------------------------------------------------------------------------------------------------------------------------------------
"""

# Dictionary initialization:
df = af.dictionary()
list(df.keys())

# Decide simulation starting time:
sim_start = '2018-10-26 12:00:00'

# Some parameters for the basin:
Verzasca_area = 186*1000.0**2 #m2
conv_factor = Verzasca_area/(1000.0*3600.0) 

# Runoff observation: open observation dataframe
obs_pattern = '/home/ciccuz/hydro/prevah/runoff/2605.dat'
obs_columns = ['year','month','day','hour','runoff']
obs_df = pd.DataFrame(pd.read_csv(obs_pattern, names=obs_columns, delim_whitespace=True, header=None))
obs_df['date'] = pd.to_datetime(obs_df[['year', 'month', 'day', 'hour']])

# Precipitation observation: open precipitation observation dataframe obtained by cosmo1+pluviometer 
#data concatenated series before the initialization of the model
prec_obs_df = af.dictionary(pattern="/home/ciccuz/hydro/PrecObs/cosmo1_{simul_time}/{otherstuff}",
                        folders_pattern = '/home/ciccuz/hydro/PrecObs/cosmo1_*')
prec_obs_series = prec_obs_df['2018-11-09 12:00:00']['Ver500.'][['P-kor','date']].loc[prec_obs_df['2018-11-09 12:00:00']['Ver500.'].date < '2018-11-09 13:00:00']

# Extract from the dictionary the dataframe containing all the different realizations of the same event: 
#every ensemble member and parameter set combination for the runoff, every ensemble member for the precipitation.
ens_df_prec = af.ensemble_df(df, sim_start, Verzasca_area, 'P-kor')
ens_df_runoff = af.ensemble_df(df, sim_start, Verzasca_area,'RGES')

# Calculate the quantiles for the variable chosen considering all the different realizations for the 120h ahead.
quant_prec = af.quantiles(ens_df_prec)
quant_runoff = af.quantiles(ens_df_runoff)

# Define the subset of runoff and precipitation observation based on quantiles dataframe date boundaries
obs_indexes_runoff = obs_df.loc[obs_df.index[obs_df['date'] == str(quant_runoff.date[0])] |
        obs_df.index[obs_df['date'] == str(quant_runoff.date[119])]]
obs_indexes_prec = prec_obs_series.loc[prec_obs_series.index[prec_obs_series['date'] == str(quant_runoff.date[0])] |
        prec_obs_series.index[prec_obs_series['date'] == str(quant_runoff.date[119])]]

obs_subset = obs_df.loc[obs_indexes_runoff.index[0]:obs_indexes_runoff.index[1]]
prec_obs_subset = prec_obs_series.loc[obs_indexes_prec.index[0]:obs_indexes_prec.index[1]]


# Observed runoff plot for 10-11 2018:
fig, ax = plt.subplots(1, 1, figsize=(10,6), dpi=100)
obsss = obs_df.loc[obs_df.year == 2018].loc[obs_df.month >= 10]
plt.plot(obsss.date, obsss.runoff)
plt.ylim(20,200)
plt.show()


"""
______________________________________________________________________
Spaghetti and hydrograph plotting for the entire set of realizations:
______________________________________________________________________
"""

# Spaghetti plot of all realizations
af.spaghetti_plot(ens_df_runoff, ens_df_prec, obs_subset, prec_obs_subset, sim_start)

# Hydrograph plot for all realizations
af.hydrograph(quant_runoff, quant_prec, obs_subset, prec_obs_subset, sim_start)


"""
________________________
Meteorological medians:
________________________
"""

# Select groups of realizations based on the same ensemble members:
# dictionaries sorted by ensemble members
rm_groups_runoff = af.ens_param_groups(ens_df_runoff)[0]

# Quantiles dictionaries from above rm groups dictionary
quant_rm_dict = lambda: defaultdict(quant_rm_dict)
quant_rm_groups_runoff = quant_rm_dict()

for rm in range(21):
    quant_rm_groups_runoff[rm] = af.quantiles(rm_groups_runoff[rm])

# Construct a dataframe having all the medians obtained for every group of realizations 
# associated to an ens member
rm_medians = pd.DataFrame(index=range(120))

for rm in range(21):
    rm_medians[rm] = quant_rm_groups_runoff[rm]['0.5']
rm_medians['date'] = quant_rm_groups_runoff[rm]['date']

rm_medians.columns = ['rm00','rm01','rm02','rm03','rm04','rm05','rm06','rm07','rm08','rm09','rm10','rm11','rm12',
                     'rm13','rm14','rm15','rm16','rm17','rm18','rm19','rm20','date']

# Quantiles on rm medians:
quant_rm_medians = af.quantiles(rm_medians)


"""
-------------------------------------------------------------------------------------------------------------------------------------------------
DECOMPOSED SOURCES OF UNCERTAINTIES: meteorological and hydrological uncertainties
-------------------------------------------------------------------------------------------------------------------------------------------------
______________________________
- Meteorological uncertainty
______________________________
"""

#Spaghetti plot with the 21 rm medians: 
af.spaghetti_plot(rm_medians, ens_df_prec, obs_subset, prec_obs_subset, sim_start, medians=True)

# Quantify the meteorological uncertainty by plotting the range of spread among all the 21 rm medians obtained:
#af.hydrograph(quant_rm_medians, quant_prec, obs_subset, prec_obs_subset, sim_start, medians=True)
af.comparison_meteo_hydrograph(quant_rm_medians, quant_runoff, quant_prec, obs_subset, prec_obs_subset, sim_start)


"""
_____________________________________________
- Hydrological uncertainty (of the forecast)
_____________________________________________
"""

# example: spaghetti and hydrograph plots for 1 selected ens member:
rm = 10
af.spaghetti_plot(rm_groups_runoff[rm], ens_df_prec, obs_subset, prec_obs_subset, sim_start,
               runoff_label='\n'.join((r'rm = %02d' % rm,  r'All pin realizations')))
af.hydrograph(quant_rm_groups_runoff[rm], quant_prec, obs_subset, prec_obs_subset, sim_start)

# Look at different rm realizations how the hydrological spread behaves: detect three realizations having different behaviours
""" For sim_start = '2018-10-27 00:00:00'

rm_high = 19
rm_medium = 11
rm_low = 13
"""

"""For sim_start = '2018-10-30 00:00:00'
"""
rm_high = 7
rm_medium = 6
rm_low = 17

af.hydrograph_rms(rm_high, rm_medium, rm_low, ens_df_prec, quant_rm_groups_runoff, quant_runoff, 
                  obs_subset, prec_obs_subset, sim_start)

# Quantify the hydrological uncertainty considering the quantiles around every rm median:
af.hydro_unc_boxplot(quant_rm_groups_runoff, sim_start, normalized = True)
#plt.savefig('/home/ciccuz/Thesis/hydro_unc_boxplot2.pdf', bbox_inches='tight', dpi=1000)


"""
________________________________________________________________________________________________________
- Hydrological uncertainty (past): needed for a certain sim_start the initialization 5 days ahead of it
________________________________________________________________________________________________________
"""

#Hydrological uncertainty in the past: look at the 5 days before the initialization date
past_sim_start = str(ens_df_runoff.date[119])

past_ens_df = af.past_hydro_unc_ensemble_df(df, past_sim_start, Verzasca_area, 'RGES') 

past_quant = af.quantiles(past_ens_df)

past_obs_indexes_runoff = obs_df.loc[obs_df.index[obs_df['date'] == str(past_quant.date[0])] |
        obs_df.index[obs_df['date'] == str(past_quant.date[119])]]
past_obs_indexes_prec = prec_obs_series.loc[prec_obs_series.index[prec_obs_series['date'] == str(past_quant.date[0])] |
        prec_obs_series.index[prec_obs_series['date'] == str(past_quant.date[119])]]

past_obs_subset = obs_df.loc[past_obs_indexes_runoff.index[0]:past_obs_indexes_runoff.index[1]]
past_prec_obs_subset = prec_obs_series.loc[past_obs_indexes_prec.index[0]:past_obs_indexes_prec.index[1]]


af.spaghetti_plot(past_ens_df, ens_df_prec, past_obs_subset, past_prec_obs_subset, past_sim_start, past=True)

af.hydrograph(past_quant, quant_prec, past_obs_subset, past_prec_obs_subset, past_sim_start, past=True)


"""
-------------------------------------------------------------------------------------------------------------------------------------------------
HYDROLOGICAL PARAMETERS ANALYSIS: use file hydro_parameters.py
-------------------------------------------------------------------------------------------------------------------------------------------------
"""


"""
-------------------------------------------------------------------------------------------------------------------------------------------------
FORECAST VERIFICATION: use file verification.py
-------------------------------------------------------------------------------------------------------------------------------------------------
"""

'''Correlation:'''

X = obs_subset['runoff']
y = quant_runoff['0.5']

X = sm.add_constant(X)
model = sm.OLS(y, X)
results = model.fit()

print(results.summary())

# Correlation plot:
af.correlation_plot(quant_runoff['0.5'], obs_subset, lead_times, title_text = ', all realizations median')


"""
-------------------------------------------------------------------------------------------------------------------------------------------------
PEAK-BOX APPROACH
-------------------------------------------------------------------------------------------------------------------------------------------------
"""

import peakbox_classic_multipeaksV2 as pb

pb.peak_box_multipeaks(rm_medians, obs_subset, sim_start,  delta_t=10, gamma=0.6, decreashours=10, beta = 0.8)
#plt.savefig('/home/ciccuz/Thesis/PeakBox/AAAAAAAAAA2.pdf', bbox_inches='tight', dpi=1000)


import peakbox_classic_multipeaksV3_cluster as pbk

pbk.peak_box_multipeaks_kmeans(rm_medians, obs_subset, sim_start,  delta_t=10, gamma=0.6)
plt.savefig('/home/ciccuz/Thesis/PeakBox/AAAAAAAAAA2.pdf', bbox_inches='tight', dpi=1000)

"""
-------------------------------------------------------------------------------------------------------------------------------------------------
CLUSTER ANALYSIS
-------------------------------------------------------------------------------------------------------------------------------------------------
""" 

"""
_______________________________
- RM extraction and dendrogram
_______________________________
"""

import cluster_funct as cl

#plot the dendrogram
pgf_with_latex = {"pgf.texsystem": "xelatex",
        "text.usetex": False}  
cl.clustered_dendrogram(ens_df_prec.drop('date', axis=1), sim_start)
#plt.savefig('/home/ciccuz/Thesis/cluster/dendrogram.pdf', bbox_inches='tight', dpi=1000)

#choose how many clusters you want (3 or 5):
Nclusters = 5

#representative members
RM = cl.clustered_RM(ens_df_prec.drop('date', axis=1), sim_start, Nclusters = Nclusters)

#extract the sub-dataframe for prec and runoff forecasts containing only the members related to the new extracted representative members:
clust_ens_df_prec = pd.DataFrame()
clust_ens_df_runoff = pd.DataFrame()

for rm_index in range(Nclusters):
    clust_ens_df_prec = pd.concat([clust_ens_df_prec, ens_df_prec.loc[:, ens_df_prec.columns == f'rm{RM[rm_index]:02d}_pin01']], axis=1, sort=False)
    for pin in range(1,26):
        clust_ens_df_runoff = pd.concat([clust_ens_df_runoff, ens_df_runoff.loc[:, ens_df_runoff.columns == f'rm{RM[rm_index]:02d}_pin{pin:02d}']], axis=1, sort=False)

clust_ens_df_prec = pd.concat([clust_ens_df_prec, ens_df_prec.date], axis=1)
clust_ens_df_runoff = pd.concat([clust_ens_df_runoff, ens_df_runoff.date], axis=1)

# Cluster quantiles:
clust_quant_prec = af.quantiles(clust_ens_df_prec)
clust_quant_runoff = af.quantiles(clust_ens_df_runoff)

"""
___________________________________________________________________________________________________________
- Hydrograph and spaghetti plots for the RMs extracted, compared to the spread obtained without clustering
___________________________________________________________________________________________________________
"""

# Spaghetti plot of clustered forecasts
af.spaghetti_plot(clust_ens_df_runoff, clust_ens_df_prec, obs_subset, prec_obs_subset, sim_start, clustered=True)

# Hydrograph plot of clustered forecasts
cl.cluster_hydrograph(clust_quant_runoff, clust_quant_prec, quant_runoff, quant_prec, obs_subset, prec_obs_subset, sim_start, Nclusters=Nclusters)
#plt.savefig(f'/home/ciccuz/Thesis/cluster/cluster_hydrograph_{Nclusters}RM.pdf', bbox_inches='tight')


"""
_________________________________
- Cluster meteorological medians
_________________________________
"""

# Select groups of realizations based on the same ensemble members:
# dictionaries sorted by ensemble members
clust_rm_groups_runoff = af.ens_param_groups(clust_ens_df_runoff)[0]

# Quantiles dictionaries from above rm groups dictionary
clust_quant_rm_dict = lambda: defaultdict(clust_quant_rm_dict)
clust_quant_rm_groups_runoff = clust_quant_rm_dict()

for rm in RM:
    clust_quant_rm_groups_runoff[rm] = af.quantiles(clust_rm_groups_runoff[rm])

# Construct a dataframe having all the medians obtained for every group of realizations 
# associated to an ens member
clust_rm_medians = pd.DataFrame(index=range(120))

for rm in RM:
    clust_rm_medians[rm] = clust_quant_rm_groups_runoff[rm]['0.5']
clust_rm_medians['date'] = clust_quant_rm_groups_runoff[rm]['date']

clust_rm_medians.columns = np.append(['rm' + f'{rm:02d}' for rm in RM], 'date')

# Quantiles on cluster rm medians:
clust_quant_rm_medians = af.quantiles(clust_rm_medians)

#Spaghetti plot with the 5 cluster rm medians: 
af.spaghetti_plot(clust_rm_medians, clust_ens_df_prec, obs_subset, prec_obs_subset, sim_start, clustered=True, medians=True)

# Quantify the meteorological uncertainty by plotting the range of spread among all rm medians obtained:
cl.cluster_hydrograph(clust_quant_rm_medians, clust_quant_prec, quant_rm_medians, quant_prec, obs_subset, prec_obs_subset, sim_start, Nclusters=Nclusters,
                      medians=True)


"""
_________________________________________
- Peak-box approach with clustered subset
_________________________________________
"""

pb.peak_box_multipeaks(clust_rm_medians, obs_subset, sim_start, delta_t=10, gamma=0.6, decreashours=10, beta = 0.6)









