import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  
#import seaborn as sns
import glob
from parse import parse

#defaultdict to use nested dictionaries
from collections import defaultdict

#quantiles calculation
from scipy.stats.mstats import mquantiles

#datetime conversion
from dateutil import parser

#statistical tools
from statsmodels import robust
import statsmodels.api as sm

#dates
import matplotlib.dates as mdates

#patches for legend
import matplotlib.patches as mpatches
from matplotlib.patches import Patch
#for legend to avoid repeating duplicates labels
from collections import OrderedDict

import seaborn as sns

#calculate area under curve for ROC curve:
from sklearn.metrics import auc

#find local peaks of a 2d function
from scipy.signal import find_peaks

#decide color series
import itertools

def dictionary(fileformat='std', pattern="/home/ciccuz/hydro/forecasts/cosmoe_prevah/cosmoe_{simul_time}/cosmoe_{something}_{Name}/{otherstuff}",
               folders_pattern = '/home/ciccuz/hydro/forecasts/cosmoe_prevah/cosmoe_*'):
    
    """
    Open every simulations run for different starting simulation times and create a dictionary of dataframes nested in
    this way: based on the simulation time, choosing the realization (e.g. 'rm00_pin01') you have a dataframe of different
    paramters and a number of rows given by the hourly time points 
    """
    
    #create a nested dictionary with two 'levels' to contain a list of dataframes for every simulation time
    #and every ensemble member
    nested_dict = lambda: defaultdict(nested_dict)
    nested_df_collection = nested_dict()
    
    #pattern to rename every dataframe (w different 'filepath') of the collection by the name of the simulation
    #pattern = "/home/ciccuz/hydro/forecasts/cosmoe_prevah/cosmoe_{simul_time}/cosmoe_{something}_{Name}/{otherstuff}"

    #sim_dates: string array to store the renamed 'filetime' variables with the initialization time of the simulation
    sim_dates = ["" for filetime in sorted(glob.iglob(folders_pattern))]
    i = 0
    
    #conditions on fileformat given in input to write the dataframes in the dictionary:
    if fileformat == 'q':
        skiprows = [1]
        usecols = range(12)
        columns = ['year', 'month', 'day', 'hour', 'RTOT', 'RTOT (l s-1 )', 'R0', 'R1', 'R2', 'RG1', 'RG2', 'RG3']
    if fileformat == 'std':
        skiprows = [0,1]
        usecols = range(20)
        columns = ['year', 'month', 'day', 'hour', 'NRTFL', 'P-uk', 'P-kor', 'P-SNO', 'EPOT', 'EREA', 'RO', 'R1', 'R2', 'RGES', 'S-SNO', 'SI', 'SSM', 'SUZ', 'SLZ', '??1', '??2', '??3', '??4']
    
    #for loop for every simulation made at different times
    for filetime in sorted(glob.iglob(folders_pattern)):
    
        #for loop to read every *.std/*.q file in every subdirectory present, sorted by name, and to create an array of 
        #dataframes
        #(all data in files *.q except RTOT (l s-1) are dimensioned in mm/h)
        #before that, if condition for distinguish different patterns considering the forecasts or the prec obs
        
        if  folders_pattern == '/home/ciccuz/hydro/forecasts/cosmoe_prevah/cosmoe_*':
            subfold = '/*/*.'
        elif folders_pattern == '/home/ciccuz/hydro/PrecObs/cosmo1_*':
            subfold = '/*.'
        
        for filepath in sorted(glob.iglob(filetime + subfold + fileformat)):
            nested_df_collection[filetime][filepath] = pd.DataFrame(pd.read_csv(filepath, skiprows=skiprows, 
                                                                                    delim_whitespace=True, header=None,
                                                                                    names=columns,
                                                                                    usecols=usecols))
            if fileformat == 'q':
                nested_df_collection[filetime][filepath].columns = columns
                
            #add complete date column to every dataframe
            nested_df_collection[filetime][filepath]['date'] = pd.to_datetime(nested_df_collection[filetime]
                                                                                  [filepath][['year', 'month', 'day', 
                                                                                              'hour']])
            
            # If considering ensemble members: change name of every dataframe ('filepath') of the dictionary by its 
            # simulation name (depending on ensemble member and parameter set used)
            if  folders_pattern == '/home/ciccuz/hydro/forecasts/cosmoe_prevah/cosmoe_*':
                newname_filepath = parse(pattern + fileformat, filepath)
                nested_df_collection[filetime][newname_filepath['Name']] = nested_df_collection[filetime].pop(filepath)
                
            elif folders_pattern == '/home/ciccuz/hydro/PrecObs/cosmo1_*':
                newname_filepath = parse(pattern + fileformat, filepath)
                nested_df_collection[filetime][newname_filepath['otherstuff']] = nested_df_collection[filetime].pop(filepath)
            
            
        #change name of every simulation time ('filetime') substituting it with the date of the simulation
        #locate characters for year, month, day, hour in filetime strings
        
        #if condition to account for cosmoe data or cosmo1 (for prec obs):
        if folders_pattern == '/home/ciccuz/hydro/forecasts/cosmoe_prevah/cosmoe_*' :
            sim_year = filetime[50:54] #[70:74] second ones used for longer file patterns i.e. located in deeper subfolders 
            sim_month = filetime[54:56] #[74:76]
            sim_day = filetime[56:58] #[76:78]
            sim_hour = filetime[58:60] #[78:80]
            
            #condition on hour: 00 or 12 UTC simulation start
            if sim_hour[0] == '0':
                sim_hour = '00'
            else:
                sim_hour = '12'
        
        elif folders_pattern == "/home/ciccuz/hydro/PrecObs/cosmo1_*":
            sim_year = filetime[34:38]
            sim_month = filetime[38:40]
            sim_day = filetime[40:42]
            sim_hour = filetime[42:44]
            
            if sim_hour[0] == '0':
                sim_hour = '00'
       
        sim_dates[i] = (sim_year+'-'+sim_month+'-'+sim_day+' '+sim_hour+':00:00')
        nested_df_collection[sim_dates[i]] = nested_df_collection.pop(filetime)
        i = i+1
                                                                        
    return nested_df_collection



def prec_obs_series():
    
    '''
    Read all the precipitation data obtained by a combination of COSMO1 and pluviometer data to obtain a precipitation series
    to be used as observation series.
    WARNING: for the day 2-11-2018 the data at 12:00 is missing!
    '''
    
    # Create a dictionary of all precipitation datasets (obtained with COSMO1) present at different sim_start
    prec_obs_df = dictionary(pattern="/home/ciccuz/hydro/PrecObs/cosmo1_{simul_time}/{otherstuff}",
                            folders_pattern = '/home/ciccuz/hydro/PrecObs/cosmo1_*')
    
    # Create a dataframe that will contain the "observed" precipitation series obtained by the different simulations/pluviometer
    # data interpolated of precipitation by taking the first 12 hours for every series in prec_obs_df and concatenate all of them
    obs_prec = pd.DataFrame(columns = ['year', 'month', 'day', 'hour', 'P-uk', 'P-kor', 'date'])
    
    #array of dates to consider every simulation start at 12 utc from 23-10 to 9-11 2018
    sim_starts = ['2018-10-23 12:00:00', '2018-10-24 12:00:00', '2018-10-25 12:00:00', '2018-10-26 12:00:00',
                  '2018-10-27 12:00:00', '2018-10-28 12:00:00', '2018-10-29 12:00:00', '2018-10-30 12:00:00',
                  '2018-10-31 12:00:00', '2018-11-01 12:00:00', '2018-11-02 13:00:00', '2018-11-03 12:00:00',
                  '2018-11-04 12:00:00', '2018-11-05 12:00:00', '2018-11-06 12:00:00', '2018-11-07 12:00:00',
                  '2018-11-08 12:00:00', '2018-11-09 12:00:00']
        
    i=0
    for sim_start in sim_starts:
        prec_set = prec_obs_df[sim_start]['Ver500.']
        
        #Compute the subset to consider just the 24 h above the initialization time:
        #to do so we need to do some if conditions because on the 2-11 the simulation starting at 12 is not present!    
        
        if sim_start == '2018-11-01 12:00:00' :
            prec_subset = prec_set.loc[(prec_set.date >= sim_start) & (prec_set.index <= 443)].drop(['NRTFL', 'P-SNO', 'EPOT', 'EREA',
                                   'RO', 'R1', 'R2', 'RGES', 'S-SNO', 'SI', 'SSM', 'SUZ', 'SLZ', '??1'], axis=1)
            prec_subset.index = range(i*24,i*24+24+1)
            
        if sim_start == '2018-11-02 13:00:00':
            prec_subset = prec_set.loc[(prec_set.date >= sim_start) & (prec_set.index <= 442)].drop(['NRTFL', 'P-SNO', 'EPOT', 'EREA',
                                   'RO', 'R1', 'R2', 'RGES', 'S-SNO', 'SI', 'SSM', 'SUZ', 'SLZ', '??1'], axis=1)
            prec_subset.index = range(i*24+1,i*24+24)
            
        else:
            prec_subset = prec_set.loc[(prec_set.date >= sim_start) & (prec_set.index <= 442)].drop(['NRTFL', 'P-SNO', 'EPOT', 'EREA',
                                       'RO', 'R1', 'R2', 'RGES', 'S-SNO', 'SI', 'SSM', 'SUZ', 'SLZ', '??1'], axis=1)
            prec_subset.index = range(i*24,i*24+24)
            
        obs_prec = pd.concat([obs_prec, prec_subset])
        
        i=i+1

    return obs_prec


def ensemble_df(df, sim_start, Verzasca_area, variable_type):
    
    """
    Create a dataframe containing all the different realizations, based on the dictionary created before,
    on simulation start time and on the variable in which we are interested. 
    The resulting dataframe will have a number of column = # realizations (525 for all the combinations 
    of realizations) and a number of rows given by the total lead time expressed in hours (120 h for our case) 
    """
    
    #initialize the dataframe that contains all the realizations for a particular variable
    ens_df = pd.DataFrame()

    #index to account for the right dates without accounting them more than one time
    j=0

    #initialization of array to store the 120 hours dates
    date_array = ["" for x in range(121)]

    #condition on the variable chosen to convert discharge in m3/s:
    if (variable_type == 'RTOT') or (variable_type == 'RGES'):
        conv_factor = Verzasca_area/(1000.0*3600.0)
    else:
        conv_factor = 1.0
    
    #for cycle on different members/paramsets
    for member in df[sim_start].keys():
      
        #for cycle on different dates 
        for date in df[sim_start][member]['date']:
        
            #series of if conditions to account for the 120 hours just after the initialization point and not before
            
            #case if we are on the same month -> must consider month and day 
            if (str(date)[5:7] == sim_start[5:7]):
                #case if we are on the same day -> must consider hour
                if (str(date)[8:10] == sim_start[8:10]):
                    if (str(date)[11:13] >= sim_start[11:13]):
                    
                        #if condition to take just the first set of the next 120 hours without having many copies of them
                        if j >=0 and j <=120:
                            date_array[j] = date
                            j = j+1
                            
                        #condition for precipitation to pick just the ensemble members and not every parameter set, 
                        #since for prec do not change
                        if variable_type == 'P-kor':
                            if member[8:10] == '01':
                                ens_df[member] = df[sim_start][member][variable_type].loc[df[sim_start][member]['date'] > sim_start]*conv_factor
                        else:
                            ens_df[member] = df[sim_start][member][variable_type].loc[df[sim_start][member]['date'] > sim_start]*conv_factor
                
                if (str(date)[8:10] > sim_start[8:10]):
            
                    #if condition to take just the first set of the next 120 hours without having many copies of them
                    if j >=0 and j <=120:
                        date_array[j] = date
                        j = j+1
                            
                    #condition for precipitation to pick just the ensemble members and not every parameter set, 
                    #since for prec do not change
                    if variable_type == 'P-kor':
                        if member[8:10] == '01':
                            ens_df[member] = df[sim_start][member][variable_type].loc[df[sim_start][member]['date'] > sim_start]*conv_factor
                    else:
                        ens_df[member] = df[sim_start][member][variable_type].loc[df[sim_start][member]['date'] > sim_start]*conv_factor
                    
            #case if we are in differen months -> can consider just the month and not the day
            if (str(date)[5:7] > sim_start[5:7]):
                
                #if condition to take just the first set of the next 120 hours without having many copies of them
                if j >=0 and j <=120:
                    date_array[j] = date
                    j = j+1
                    
                #condition for precipitation to pick just the ensemble members and not every parameter set, 
                #since for prec do not change
                if variable_type == 'P-kor':
                    if member[8:10] == '01':
                        ens_df[member] = df[sim_start][member][variable_type].loc[df[sim_start][member]['date'] > sim_start]*conv_factor
                else:
                    ens_df[member] = df[sim_start][member][variable_type].loc[df[sim_start][member]['date'] > sim_start]*conv_factor

    ens_df['date'] = date_array[1:]
    ens_df.index = range(120)
    
    return ens_df

def ens_param_groups(ens_df_runoff):
    
    """
    From the ensemble dataframe, select groups of realizations based on the same ensemble members/parameters set:
    create two dictionaries to contain the data based on choice of the representative members or of the parameters set 
    """
    
    
    #define a dictionary to contain the realizations based on groups of different rm
    ens_members_dict = lambda: defaultdict(ens_members_dict)
    ens_members_groups = ens_members_dict()

    #considering all representative members from 00 to 20
    for rm in range(21):

        ens_members_groups[rm] = pd.DataFrame(index=range(120))

        for realization in ens_df_runoff.columns[~ens_df_runoff.columns.isin(['date'])]:

            #take just the realizations corresponding to the same rm
            if str(realization)[2:4] == str('%02d' % rm):
                ens_members_groups[rm][str(realization)] = ens_df_runoff[str(realization)]

        ens_members_groups[rm]['date'] = ens_df_runoff['date']

    
    #define a dictionary to contain the realizations based on groups of different parameter sets
    param_sets_dict = lambda: defaultdict(param_sets_dict)
    param_sets_groups = param_sets_dict()

    #considering all representative members from 00 to 20
    for pin in range(1,26):

        param_sets_groups[pin] = pd.DataFrame(index=range(120))

        for realization in ens_df_runoff.columns[~ens_df_runoff.columns.isin(['date'])]:

            #take just the realizations corresponding to the same rm
            if str(realization)[8:10] == str('%02d' % pin):
                param_sets_groups[pin][str(realization)] = ens_df_runoff[str(realization)]

        param_sets_groups[pin]['date'] = ens_df_runoff['date']

    return ens_members_groups, param_sets_groups


def quantiles(ens_df):
    
    """
    Calculate the quantiles for the ensemble dataframe considered (e.g. all realizations, or all param sets chosen a rm,...)
    """
    
    #define a dataframe to contain the quantiles
    quantiles = pd.DataFrame()
    columns = ['0.0', '0.1', '0.2', '0.25', '0.3', '0.4', '0.5', '0.6', '0.7', '0.75', '0.8', '0.9', '0.95',
               '0.975', '0.99', '1.0']
    j=0

    #calculate quantiles for every date considering every different simulation run
    for i in ens_df['date']:
        quantiles[j] = mquantiles(ens_df.loc[ens_df['date'] == i].drop('date', axis=1), 
                                  prob=[0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 0.95,
                                        0.975, 0.99, 1.0])
        j = j+1

    #transpose the dataframe
    quantiles = quantiles.T
    quantiles.columns = columns
    quantiles['date'] = ens_df['date']

    return quantiles    


def spaghetti_plot(ens_df_runoff, ens_df_prec, obs_subset, prec_obs_subset, sim_start, past=False, clustered=False, medians=False):

    """
    Produce a spaghetti plot considering a set of the ensemble members: upper precipitation realizations, lower runoff
    realizations, altogether with observations
    Condition on variable "past": if it is False it's for the forecast with precipitation variability,
    if it is True it's for looking at hydro param uncertainty in the past foreast where there is no prec variability
    """
        
    fig, (ax1, ax2) = plt.subplots(2, 2, figsize=(13,8), dpi=100)

    ax1 = plt.subplot2grid((6,1), (0,0), rowspan=2, colspan=1)
    if past == False:
        if clustered == False:
            plt.title('Spaghetti plot for runoff and precipitation realizations for initialization ' + sim_start)
        else: 
            plt.title('Spaghetti plot for clustered (5 RM) runoff and precipitation realizations for initialization ' + sim_start)
    else:
        plt.title('Spaghetti plot for runoff realizations, 5 days before initialization ' + sim_start)
        
    plt.ylabel('Precipitation [mm h$^{-1}$]')
    ax2 = plt.subplot2grid((6,1), (2,0), rowspan=4, colspan=1, sharex=ax1)
    plt.ylabel('Discharge [m$^3$ s$^{-1}$]')
    
    if past == False:
        for member in ens_df_prec.columns[~ens_df_prec.columns.isin(['date'])]:
            prec_member = ax1.plot(ens_df_prec.date, ens_df_prec[member], color='#023FA5', linewidth=0.5)      
    
    l1 = ax1.plot(prec_obs_subset.date, prec_obs_subset['P-kor'], linewidth=2, label='Prec obs', color='red')
    
    ax1.invert_yaxis()
    ax1.grid(True)
    ax1.yaxis.tick_right()
    ax1.yaxis.set_label_position("right")
    ax1.spines["bottom"].set_visible(False)
    
    if past == False:
        #label text box
        if clustered == False:
            prec_label='All ens members'
        else:
            prec_label='Cluster 5 rm'
        ax1.text(0.015, 0.135, prec_label, transform=ax1.transAxes, fontsize=13, 
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='#023FA5', alpha=0.3))
    
    for member in ens_df_runoff.columns[~ens_df_runoff.columns.isin(['date'])]:
        runoff_member = ax2.plot(ens_df_runoff.date, ens_df_runoff[member], color='#32AAB5', linewidth=0.5)  
    l2 = ax2.plot(obs_subset.date, obs_subset.runoff, linewidth=2, label='Runoff obs', color='orange')

    ax2.grid(True)
    ax2.spines["top"].set_visible(False)

    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.subplots_adjust(hspace=0.0)

    #label text box
    if past == False:
        if clustered == False:
            if medians == True: 
                runoff_label='Ens medians'
            else:
                runoff_label='\n'.join((r'All ens realizations',  r'All pin realizations'))
        else:
            if medians == True: 
                runoff_label='Cluster 5 rm medians'
            else:
                runoff_label='\n'.join((r'Cluster 5 rm',  r'All pin realizations'))
    if past == True:
        runoff_label = 'All pin realizations'
   
    ax2.text(0.015, 0.965, runoff_label, transform=ax2.transAxes, fontsize=13,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='#32AAB5', alpha=0.3))
        
    #y axis limits
    #ax2.set_ylim([0,500])
    
    #x axis ticks and limits
    days = mdates.DayLocator()
    hours = mdates.HourLocator()
    yearsFmt = mdates.DateFormatter('%Y-%m-%d') # %H:%M')

    ax2.xaxis.set_major_locator(days)
    ax2.xaxis.set_major_formatter(yearsFmt)
    ax2.xaxis.set_minor_locator(hours)
    # min and max on x axis
    datemin = np.datetime64(ens_df_runoff.date[0], 'm') - np.timedelta64(60, 'm')
    datemax = np.datetime64(ens_df_runoff.date[119], 'm') + np.timedelta64(25, 'm')
    ax2.set_xlim(datemin, datemax)
        
    if past == False:
    
        fig.legend(handles=[prec_member[0], l1[0], runoff_member[0], l2[0]], ncol=2, framealpha=0.5, 
                   loc=(0.5425,0.545), labels=['Prec member', 'Prec obs', 'Runoff member', 'Runoff obs']);
    
    else:
        fig.legend(handles=[l1[0], runoff_member[0], l2[0]], ncol=1, framealpha=0.5, 
                   loc=(0.7425,0.545), labels=['Prec obs', 'Runoff member', 'Runoff obs']);
    

    plt.rcParams.update({'font.size': 12})

    return plt.show()


def hydrograph(quant_runoff, quant_prec, obs_subset, prec_obs_subset, sim_start, past=False, medians=False):
    
    """
    Similar to spaghetti plot but with quantiles values, showing the median, the IQR and the total spread of both
    precipitation and runoff forecast, altogether with observations
    """
    
    #datetime conversion to use plt.fill_between otherwise it would not work with quantiles.date on x axis
    date_conv = [''  for x in range(120)]
    i=0
    for date in quant_prec.date:
        date_conv[i] = parser.parse(str(date))
        i = i+1
    
    fig, (ax1, ax2) = plt.subplots(2, 2, figsize=(13,8), dpi=100)
    
    
       
    ax1 = plt.subplot2grid((6,1), (0,0), rowspan=2, colspan=1)
    if past == False:
        plt.title('Discharge hydrograph and forecast precipitation for initialization ' + sim_start)
    else:
        plt.title('Discharge hydrograph, 5 days before initialization ' + sim_start)
    plt.ylabel('Precipitation [mm h$^{-1}$]')
    ax2 = plt.subplot2grid((6,1), (2,0), rowspan=4, colspan=1, sharex=ax1)
    plt.ylabel('Discharge [m$^3$ s$^{-1}$]')
    
    if past == False:
        ax1.fill_between(date_conv, quant_prec ['0.75'], quant_prec ['0.25'], facecolor='#023FA5', alpha='0.3')
        ax1.fill_between(date_conv, quant_prec ['1.0'], quant_prec ['0.75'], facecolor='#023FA5', alpha='0.5')
        ax1.fill_between(date_conv, quant_prec ['0.25'], quant_prec ['0.0'], facecolor='#023FA5', alpha='0.5')
        l1 = ax1.plot(date_conv, quant_prec ['0.5'], linewidth=2, label='Prec $q_{50\%}$', color='#023FA5', alpha=1)
    
        #label text box
        prec_label='All ens members'
        ax1.text(0.015, 0.135, prec_label, transform=ax1.transAxes, fontsize=13, 
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='#023FA5', alpha=0.3))
    
    l2 = ax1.plot(prec_obs_subset.date, prec_obs_subset['P-kor'], linewidth=2, label='Prec obs', color='red')
    
    ax1.invert_yaxis()
    ax1.grid(True)
    ax1.yaxis.tick_right()
    ax1.yaxis.set_label_position("right")
    ax1.spines["bottom"].set_visible(False)
       
    
    ax2.fill_between(date_conv, quant_runoff ['0.75'], quant_runoff ['0.25'], facecolor='#32AAB5', alpha='0.3')
    ax2.fill_between(date_conv, quant_runoff ['1.0'], quant_runoff ['0.75'], facecolor='#32AAB5', alpha='0.5')
    ax2.fill_between(date_conv, quant_runoff ['0.25'], quant_runoff ['0.0'], facecolor='#32AAB5', alpha='0.5')
    l3 = ax2.plot(date_conv, quant_runoff ['0.5'], linewidth=2, label='Runoff $q_{50\%}$', color='#32AAB5', alpha=1)
    l4 = ax2.plot(obs_subset.date, obs_subset.runoff, linewidth=2, label='Runoff obs', color='orange')
    
    ax2.grid(True)
    ax2.spines["top"].set_visible(False)
    
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.subplots_adjust(hspace=0.0)
    
    if past == False:
        if medians == False:
            runoff_label='\n'.join((r'All ens realizations',  r'All pin realizations'))
        else:
            runoff_label='Ens medians'            
    else:
        runoff_label = 'All pin realizations'
    
    #label text box
    ax2.text(0.015, 0.965, runoff_label, transform=ax2.transAxes, fontsize=13,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='#32AAB5', alpha=0.3))
    
    #y axis limits
    #ax2.set_ylim([0,500])
    
    #x axis ticks and limits
    days = mdates.DayLocator()
    hours = mdates.HourLocator()
    yearsFmt = mdates.DateFormatter('%Y-%m-%d') # %H:%M')
    
    ax2.xaxis.set_major_locator(days)
    ax2.xaxis.set_major_formatter(yearsFmt)
    ax2.xaxis.set_minor_locator(hours)
    # min and max on x axis
    datemin = np.datetime64(quant_runoff.date[0], 'm') - np.timedelta64(60, 'm')
    datemax = np.datetime64(quant_runoff.date[119], 'm') + np.timedelta64(25, 'm')
    ax2.set_xlim(datemin, datemax)
    
    runoff_IQR = mpatches.Patch(color='#32AAB5',alpha=0.3, label='Runoff IQR')
    runoff_spread = mpatches.Patch(color='#32AAB5',alpha=0.5, label='Runoff spread')
                                   
    if past == False:                            
        prec_IQR = mpatches.Patch(color='#023FA5',alpha=0.3, label='Prec IQR')
        prec_spread = mpatches.Patch(color='#023FA5',alpha=0.5, label='Prec spread')
                                   
        
        legend = fig.legend(title='Precipitation                    Runoff', handles=[l1[0], prec_IQR, prec_spread, l2[0], l3[0], 
                                                                                      runoff_IQR, runoff_spread, l4[0]], 
        ncol=2, framealpha=0.5, loc=(0.645,0.526), 
        labels=['       Median $q_{50\%}$', 
                '              IQR', 
                '       Total spread',  
                '       Observation', '', '', '', '']);
                                     
    if past == True:
        fig.legend(handles=[l2[0], l3[0], runoff_IQR, runoff_spread, l4[0]], ncol=1, framealpha=0.5,
                   loc=(0.7,0.5), labels=['Prec obs', 'Runoff $q_{50\%}$', 'Runoff IQR', 'Runoff spread', 'Runoff obs']);
    
    
    plt.rcParams.update({'font.size': 12})
    
    return plt.show()


def hydrograph_rms(rm_high, rm_medium, rm_low, ens_df_prec, quant_rm_groups_runoff, quant_runoff, obs_subset, 
                   prec_obs_subset, sim_start):
   
    
    date_conv = [''  for x in range(120)]
    i=0
    for date in quant_runoff.date:
        date_conv[i] = parser.parse(str(date))
        i = i+1
    
    date_conv_prec = [''  for x in range(len(prec_obs_subset))]
    i=0
    for date in prec_obs_subset.date:
        date_conv_prec[i] = parser.parse(str(date))
        i = i+1
        
    fig, (ax1, ax2) = plt.subplots(2, 2, figsize=(13,8), dpi=100)
    
    ax1 = plt.subplot2grid((6,1), (0,0), rowspan=2, colspan=1)
    plt.title('Discharge hydrograph and forecast precipitation for initialization ' + sim_start)
    plt.ylabel('Precipitation [mm h$^{-1}$]')
    ax2 = plt.subplot2grid((6,1), (2,0), rowspan=4, colspan=1, sharex=ax1)
    plt.ylabel('Discharge [m3 s$^{-1}$]')
           
    ax1.plot(ens_df_prec.date, ens_df_prec[f'rm{str(rm_high).zfill(2)}_pin01'], color='#C94B7C', linewidth=1.5, linestyle='--') 
    ax1.plot(ens_df_prec.date, ens_df_prec[f'rm{str(rm_medium).zfill(2)}_pin01'], color='#848B00', linewidth=1.5, linestyle='--') 
    ax1.plot(ens_df_prec.date, ens_df_prec[f'rm{str(rm_low).zfill(2)}_pin01'], color='#32AAB5', linewidth=1.5, linestyle='--') 
    
    ax1.fill_between(date_conv_prec, prec_obs_subset['P-kor'], 0, facecolor='#023FA5', alpha='0.3')
    
    ax1.invert_yaxis()
    ax1.grid(True)
    ax1.yaxis.tick_right()
    ax1.yaxis.set_label_position("right")
    ax1.spines["bottom"].set_visible(False)
    
    
    ax2.fill_between(date_conv, quant_rm_groups_runoff[rm_high]['0.75'], quant_rm_groups_runoff[rm_high]['0.25'], facecolor='#C94B7C', alpha='0.3')
    ax2.fill_between(date_conv, quant_rm_groups_runoff[rm_high]['1.0'], quant_rm_groups_runoff[rm_high]['0.75'], facecolor='#C94B7C', alpha='0.5')
    ax2.fill_between(date_conv, quant_rm_groups_runoff[rm_high]['0.25'], quant_rm_groups_runoff[rm_high]['0.0'], facecolor='#C94B7C', alpha='0.5')
    l3_1 = ax2.plot(date_conv, quant_rm_groups_runoff[rm_high]['0.5'], linewidth=2, label='Runoff $q_{50\%}$', color='#C94B7C', alpha=1)
    
    ax2.fill_between(date_conv, quant_rm_groups_runoff[rm_medium]['0.75'], quant_rm_groups_runoff[rm_medium]['0.25'], facecolor='#848B00', alpha='0.3')
    ax2.fill_between(date_conv, quant_rm_groups_runoff[rm_medium]['1.0'], quant_rm_groups_runoff[rm_medium]['0.75'], facecolor='#848B00', alpha='0.5')
    ax2.fill_between(date_conv, quant_rm_groups_runoff[rm_medium]['0.25'], quant_rm_groups_runoff[rm_medium]['0.0'], facecolor='#848B00', alpha='0.5')
    l3_2 = ax2.plot(date_conv, quant_rm_groups_runoff[rm_medium]['0.5'], linewidth=2, label='Runoff $q_{50\%}$', color='#848B00', alpha=1)
    
    ax2.fill_between(date_conv, quant_rm_groups_runoff[rm_low]['0.75'], quant_rm_groups_runoff[rm_low]['0.25'], facecolor='#32AAB5', alpha='0.3')
    ax2.fill_between(date_conv, quant_rm_groups_runoff[rm_low]['1.0'], quant_rm_groups_runoff[rm_low]['0.75'], facecolor='#32AAB5', alpha='0.5')
    ax2.fill_between(date_conv, quant_rm_groups_runoff[rm_low]['0.25'], quant_rm_groups_runoff[rm_low]['0.0'], facecolor='#32AAB5', alpha='0.5')
    l3_3 = ax2.plot(date_conv, quant_rm_groups_runoff[rm_low]['0.5'], linewidth=2, label='Runoff $q_{50\%}$', color='#32AAB5', alpha=1)              
                  
    l4 = ax2.plot(obs_subset.date, obs_subset.runoff, linewidth=1.5, label='Runoff obs', color='k')
    
    ax2.grid(True)
    ax2.spines["top"].set_visible(False)
    
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.subplots_adjust(hspace=0.0)
    
    #label text box
    ax2.text(0.015, 0.965, f'rm={rm_high}', transform=ax2.transAxes, fontsize=13,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='#C94B7C', alpha=0.3))
    ax2.text(0.015, 0.875, f'rm={rm_medium}', transform=ax2.transAxes, fontsize=13,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='#848B00', alpha=0.3))
    ax2.text(0.015, 0.785, f'rm={rm_low}', transform=ax2.transAxes, fontsize=13,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='#32AAB5', alpha=0.3))
    
    #y axis limits
    #ax2.set_ylim([0,500])
    
    #x axis ticks and limits
    days = mdates.DayLocator()
    hours = mdates.HourLocator()
    yearsFmt = mdates.DateFormatter('%Y-%m-%d') # %H:%M')
    
    ax2.xaxis.set_major_locator(days)
    ax2.xaxis.set_major_formatter(yearsFmt)
    ax2.xaxis.set_minor_locator(hours)
    # min and max on x axis
    datemin = np.datetime64(quant_runoff.date[0], 'm') - np.timedelta64(60, 'm')
    datemax = np.datetime64(quant_runoff.date[119], 'm') + np.timedelta64(25, 'm')
    ax2.set_xlim(datemin, datemax)
    
    prec_obs = mpatches.Patch(color='#023FA5',alpha=0.3, label='Prec obs')
    runoff_IQR = mpatches.Patch(color='#32AAB5',alpha=0.3, label='Runoff IQR')
    runoff_spread = mpatches.Patch(color='#32AAB5',alpha=0.5, label='Runoff spread')
    fig.legend(handles=[prec_obs, l3_3[0], runoff_IQR, runoff_spread, l4[0]], ncol=1, framealpha=0.5,
               loc=(0.75,0.465), labels=['Prec obs','Runoff $q_{50\%}$', 'Runoff IQR', 'Runoff spread', 'Runoff obs']);
    
    plt.rcParams.update({'font.size': 12})

    return plt.show()


def past_hydro_unc_ensemble_df(df, sim_start, Verzasca_area, variable_type):
    
    """
    Similarly to ensemble_df() it creates a dataframe containing all the different (hydrological) realizations
    but for a period of time comprised in the 5 days before the simulation start, to look at the hydrological
    uncertainty a posteriori (i.e. when meteorological uncertainty is not present because meteorological observations 
    are used in the past while the hydrological parameters can continue to change)
    """
    
    #initialize the dataframe that contains all the realizations for a particular variable
    past_ens_df = pd.DataFrame()
    
    #index to account for the right dates without accounting them more than one time
    j=0
    
    #initialization of array to store the 120 hours dates
    date_array = ["" for x in range(121)]
    
    #condition on the variable chosen to convert discharge in m3/s:
    if (variable_type == 'RTOT') or (variable_type == 'RGES'):
        conv_factor = Verzasca_area/(1000.0*3600.0)
    else:
        conv_factor = 1.0
        
    #5 days before the simulation start:
    index_sim_start = int(df[sim_start]['rm00_pin01']['date'].loc[df[sim_start]['rm00_pin01']['date'] == sim_start].index.values)
    sim_start_minus5days = str(df[sim_start]['rm00_pin01']['date'].loc[df[sim_start]['rm00_pin01']['date'].index == index_sim_start-120])[6:25]
    
    #for cycle on different members/paramsets (pick just the first 25 because all the other are the same, meteo doesnt change)
    for member in list(df[sim_start].keys())[0:25]:
    
        #for cycle on different dates 
        for date in df[sim_start][member]['date']:
    
            #series of if conditions to account for the 120 hours just BEFORE the initialization point and not AFTER
                
            #case if we are on the same month -> must consider month and day 
            if (str(date)[5:7] == sim_start_minus5days[5:7]):
                #case if we are on the same day -> must consider hour
                if (str(date)[8:10] == sim_start_minus5days[8:10]):
                    if (str(date)[11:13] >= sim_start_minus5days[11:13]):
                    
                        #if condition to take just the first set of the next 120 hours without having many copies of them
                        if j >=0 and j <=120:
                            date_array[j] = date
                            j = j+1
                            
                        #condition for precipitation to pick just the ensemble members and not every parameter set, 
                        #since for prec do not change
                        if variable_type == 'P-kor':
                            if member[8:10] == '01':
                                #take the 120 hours in between the sim_start_minus5days and sim_start
                                past_ens_df[member] = df[sim_start][member][variable_type].loc[df[sim_start][member]['date'] > sim_start_minus5days].loc[df[sim_start][member]['date'] <= sim_start]*conv_factor
                        else:
                            past_ens_df[member] = df[sim_start][member][variable_type].loc[df[sim_start][member]['date'] > sim_start_minus5days].loc[df[sim_start][member]['date'] <= sim_start]*conv_factor
                
                if (str(date)[8:10] > sim_start_minus5days[8:10]):
            
                    #if condition to take just the first set of the next 120 hours without having many copies of them
                    if j >=0 and j <=120:
                        date_array[j] = date
                        j = j+1
                            
                    #condition for precipitation to pick just the ensemble members and not every parameter set, 
                    #since for prec do not change
                    if variable_type == 'P-kor':
                        if member[8:10] == '01':
                            past_ens_df[member] = df[sim_start][member][variable_type].loc[df[sim_start][member]['date'] > sim_start_minus5days].loc[df[sim_start][member]['date'] <= sim_start]*conv_factor
                    else:
                        past_ens_df[member] = df[sim_start][member][variable_type].loc[df[sim_start][member]['date'] > sim_start_minus5days].loc[df[sim_start][member]['date'] <= sim_start]*conv_factor
                    
            #case if we are in differen months -> can consider just the month and not the day
            if (str(date)[5:7] > sim_start[5:7]):
                
                #if condition to take just the first set of the next 120 hours without having many copies of them
                if j >=0 and j <=120:
                    date_array[j] = date
                    j = j+1
                    
                #condition for precipitation to pick just the ensemble members and not every parameter set, 
                #since for prec do not change
                if variable_type == 'P-kor':
                    if member[8:10] == '01':
                        past_ens_df[member] = df[sim_start][member][variable_type].loc[df[sim_start][member]['date'] > sim_start_minus5days].loc[df[sim_start][member]['date'] <= sim_start]*conv_factor
                else:
                    past_ens_df[member] = df[sim_start][member][variable_type].loc[df[sim_start][member]['date'] > sim_start_minus5days].loc[df[sim_start][member]['date'] <= sim_start]*conv_factor
    
    past_ens_df['date'] = date_array[1:]
    past_ens_df.index = range(120)
    
    return past_ens_df




def hydro_unc_boxplot(quant_rm_groups_runoff, sim_start, normalized=False):
    
    """
    For every timestep (hour) calculate the spread range q100-q0 and the IQR range q75-q25 for every realization (meteo median) 
    and based on where the (median?) is, place it in the right runoff range, then calculate the boxplot for every range of
    discharge and for the total spread and the IQR
    """
    #decide in which way to split the discharge value along the y-axis:
    runoff_ranges_names = ['0-25', '25-50', '50-75', '75-100', '>100']
    runoff_ranges_values = [25.0, 50.0, 75.0, 100.0]
    #runoff_ranges_names = ['0-50', '50-100', '100-150', '150-200', '>200']
    #runoff_ranges_values = [50.0, 100.0, 150.0, 200.0]

    # Dictionary of dataframes for every ens member look at hydro unc around it
    hydro_unc_dict = lambda: defaultdict(hydro_unc_dict)
    hydro_unc = hydro_unc_dict()
        
    for rm in range(21):
        hydro_unc[rm] = pd.DataFrame(index=range(120*2),columns=runoff_ranges_names)
    
        hydro_unc[rm]['unc_interval'] = np.nan
    
    for rm in range(21):
    
        j=0
        for i in range(120):
            
            #calculate the spread range and IQR range for every time step,
            #choose if normalized with the median value at that point or not:
            if normalized == True :    
                spread_range = (quant_rm_groups_runoff[rm]['1.0'][i] - quant_rm_groups_runoff[rm]['0.0'][i]) / quant_rm_groups_runoff[rm]['0.5'][i]
                IQR_range = (quant_rm_groups_runoff[rm]['0.75'][i] - quant_rm_groups_runoff[rm]['0.25'][i]) / quant_rm_groups_runoff[rm]['0.5'][i]
            
            else :
                spread_range = (quant_rm_groups_runoff[rm]['1.0'][i] - quant_rm_groups_runoff[rm]['0.0'][i])
                IQR_range = (quant_rm_groups_runoff[rm]['0.75'][i] - quant_rm_groups_runoff[rm]['0.25'][i])
                
            #series of if conditions to address in which range we are, look at the median, GOOD APPROACH???
            if (quant_rm_groups_runoff[rm]['0.5'][i] < runoff_ranges_values[0]):
                hydro_unc[rm][runoff_ranges_names[0]][j+1] = spread_range
                hydro_unc[rm][runoff_ranges_names[0]][j] = IQR_range
                
            
            if ((quant_rm_groups_runoff[rm]['0.5'][i] >= runoff_ranges_values[0]) & (quant_rm_groups_runoff[rm]['0.5'][i] < runoff_ranges_values[1]) ) :
                hydro_unc[rm][runoff_ranges_names[1]][j+1] = spread_range
                hydro_unc[rm][runoff_ranges_names[1]][j] = IQR_range
                
            if ((quant_rm_groups_runoff[rm]['0.5'][i] >= runoff_ranges_values[1]) & (quant_rm_groups_runoff[rm]['0.5'][i] < runoff_ranges_values[2]) ) :
                hydro_unc[rm][runoff_ranges_names[2]][j+1] = spread_range
                hydro_unc[rm][runoff_ranges_names[2]][j] = IQR_range
            
            if ((quant_rm_groups_runoff[rm]['0.5'][i] >= runoff_ranges_values[2]) & (quant_rm_groups_runoff[rm]['0.5'][i] <= runoff_ranges_values[3]) ) :
                hydro_unc[rm][runoff_ranges_names[3]][j+1] = spread_range
                hydro_unc[rm][runoff_ranges_names[3]][j] = IQR_range
                
            if (quant_rm_groups_runoff[rm]['0.5'][i] > runoff_ranges_values[3]) :
                hydro_unc[rm][runoff_ranges_names[4]][j+1] = spread_range
                hydro_unc[rm][runoff_ranges_names[4]][j] = IQR_range
            
            hydro_unc[rm]['unc_interval'][j+1] = 'Total spread: q100 - q0'
            hydro_unc[rm]['unc_interval'][j] = 'IQR: q75 - q25'
            
            j=j+2
    
    # Merge all dataframes together
    hydro_unc_tot = pd.concat((hydro_unc[rm] for rm in range(21)), ignore_index=True)
    
    sns.set(style="ticks", palette="pastel")
    
    fig, ax = plt.subplots(1, 1, figsize=(10,7), dpi=100)
    
    plt.title('Hydrological uncertainty obtained for every rm median for initialization ' + sim_start)
    
    melted_hydro_unc = pd.melt(hydro_unc_tot, id_vars=['unc_interval'])
    melted_hydro_unc.value = melted_hydro_unc.value.astype(float)
    
    sns.boxplot(data=melted_hydro_unc,x="value", y='variable', hue='unc_interval', 
                palette=["#E4CBF9", "#9AE1E1"])
    
    
    ax.invert_yaxis()
    
    sns.despine(offset=10, trim=True)
             
    plt.ylabel('Discharge interval [m3 s$^{-1}$]')
    
    if normalized == True :  
        xlabel = 'Normalized spread interval range'
    else : 
        xlabel = 'Spread interval range [m3 s$^{-1}$]'
    
    plt.xlabel(xlabel)
    plt.legend(title='Hydro param spread dispersion', loc='lower right')
    plt.grid()
    return plt.show()



"""
Some basic statistical functions on the forecast realizations:
"""

def IQR(q075, q025):
    return (q075-q025)

def MAD(arr):
    """ Median Absolute Deviation: a "Robust" version of standard deviation.
        Indices variabililty of the sample.
        https://en.wikipedia.org/wiki/Median_absolute_deviation 
    """
    arr = np.ma.array(arr).compressed() # should be faster to not use masked arrays.
    med = np.median(arr)
    return np.median(np.abs(arr - med))

def YuleKendall(q025, q05, q075):
    """ Yule-Kendall index: a robust and resistant alternative to the sample skewness
        If the dataset is right-skewed the Yule-Kendall index is positive
    """
    return ((q025 - 2*q05 + q075)/(q075 - q025))


def stat_quant_df(quantiles, ens_df):
    
    """
    Same statistical functions but on quantiles
    """
    
    quantiles_stat = pd.DataFrame(columns=['IQR', 'MAD', 'YuleKendall'], index=quantiles.index)

    for i in quantiles.index:
        quantiles_stat.IQR[i] = IQR(quantiles['0.75'][i], quantiles['0.25'][i])
        quantiles_stat.MAD[i] = MAD(ens_df.loc[ens_df.index == i].drop('date', axis=1))
        quantiles_stat.YuleKendall[i] = YuleKendall(quantiles['0.25'][i], quantiles['0.5'][i], quantiles['0.75'][i])

    IQR_avg = np.mean(quantiles['0.75']) - np.mean(quantiles['0.25'])
    spread_range_avg = np.mean(quantiles['1.0']) - np.mean(quantiles['0.0'])
    quantiles_stat['date'] = quantiles['date']
    print('The average IQR = <q_75> - <q_25> =  %e' % IQR_avg)
    print('The average range of spread = <q_100> - <q_0> =  %e' % spread_range_avg )

    return quantiles_stat


"""
Forecast verification tools: Brier score, Brier skill score (MUST BE CORRECTED FOR #ENSEMBLE MEMBERS!)
and plots, calculation of POD,FAR and POFD to plot the ROC curve
"""

def BS(ens,obs,y0,lead_time):
    
    #y0: threshold value
    
    #rename obs.index from 0 to 119 (lead time hours)
    obs.index = range(len(obs))
    
    #define obs as binary variable: o=1 if the event occured, o=0 if the event did not occur
    o = obs*0
    
    for k in obs.index[0:lead_time]:
        if obs[k] >= y0[k]:
            o[k] = 1
        else:
            o[k] = 0
    
    j=0
    y = np.zeros(len(ens))
    
    #calculate the yk probability that the event was forecasted, as a probability among all different realizations
    for i in ens.index:
        for column in ens.columns[~ens.columns.isin(['date'])]: #drop the last column of dates
            if ens[column][i] >= y0[i]: #if ensemble value higher than threshold
                j=j+1
        y[i] = j/len(ens.columns) #calculation of probability of threshold exceedance
        j=0
        
    n=len(ens.index)
    
    return y,o,(1/n)*sum((y-o)**2)

def BS_plot(ens_df_runoff, rm_medians, obs_subset, y0, lead_times, plotting=True):
    
    BSs_runoff_tot = pd.DataFrame(index = range(len(lead_times)), columns=['BS', 'lead_time [h]'])
    BSs_runoff_met = pd.DataFrame(index = range(len(lead_times)), columns=['BS', 'lead_time [h]'])
    
    for lead_time in lead_times:
        BSs_runoff_tot['BS'][lead_time/24-1] = BS(ens_df_runoff, obs_subset.runoff, y0, lead_time)[2]
        BSs_runoff_met['BS'][lead_time/24-1] = BS(rm_medians, obs_subset.runoff, y0, lead_time)[2]
        
        BSs_runoff_tot['lead_time [h]'][lead_time/24-1] = lead_time
        BSs_runoff_met['lead_time [h]'][lead_time/24-1] = lead_time
        
    if (plotting == True):
        fig, ax = plt.subplots(1, 1, figsize=(6,4), dpi=100)
        plt.rcParams.update({'font.size': 13})
        
        plt.scatter(BSs_runoff_tot['lead_time [h]'], BSs_runoff_tot['BS'], color='red', label='tot')
        plt.scatter(BSs_runoff_met['lead_time [h]'], BSs_runoff_met['BS'], color='blue', label='met')
        ax.xaxis.set_major_locator(plt.FixedLocator(locs=lead_times))
        
        plt.grid(linestyle='--')
        plt.xlabel('Lead times [h]')
        plt.ylabel('BS');
        plt.legend()
        plt.title('Brier scores for threshold runoff > %i' % int(float(y0[0])) + ' m3 s-1'); #when considering e.g. median y0.name*100
        
    return BSs_runoff_tot, BSs_runoff_met, plt.show()

def BSS(BS,obs,y0):
    
    #reinitialize obs.index from 0 to 119 (lead time hours)        
    obs.index = range(len(obs))
        
    o = obs*0
    
    for k in obs.index:
        if obs[k] >= y0[k]:
            o[k] = 1
        else:
            o[k] = 0
            
    o_avg = np.mean(o)
    
    n=len(obs.index)
    
    BSref = (1/n)*sum((o_avg-o)**2)
    
    return o,o_avg, BSref, 1-BS/BSref

def BSS_plot(BSs_runoff_tot, BSs_runoff_met, obs_subset, y0, lead_times):

    fig, ax = plt.subplots(1, 1, figsize=(6,4), dpi=100)
    plt.rcParams.update({'font.size': 13})
    
    plt.scatter(BSs_runoff_tot['lead_time [h]'], BSS(BSs_runoff_tot.BS, obs_subset.runoff,y0)[3], 
                color='red', label='tot')
    plt.scatter(BSs_runoff_tot['lead_time [h]'], BSS(BSs_runoff_met.BS, obs_subset.runoff,y0)[3], 
                color='blue', label='met')
    #plt.scatter(BSs_runoff_met['lead_time [h]'], BSs_runoff_met['BS'], color='blue', label='met')
    #plt.scatter(BSs_runoff_hyd['lead_time [h]'], BSs_runoff_hyd['BS'], color='green', label='hyd')
    ax.xaxis.set_major_locator(plt.FixedLocator(locs=lead_times))
    
    plt.grid(linestyle='--')
    plt.xlabel('Lead times [h]')
    plt.ylabel('BSS');
    plt.legend()
    plt.title('Brier skill scores for threshold q%i' % int(float(y0[0])) + ' m3 s-1');

    return plt.show()

def POD(realizations,obs,y0,lead_time):
    # #threshold exceedance correctly forecasted:num
    num = 0
    # #threshold exceedance occured
    den = 0
    
    #change index to have both dfs with the same one:
    obs.index = range(len(obs))
    
    for i in obs.index[0:lead_time]:
        if obs[i] >= y0[i]:
            den = den+1
            if realizations[i] >= y0[i]:
                num = num+1
                
    if den == 0:
        den=1
        
    return num/den

def FAR(realizations,obs,y0,lead_time):
    # #false alarms
    num = 0
    # #forecasted threshold exceedances
    den = 0
    
    #change index to have both dfs with the same one:
    obs.index = range(len(obs))
    
    for i in obs.index[0:lead_time]:
        if realizations[i] >= y0[i]:
            den = den+1
            if obs[i] < y0[i]:
                num = num+1
                
    if den == 0:
        den=1
        
    return num/den

def POFD(realizations,obs,y0,lead_time):
    # #false alarms
    num = 0
    # #observed non-events
    den = 0
    
    #change index to have both dfs with the same one:
    obs.index = range(len(obs))
    
    for i in obs.index[0:lead_time]:
        if obs[i] < y0[i]:
            den = den+1
            if realizations[i] >= y0[i]:
                num = num+1
     
    if den == 0:
        den=1
        
    return num/den

def PODs_FARs_POFDs(ens, obs, y0, lead_times, variable='runoff'):
    #create dataframes with #rows given by the number of realizations in ens_df_runoff and #columns given by the lead times
    PODs = pd.DataFrame(index = range(len(ens.columns[~ens.columns.isin(['date'])])),
                        columns=['24','48','72','96','120'])
    FARs = pd.DataFrame(index = range(len(ens.columns[~ens.columns.isin(['date'])])),
                        columns=['24','48','72','96','120'])
    POFDs = pd.DataFrame(index = range(len(ens.columns[~ens.columns.isin(['date'])])),
                        columns=['24','48','72','96','120'])
    
    #different lead times: 1-5 days forecasts
    for lead_time in [24,48,72,96,120]:
        
        #different realizations, from 0 to 525
        for column in ens.columns[~ens.columns.isin(['date'])]:
            PODs[str(lead_time)][ens.columns.get_loc(column)] = POD(ens[column], obs[f'{variable}'], y0, lead_time)
            FARs[str(lead_time)][ens.columns.get_loc(column)] = FAR(ens[column], obs[f'{variable}'], y0, lead_time)
            POFDs[str(lead_time)][ens.columns.get_loc(column)] = POFD(ens[column], obs[f'{variable}'], y0, lead_time)
    
    #sort all of the values in ascending order            
    PODs_sorted = PODs*0.0
    FARs_sorted = PODs*0.0
    POFDs_sorted = PODs*0.0
    for column in PODs.columns:
        PODs_sorted[column] = PODs[column].sort_values(ascending=True).values
        FARs_sorted[column] = FARs[column].sort_values(ascending=True).values
        POFDs_sorted[column] = POFDs[column].sort_values(ascending=True).values
        
    return PODs_sorted, FARs_sorted, POFDs_sorted

def ROC_plot(PODs, FARs_or_POFDs, y0, xlabel, variable='runoff', title_text=''):
    
    fig, ax = plt.subplots(1, 1, figsize=(7,7))
    plt.rcParams.update({'font.size': 14})
    
    jet= plt.get_cmap('rainbow')
    colors = iter(jet(np.linspace(0,1,5)))
    
    for column in PODs.columns:
        color=next(colors)
        
        #xx,yy to calculate ROCa
        xx = np.concatenate((np.array([0.0]), FARs_or_POFDs[column].values,  np.array([1.0])))
        yy = np.concatenate((np.array([0.0]), PODs[column].values,  np.array([1.0])))
        
        ax.plot([0,FARs_or_POFDs[column][0]],[0,PODs[column][0]], '.-', lw=2, markersize=8, color=color)
        ax.plot(FARs_or_POFDs[column],PODs[column],'.-', lw=2, markersize=8, 
                 color=color, label=(f'{int(column)} h      {auc(xx,yy):.4}'))
        
        ax.plot([FARs_or_POFDs[column][-1:],1],[PODs[column][-1:],1], '.-', lw=2, markersize=8, color=color)
    
    ax.hlines(y=1.0, xmin=-0.05, xmax=1, linewidth=1, color='black', linestyle='--')
    ax.plot([0,1],[0,1], linewidth=1, color='black', linestyle='--')
    
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    
    plt.xlim(-0.025,1.025)
    plt.ylim(-0.025,1.025)
    plt.xlabel(xlabel)
    plt.ylabel('POD')
    plt.legend(title='Lead times - ROCa',fontsize=13, loc='lower right', frameon=True)
    
    if variable == 'precipitation':
        units = 'mm'
    else:
        units = 'm3 s-1'
    
    plt.title(f'ROC curve ' + title_text + f'for {variable} threshold > {int(float(y0[0]))} {units}'); #when considering e.g. median y0.name*100

    return plt.show()

def rank_histogram(ens, obs, title_text, realizations_size=525, ranks_number=21, member_extracted=(9,11)):
    
    """
    Plot the verification rank histogram to check on forecast consistency
    """
    
    obs.index = range(len(obs))

    #condition on ranks_number: if 25 we are considering pin members that goes [1:25], otherwise ens members that goes
    #[0:20] for medians or [0:525] considering all possible realizations
    if ranks_number == 25:
        df_rank = pd.DataFrame(index = range(1,ranks_number+1), columns=['rank','pin_member'])
    else:
        df_rank = pd.DataFrame(index = range(ranks_number), columns=['rank','ens_member'])

    #initialize ranks with all zeros
    df_rank['rank'][:] = 0

    for i in obs.index:

        #consider all possible ensemble realizations and obs at the same time
        df = ens.loc[i]

        #merge obs value to dataframe
        df.loc['obs'] = obs.loc[i]

        #sort all values ascendingly
        df = df.sort_values(ascending=True)

        #create new dataframe with new index (range(0,526)) and with a column with ensemble names and obs
        members = df.index
        new_index = range(len(df))
        df = pd.DataFrame(df)
        df['members'] = members
        df.index = new_index

        #extract obs row in dataframe
        obs_merged = df.loc[df['members'] == 'obs']

        #if conditions to account for cases when obs is at the beginning or end of df
        if (obs_merged.index == 0):
            nearest = df.loc[obs_merged.index+1]

        if (obs_merged.index == realizations_size):
            nearest = df.loc[obs_merged.index-1]

        elif ((obs_merged.index != 0) and (obs_merged.index != realizations_size)):
            #select the two nearest element to obs (general case)
            obs_near = df.loc[df.loc[df['members'] == 'obs'].index-1 | df.loc[df['members'] == 'obs'].index | 
                          df.loc[df['members'] == 'obs'].index+1]
            nearest = obs_near.iloc[(obs_near[i]-obs_near[i].loc[df['members'] == 'obs']).abs().argsort()[:1]]

        #extract ensemble member from nearest i.e. # bin associated to histogram
        rank_point=int(str(nearest['members'])[member_extracted[0]:member_extracted[1]]) #[9:11] for 525 realizations

        #add the rank point to the correspondent element in df rank
        df_rank['rank'][rank_point] = df_rank['rank'][rank_point] + 1 

    if ranks_number == 25:
        df_rank['pin_member'] = range(1,ranks_number+1)
        ens_or_pin_column = df_rank['pin_member']
    else:
        df_rank['ens_member'] = range(ranks_number)
        ens_or_pin_column = df_rank['ens_member']

    #plotting the histogram:
    fig, ax = plt.subplots(1, 1, figsize=(7,4), dpi=100)
    plt.rcParams.update({'font.size': 13})

    plt.bar(ens_or_pin_column, df_rank['rank']/120, width=0.95);
    ax.xaxis.set_major_locator(plt.FixedLocator(locs=ens_or_pin_column))
    ax.tick_params(axis='both', labelsize=10)

    plt.ylabel('Frequency')
    plt.xlabel('Ensemble member');
    plt.title('Verification rank histogram'+title_text);
    
    return df_rank, plt.show()


def correlation_plot(y0, obs_subset, lead_times, title_text):

    """
    Plot the correlation between e.g. the median of the realizations and the observation at different lead times, 
    report in the legend the values of r2 for every different lead time considered 
    """
    
    plt.figure(figsize=(8,8))
    plt.rcParams.update({'font.size': 15})

    #set of colors for different lead times
    colors = ['#242989','#445BA6','#658BCF','#87BAFB','#A8E9FF']
    ncolor = 0

    obs_subset.index = range(len(obs_subset))
    
    for lead_time in lead_times:
        #compute the fit between obs and forecast
        X = obs_subset['runoff'][0:lead_time]
        y = y0[0:lead_time]

        X = sm.add_constant(X)
        model = sm.OLS(y, X)
        results = model.fit()
        for k in obs_subset.index[0:lead_time]:

            plt.plot(obs_subset['runoff'][k], y0[k], 'o',markersize=10, color=colors[ncolor],
                     alpha = 1, zorder=1/(ncolor+1), label='%i' % lead_time + ' h, R$^2$ = %f' % (results.rsquared))
        ncolor+=1
    plt.plot([-10, max(obs_subset.runoff+10)],[-10,max(y0)+10], linewidth=1, color='black', 
             linestyle='--')

    plt.xlabel('Observed runoff [m3 s$^{-1}$]')
    plt.ylabel('Forecast median runoff [m3 s$^{-1}$]');
    plt.xlim(-5, max(obs_subset.runoff+10))
    plt.ylim(-5, max(y0+10));

    #print legend without repetions of labels
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='lower right', numpoints = 1);
    plt.title('Correlation plot'+title_text)
    
    return(plt.show())
    
    
def peak_box(ens, obs, all_dates_hours, sim_start, title_text='all realizations', A=186):

    """
    Plot the peak-box approach for the group of runoff realizations considered together with observation:
    find the peak for every realization in the entire temporal domain, find out the first and the last one happening in time
    and the ones with highest and lowest magnitudes, plot the peak-box, find the IQR box from all the peaks and timings
    and plot it, find the peak and timing medians and plot it.
    Calculate the full and IQR sharpness of the forecasts and the deviations of observation peak from the peak represented
    by peak and timing median.    
    """
    
    fig, ax = plt.subplots(1, 1, figsize=(10,6), dpi=100)

    plt.title('Peak-box approach for '+title_text+ ' for initialization ' + sim_start)

    #dataframe containing the values of peak discharges for every realization
    df_max_runoff = pd.DataFrame(index=(ens.columns[~ens.columns.isin(['date'])]),
                                 columns=['max', 'date', 'hour'])

    for member in ens.columns[~ens.columns.isin(['date'])]:

        # Find all the local peak maximums for every realization, excluding borders of x domain (hour 0 and hour 120)
        peaks = find_peaks(ens[member][1:-1], height=0)
        
        # Select the maximum value of the peaks found and find its date and hour associated
        df_max_runoff['max'][member] = max(peaks[1]['peak_heights'])
        df_max_runoff['date'][member] = ens['date'][1:-1].loc[ens[member][1:-1] == df_max_runoff['max'][member]].iloc[0]
        df_max_runoff['hour'][member] = int(ens.loc[ens['date'] == df_max_runoff['date'][member]].index.values) 
        
        ax.plot(ens.date, ens[member], color='#32AAB5', linewidth=0.5)
        ax.plot(df_max_runoff.date[member], df_max_runoff['max'][member], 'o',markersize=5, color='blue', alpha=0.15,
               zorder=1000)

    #observation
    l2 = ax.plot(obs.date, obs.runoff, linewidth=2, label='Runoff obs', color='orange')
    
    #observation peak:
    # Find all the local peak maximums for obs, excluding borders of x domain (hour 0 and hour 120)
    peaks_obs = find_peaks(obs.runoff[1:-1], height=0)
    max_peak = max(peaks_obs[1]['peak_heights'])
    
    l3 = ax.plot(obs.date.loc[obs.runoff == max_peak], max_peak, 'o', markersize=8, color='red', 
                 alpha=0.8, zorder=1001, label='($t_{obs}$, $p_{obs}$)')

    #report all peak and timing(hour) and correspondent dates quantiles in a dataframe
    peaks_timings = pd.DataFrame(index=range(5), columns=['peak', 'timing', 'date'])
    peaks_timings['peak'] = mquantiles(df_max_runoff['max'], prob=[0.0,0.25,0.5,0.75,1.0])
    peaks_timings['timing'] = mquantiles(df_max_runoff.hour, prob=[0.0,0.25,0.5,0.75,1.0]).astype(int)
    for i in range(5):
        peaks_timings['date'][i] = str(all_dates_hours['date'].loc[all_dates_hours['hour'] == 
                                                                   peaks_timings['timing'][i]].iloc[0])
    """
    Peak-Box (outer rectangle):
    """

    #the lower left coordinate set to the earliest time when a peak flow occurred in the available ensemble members (t0) 
    #and the lowest peak discharge of all members during the whole forecast period (p0)
    lower_left_pb = [peaks_timings['date'][0], peaks_timings['peak'][0]]

    #upper right coordinate set to the latest time when a peak flow occurred in the available ensemble members (t100) 
    #and the highest peak discharge of all members during the whole forecast period (p100)
    upper_right_pb =  [peaks_timings['date'][4], peaks_timings['peak'][4]]

    alpha=0.5
    color='blue'
    lw=2

    plt.plot([pd.to_datetime(lower_left_pb[0]), pd.to_datetime(lower_left_pb[0])],
                [lower_left_pb[1], upper_right_pb[1]], color=color, alpha=alpha, lw=lw)

    plt.plot([pd.to_datetime(lower_left_pb[0]), pd.to_datetime(upper_right_pb[0])],
                [lower_left_pb[1], lower_left_pb[1]], color=color, alpha=alpha, lw=lw)

    plt.plot([pd.to_datetime(upper_right_pb[0]), pd.to_datetime(upper_right_pb[0])],
                [lower_left_pb[1], upper_right_pb[1]], color=color, alpha=alpha, lw=lw)

    plt.plot([pd.to_datetime(lower_left_pb[0]), pd.to_datetime(upper_right_pb[0])],
                [upper_right_pb[1], upper_right_pb[1]], color=color, alpha=alpha, lw=lw)

    """
    IQR-box (inner rectangle):
    """
   
    #lower left coordinate set to the 25% quartile of the peak timing (t25) 
    #and the 25% quartile of the peak discharges of all members during the whole forecast period (p25)
    lower_left_IQRbox = [peaks_timings['date'][1], peaks_timings['peak'][1]]

    #[str(all_dates_hours['date'].loc[all_dates_hours['hour'] == int(df_max_quantiles_timing[1])].iloc[0]),
                    #mquantiles(df_max_runoff['max'], prob=[0.0,0.25,0.5,0.75,1.0])[1]]

    #upper right coordinate of the IQR-Box is defined as the 75% quartile of the peak timing (t75) 
    #and the 75% quartile of the peak discharges of all members (p75)
    upper_right_IQRbox = [peaks_timings['date'][3], peaks_timings['peak'][3]]

    #[str(all_dates_hours['date'].loc[all_dates_hours['hour'] == int(df_max_quantiles_timing[3])].iloc[0]),
    #                      mquantiles(df_max_runoff['max'], prob=[0.0,0.25,0.5,0.75,1.0])[3]]




    plt.plot([pd.to_datetime(lower_left_IQRbox[0]), pd.to_datetime(lower_left_IQRbox[0])],
                [lower_left_IQRbox[1], upper_right_IQRbox[1]], color=color, alpha=alpha, lw=lw)

    plt.plot([pd.to_datetime(lower_left_IQRbox[0]), pd.to_datetime(upper_right_IQRbox[0])],
                [lower_left_IQRbox[1], lower_left_IQRbox[1]], color=color, alpha=alpha, lw=lw)

    plt.plot([pd.to_datetime(upper_right_IQRbox[0]), pd.to_datetime(upper_right_IQRbox[0])],
                [lower_left_IQRbox[1], upper_right_IQRbox[1]], color=color, alpha=alpha, lw=lw)

    plt.plot([pd.to_datetime(lower_left_IQRbox[0]), pd.to_datetime(upper_right_IQRbox[0])],
                [upper_right_IQRbox[1], upper_right_IQRbox[1]], color=color, alpha=alpha, lw=lw)

    """
    Median of the peak discharge:
    """
    #horizontal line going from t0 to t100 representing the median of the peak discharge (p50) 
    #of all members of the ensemble forecast
    plt.plot([pd.to_datetime(lower_left_pb[0]), pd.to_datetime(upper_right_pb[0])],
                [peaks_timings['peak'][2], peaks_timings['peak'][2]], color=color, alpha=alpha, lw=lw)

    """
    Median of the peak timing:
    """
    #vertical line going from p0 to p100 representing the median of the peak timing (t50)
    plt.plot([pd.to_datetime(peaks_timings['date'][2]), pd.to_datetime(peaks_timings['date'][2])],
                [lower_left_pb[1], upper_right_pb[1]], color=color, alpha=alpha, lw=lw)

    ax.grid(True)

    #y axis limits
    #ax.set_ylim([0,500])

    #x axis ticks and limits
    days = mdates.DayLocator()
    hours = mdates.HourLocator()
    yearsFmt = mdates.DateFormatter('%Y-%m-%d') # %H:%M')

    ax.xaxis.set_major_locator(days)
    ax.xaxis.set_major_formatter(yearsFmt)
    ax.xaxis.set_minor_locator(hours)
    # min and max on x axis
    datemin = np.datetime64(ens.date[0], 'm') - np.timedelta64(60, 'm')
    datemax = np.datetime64(ens.date[119], 'm') + np.timedelta64(25, 'm')
    ax.set_xlim(datemin, datemax)

    runoff_member = ax.plot(ens.date, ens[ens.columns[0]], color='#32AAB5', 
                             linewidth=0.5, label='Runoff member')
    peak_member = ax.plot(df_max_runoff.date[member], df_max_runoff['max'][member], 'o',markersize=5, color='blue', 
                          alpha=0.3, zorder=1000, label='($t_i$, $p_i$), $i \in $#realizations ')
    median_peak = ax.plot(ens.date.loc[ens.date == peaks_timings['date'][2]], 
                          peaks_timings['peak'][2], '*', markersize=15, color='red', 
                          alpha=10, zorder=1001, label='($t_{50}$, $p_{50}$)')
    
    fig.legend(handles=[runoff_member[0], l2[0], peak_member[0], median_peak[0], l3[0]], ncol=1, numpoints = 1,
               labels=['Runoff member', 'Runoff obs', '($t_i$, $p_i$), $i \in $#realizations', '($t_{50}$, $p_{50}$)',
                       '($t_{obs}$, $p_{obs}$)'], loc=(0.66,0.66));

    plt.rcParams.update({'font.size': 10});

    """
    Sharpness of the forecast:
    PB_full = (p100-p0)(t100-t0)*3.6/A  with A the area of the basin in km2
    PB_IQR = (p75-p25)(t75-t25)*3.6/A
    """

    PB_full = ((peaks_timings['peak'][4] - peaks_timings['peak'][0])*
               (peaks_timings['timing'][4] - peaks_timings['timing'][0])*3.6/A)

    PB_IQR = ((peaks_timings['peak'][3] - peaks_timings['peak'][1])*
               (peaks_timings['timing'][3] - peaks_timings['timing'][1])*3.6/A)
    
    """
    Verification of peak median vs obs:
    Dpeak = |p50-pobs|
    Dtime = |t50-tobs|
    """
    
    D_peak = abs(peaks_timings['peak'][2] - max(obs.runoff))
    
    D_time = abs(peaks_timings['timing'][2] - int(obs.runoff.loc[obs.runoff == max(obs.runoff)].index.values))
    
    return (plt.show(), print(f'PBfull = {PB_full:.5} mm'), print(f'PBiqr = {PB_IQR:.5} mm'), 
            print(f'Dpeak = {D_peak:.5} m3 s-1'), print(f'Dtime = {D_time} h'))
    
    
    
    
def peak_box_multipeaks(ens, obs, all_dates_hours, sim_start, title_text='all realizations', delta_t=10, gamma=3.0/5.0, 
                        decreas_hours=8, beta = 1.0/4.0, A=186):
    
    """
    Plot the peak-box approach for the group of runoff realizations considered together with observation:
    find the peak for every realization in the entire temporal domain, find out the first and the last one happening in time
    and the ones with highest and lowest magnitudes, plot the peak-box, find the IQR box from all the peaks and timings
    and plot it, find the peak and timing medians and plot it.
    Calculate the full and IQR sharpness of the forecasts and the deviations of observation peak from the peak represented
    by peak and timing median.    
    """
    
    """
    MULTI-PEAKS APPROACH: deeveloped for sim_start = '2018-10-27 12:00:00'
        
    1 - Know all the peaks presented by a realization
    2 - Decide a criteria to consider just "relevant" peaks i.e. peaks that can be associated to different events
    3 - Based on the remained peaks regroup them considering all realizations and procced in  drawing the boxes
    
    """
    
    # Implement the division of peaks for all realizations considered:
    
    # dictionary to contain all the event peaks for different ens members
    
    peaks_dict = lambda: defaultdict(peaks_dict)
    event_peaks_dict = peaks_dict()
    
    # dictionary to contain the decrescency boolean values for every realization
    
    decreas_dict = lambda: defaultdict(decreas_dict)
    decreas = decreas_dict()
    
    count_trues_array = np.zeros(120)
    
    for member in ens.columns[~ens.columns.isin(['date'])]:
        
        #Find all the local peak maximums, excluding borders of x domain (hour 0 and hour 119)
        peaks = find_peaks(ens[member][1:-1], height=0)
        peak_date = pd.DataFrame(index=range(len(peaks[1]['peak_heights'])), columns=['date'])
        
        for p in range(len(peaks[1]['peak_heights'])):
            peak_date['date'][p] = ens['date'][1:-1].loc[ens[member][1:-1] == peaks[1]['peak_heights'][p]].iloc[0]
    
        # DECIDE A CRITERIA TO KEEP JUST THE "IMPORTANT" PEAKS:
        # must take into consideration the behaviour of the function i.e. how much it increases/decreases between two peaks
        # and also the amount of time to consider to distinguish between different events
        
        # empty dataframe to contain so-called event peaks i.e. the relatively important peaks associated to events
        event_peaks = pd.DataFrame(index=range(120),columns=['hour','date', 'peak'])
        
        # delta timing to consider the behaviour of the realization: consider the previous and the next delta_t hours to keep a peak or not,
        # if in the next delta_t hours the function decreases at least 2/5 (i.e. 1-gamma) since the peak value -> keep the peak
        
        #delta_t = 10 #hours
        #gamma = 3.0/5.0
        
        # look if the amount of discharge decreases at least 1-gamma after the peak value before increasing again:
        
        n_event = 0
        
        for p in range(len(peaks[1]['peak_heights'])):
            for k in range(-delta_t, delta_t):
                
                # if conditions: - must not go beyond the 120 hours limit and before the beginning at 0 hours, 
                #                - the function must decrease after the peak
                #                - at least one of the delta_t point after/before the peak must be lower of 1-gamma: 2/5? (1/3 ? must tune for the 
                #                  right number) the value of the peak
                
                if (peaks[0][p]+k > 0) and (peaks[0][p]+k < 120) and (ens[member][peaks[0][p]+2] < ens[member][peaks[0][p]+1]) and (ens[member][peaks[0][p]+k] < ens[member][peaks[0][p]+1]*gamma) :
                    
                    event_peaks['hour'][n_event] = peaks[0][p]+1
                    event_peaks['date'][n_event] = ens['date'][1:-1].loc[ens[1:-1].index == event_peaks['hour'][n_event]].iloc[0]
                    event_peaks['peak'][n_event] = ens[member][peaks[0][p]+1]
                    n_event = n_event+1
                    break
                
        #keep just the rows with peaks
        event_peaks = event_peaks[pd.notnull(event_peaks['peak'])]
            
        # for loop to keep just one peak if other peaks are very near (+- 7 hours?):
        
        while True:
            
            # "save" old index to compare it with the new one at the end when some peak are withdrawn
            old_event_peaks_index = event_peaks.index
            
            for i,j in zip(event_peaks.index, event_peaks.index+1):
                
                # conditions to avoid problems when considering the last peak of the domain
                
                if (i == event_peaks.index[-1] + 1) or (j == event_peaks.index[-1] + 1):
                    break
                
                #condition to discard very near in time peaks with very similar values:
                
                if (event_peaks.hour[i] >= event_peaks.hour[j] - 7): #or (event_peaks.hour[i] <= event_peaks.hour[j] + 4):
                    
                    # condition to keep the highest peak between the two near peaks considered:
                    
                    if event_peaks['peak'][j] > event_peaks['peak'][i]:
                        event_peaks = event_peaks.drop(event_peaks.index[i])
                    
                    elif event_peaks['peak'][j] < event_peaks['peak'][i]:
                        event_peaks = event_peaks.drop(event_peaks.index[j])
                        
                    event_peaks.index = range(len(event_peaks))
            
            # if condition to keep the length of the index correct: if old index and new index lengths are equal exit the while loop
            if len(old_event_peaks_index) == len(event_peaks.index):
                break
        
        # write all the event peaks obtained in a dictionary for different members:
        event_peaks_dict[member] = event_peaks
        
            
        # NOW: must seek a criteria to split all the peaks found by groups related to different runoff maxima events.
        # 1st approach: look if the majority of the realizations decrease altogether in a certain temporal window:
        #               for every realizations check if for every hour timestep the next 10? hours (=decreas_hour) decreases from that value
        #               then check every realization for every hour timestep (120x525 matrix) and if for a specific timestep
        #               at least 2/3? of the realizations show decreasing behaviour split the domain among peaks
           
        decreas[member] = np.array(range(120), dtype=bool)
        #decreas_hours = 8
        for h in range(120):
            if all(x > y for x, y in zip(ens[member][h:h+decreas_hours], ens[member][h+1:h+decreas_hours+1])):
                decreas[member][h] = True
            else:
                decreas[member][h] = False
        
        #count for every hour the amount of Trues i.e. how many realizations show a decreasing behaviour for the next decreas_hours
        for h in range(120):
            if decreas[member][h] == True:
                count_trues_array[h] = count_trues_array[h] + 1 
        
        
        
    peak_groups_dict = lambda: defaultdict(peak_groups_dict)         
    peak_groups = peak_groups_dict()
    
    #initialize the splitting_hour list with the first element given by the 0th hour (i.e. the first group start from the beginning of the
    #time domain)
    splitting_hour = []
    splitting_hour.append(0)
    
    #decreasing parameter: the amount of realizations that show the decreasing behaviour
    #beta = 1.0/4.0
    
    for h in range(120):
        
        # condition to divide all the peaks in different groups: 
        # if at least beta of all realizations after a peak are decreasing for at least decreas_hours -> splitting
        if count_trues_array[h] >= len(ens.columns[~ens.columns.isin(['date'])])*beta :
            
            # add the splitting hour found to the splitting_hour list
            splitting_hour.append(h)
            
            # write in peak_groups dictionary all the peaks for every different realizations that stay between two splitting hours
            for member in ens.columns[~ens.columns.isin(['date'])]:
                for peak_hour in event_peaks_dict[member]['hour']:
                    if peak_hour <= splitting_hour[-1]:
                        peak_groups[splitting_hour[-1]][member] = event_peaks_dict[member].loc[(event_peaks_dict[member]['hour'] > splitting_hour[-2]) & 
                                   (event_peaks_dict[member]['hour'] < splitting_hour[-1])]
    
    # conditions to drop all the empty groups from peak_groups (must check if correct!):
                        
    # if all the dataframes of one group are empty -> delete group
    for group in list(peak_groups):
        if all(peak_groups[group][member].empty for member in peak_groups[group].keys()):
            
            #remove empty groups
            peak_groups.pop(group)
            
    # if more than 8.5/10 (15%) of the dataframes of a group are empty -> remove group???
    for group in list(peak_groups):
        empty_dataframes = 0
        for member in peak_groups[group].keys():
            if peak_groups[group][member].empty :
                empty_dataframes = empty_dataframes + 1
        if (empty_dataframes >= len(peak_groups[group].keys())*8.5/10.0):
            peak_groups.pop(group)
            
    # if in a group an element is not a dataframe (dont know why?!) remove that element:
    for group in list(peak_groups):
        for member in peak_groups[group].keys():    
            if (isinstance(peak_groups[group][member], pd.DataFrame) == False) : 
                peak_groups[group].pop(member)
    
    
    # OBSERVATION PEAKS:
    # apply the same procedure as before to distinguish peaks related to different events:
    
    #reset obs index    
    obs = obs.reset_index()

    #Find all the local peak maximums for obs, excluding borders of x domain (hour 0 and hour 120)
    OBSpeaks = find_peaks(obs.runoff[1:-1], height=0)
    OBSpeak_date = pd.DataFrame(index=range(len(OBSpeaks[1]['peak_heights'])), columns=['date'])
    
    for p in range(len(OBSpeaks[1]['peak_heights'])):
        OBSpeak_date['date'][p] = obs['date'][1:-1].loc[obs['runoff'][1:-1] == OBSpeaks[1]['peak_heights'][p]].iloc[0]
    
    # empty dataframe to contain so-called event peaks i.e. the relatively important peaks associated to events
    OBSevent_peaks = pd.DataFrame(index=range(120),columns=['hour','date', 'peak'])
    
    # delta timing to consider the behaviour of the realization: consider the previous and the next delta_t hours to keep a peak or not,
    # if in the next delta_t hours the function decreases at least 1/3 since the peak value -> keep the peak
    
    #delta_t = 10 #hours
    
    # look if the amount of discharge decreases at least 1/3 after the peak value before increasing again:
    
    n_event = 0
    
    for p in range(len(OBSpeaks[1]['peak_heights'])):
        for k in range(-delta_t, delta_t):
            
            # if conditions: - must not go beyond the 120 hours limit and before the beginning at 0 hours, 
            #                - the function must decrease after the peak
            #                - at least one of the delta_t point after the peak must be lower of 2/5 (1/3 ? must tune for the 
            #                  right number) the value of the peak
            
            if (OBSpeaks[0][p]+k > 0) and (OBSpeaks[0][p]+k < 120) and (obs.runoff[OBSpeaks[0][p]+2] < obs.runoff[OBSpeaks[0][p]+1]) and (obs.runoff[OBSpeaks[0][p]+k] < obs.runoff[OBSpeaks[0][p]+1]*gamma) :
                #print(p)
                OBSevent_peaks['hour'][n_event] = OBSpeaks[0][p]+1
                OBSevent_peaks['date'][n_event] = obs['date'][1:-1].loc[ens[1:-1].index == OBSevent_peaks['hour'][n_event]].iloc[0]
                OBSevent_peaks['peak'][n_event] = obs.runoff[OBSpeaks[0][p]+1]
                n_event = n_event+1
                break
            
    #keep just the rows with peaks
    OBSevent_peaks = OBSevent_peaks[pd.notnull(OBSevent_peaks['peak'])]
        
    # for loop to keep just one peak if other peaks are very near (+- 7 hours?):
    
    while True:
        
        # "save" old index to compare it with the new one at the end when some peak are withdrawn
        OBSold_event_peaks_index = OBSevent_peaks.index
        
        for i,j in zip(OBSevent_peaks.index, OBSevent_peaks.index+1):
            
            # conditions to avoid problems when considering the last peak of the domain
            
            if (i == OBSevent_peaks.index[-1] + 1) or (j == OBSevent_peaks.index[-1] + 1):
                break
            
            #condition to discard very near in time peaks with very similar values:
            
            if (OBSevent_peaks.hour[i] >= OBSevent_peaks.hour[j] - 7): #or (event_peaks.hour[i] <= event_peaks.hour[j] + 4):
                
                # condition to keep the highest peak between the two near peaks considered:
                
                if OBSevent_peaks['peak'][j] > OBSevent_peaks['peak'][i]:
                    OBSevent_peaks = OBSevent_peaks.drop(OBSevent_peaks.index[i])
                
                elif OBSevent_peaks['peak'][j] < OBSevent_peaks['peak'][i]:
                    OBSevent_peaks = OBSevent_peaks.drop(OBSevent_peaks.index[j])
                    
                OBSevent_peaks.index = range(len(OBSevent_peaks))
        
        # if condition to keep the length of the index correct: if old index and new index lengths are equal exit the while loop
        if len(OBSold_event_peaks_index) == len(OBSevent_peaks.index):
            break
    
    
    
    # PLOT:
    
    # plot all peaks in different groups 
    #jet= plt.get_cmap('tab10')
    #colors = iter(jet(np.linspace(0,len(peak_groups.keys()),5)))
    colors = itertools.cycle(["#e60000", "#0000e6", "#e6e600", "#bf00ff", "#009933", "#b35900"])
    
    fig, ax = plt.subplots(1, 1, figsize=(10,6), dpi=100)
    
    plt.title('Peak-box approach for initialization ' + sim_start)
    
    for member in ens.columns[~ens.columns.isin(['date'])]:
        runoff_member = ax.plot(ens.date, ens[member], color='#32AAB5', linewidth=0.5, alpha=0.65)
    for group in peak_groups.keys():
        color = next(colors)
        for member in peak_groups[group].keys():    
            peak_member = ax.plot(peak_groups[group][member]['date'], peak_groups[group][member]['peak'],'o',markersize=2, color=color, 
                                  alpha=0.5, zorder=10)
    #observation series plot
    l2 = ax.plot(obs.date, obs.runoff, linewidth=2, label='Runoff obs', color='orange', zorder = 15)
    #observation peaks plot
    for OBSpeak in OBSevent_peaks.index:
        peak_obs = ax.plot(OBSevent_peaks['date'][OBSpeak], OBSevent_peaks['peak'][OBSpeak],'*',markersize=20, color='orange', 
                           markeredgecolor='black', markeredgewidth=1.5, alpha=1, zorder=100)
    
    # NOW: develop peak boxes for every different group:
    
    
    """
    Peak-Box (outer rectangle):
    IQR-box (inner rectangle):
    Median of the peak discharge:
    Median of the peak timing:
    """
    
    #lower_left_pb = pd.DataFrame(index=range(len(peak_groups.keys())))
    #upper_right_pb = pd.DataFrame(index=range(len(peak_groups.keys())))
    
    peak_legend = pd.DataFrame(index=range(len(peak_groups.keys())))
    median_legend = pd.DataFrame(index=range(len(peak_groups.keys())))
    
    #jet= plt.get_cmap('tab10')
    colors = itertools.cycle(["#e60000", "#0000e6", "#e6e600", "#bf00ff", "#009933", "#b35900"])#iter(jet(np.linspace(0,len(peak_groups.keys()),5)))
    
    for group in peak_groups.keys():
        
        color = next(colors)
        
        # empty arrays to contain all the dates/peaks for every different realization of one specific group
        all_dates_of_group = []
        all_hours_of_group = []
        all_peaks_of_group = []
        
        # write all dates, hours and peaks for every possible realizations for every group in peak_groups
        for member in peak_groups[group].keys():
             for date in peak_groups[group][member]['date']:
                 all_dates_of_group.append(str(date))
             for peak in peak_groups[group][member]['peak']:
                 all_peaks_of_group.append(peak)
             for hour in peak_groups[group][member]['hour']:
                 all_hours_of_group.append(hour)
        
        # PEAK-BOX:
        
        #the lower left coordinate set to the earliest time when a peak flow occurred in the available ensemble members (t0) 
        #and the lowest peak discharge of all members during the whole forecast period (p0)
        lower_left_pb = [min(all_dates_of_group), min(all_peaks_of_group)]
            
        #upper right coordinate set to the latest time when a peak flow occurred in the available ensemble members (t100) 
        #and the highest peak discharge of all members during the whole forecast period (p100)
        upper_right_pb =  [max(all_dates_of_group), max(all_peaks_of_group)]
    
        #plot the peak-boxes
        alpha=0.75
        lw=2
        zorder = 20
        
        plt.plot([pd.to_datetime(lower_left_pb[0]), pd.to_datetime(lower_left_pb[0])],
                    [lower_left_pb[1], upper_right_pb[1]], color=color, alpha=alpha, lw=lw, zorder=zorder)
        
        plt.plot([pd.to_datetime(lower_left_pb[0]), pd.to_datetime(upper_right_pb[0])],
                    [lower_left_pb[1], lower_left_pb[1]], color=color, alpha=alpha, lw=lw, zorder=zorder)
        
        plt.plot([pd.to_datetime(upper_right_pb[0]), pd.to_datetime(upper_right_pb[0])],
                    [lower_left_pb[1], upper_right_pb[1]], color=color, alpha=alpha, lw=lw, zorder=zorder)
        
        plt.plot([pd.to_datetime(lower_left_pb[0]), pd.to_datetime(upper_right_pb[0])],
                    [upper_right_pb[1], upper_right_pb[1]], color=color, alpha=alpha, lw=lw, zorder=zorder)
    
    
        # IQR-BOX:
        
        #calculate the quantiles of peaks and timings and convert timings in dates
        peaks_quantiles = mquantiles(all_peaks_of_group, prob=[0.0,0.25,0.5,0.75,1.0])
        hours_quantiles = mquantiles(sorted(all_hours_of_group), prob=[0.0,0.25,0.5,0.75,1.0]).astype(int)
        dates_quantiles = ['']*5
        for i in range(5):
            dates_quantiles[i] = str(all_dates_hours['date'].loc[all_dates_hours['hour'] == 
                                                                       hours_quantiles[i]].iloc[0])
           
        #lower left coordinate set to the 25% quartile of the peak timing (t25) 
        #and the 25% quartile of the peak discharges of all members during the whole forecast period (p25)
        lower_left_IQRbox = [dates_quantiles[1], peaks_quantiles[1]]
    
        #upper right coordinate of the IQR-Box is defined as the 75% quartile of the peak timing (t75) 
        #and the 75% quartile of the peak discharges of all members (p75)
        upper_right_IQRbox = [dates_quantiles[3], peaks_quantiles[3]]
        
        plt.plot([pd.to_datetime(lower_left_IQRbox[0]), pd.to_datetime(lower_left_IQRbox[0])],
                    [lower_left_IQRbox[1], upper_right_IQRbox[1]], color=color, alpha=alpha, lw=lw, zorder=zorder)
        
        plt.plot([pd.to_datetime(lower_left_IQRbox[0]), pd.to_datetime(upper_right_IQRbox[0])],
                    [lower_left_IQRbox[1], lower_left_IQRbox[1]], color=color, alpha=alpha, lw=lw, zorder=zorder)
        
        plt.plot([pd.to_datetime(upper_right_IQRbox[0]), pd.to_datetime(upper_right_IQRbox[0])],
                    [lower_left_IQRbox[1], upper_right_IQRbox[1]], color=color, alpha=alpha, lw=lw, zorder=zorder)
        
        plt.plot([pd.to_datetime(lower_left_IQRbox[0]), pd.to_datetime(upper_right_IQRbox[0])],
                    [upper_right_IQRbox[1], upper_right_IQRbox[1]], color=color, alpha=alpha, lw=lw, zorder=zorder)
    
        # MEDIAN OF THE PEAK DISCHARGE:
        
        #horizontal line going from t0 to t100 representing the median of the peak discharge (p50) 
        #of all members of the ensemble forecast
        plt.plot([pd.to_datetime(lower_left_pb[0]), pd.to_datetime(upper_right_pb[0])],
                    [peaks_quantiles[2], peaks_quantiles[2]], color=color, alpha=alpha, lw=lw, zorder=zorder)
    
        # MEDIAN OF THE PEAK TIMING:
        
        #vertical line going from p0 to p100 representing the median of the peak timing (t50)
        plt.plot([pd.to_datetime(dates_quantiles[2]), pd.to_datetime(dates_quantiles[2])],
                    [lower_left_pb[1], upper_right_pb[1]], color=color, alpha=alpha, lw=lw)
    
        # MEDIAN VALUE: CROSS OF THE TWO MEDIANS
        
        median_value = ax.plot(pd.to_datetime(dates_quantiles[2]), peaks_quantiles[2], '*', markersize=20, color=color, alpha=1.0, lw=lw, zorder=zorder+1,
                               markeredgecolor='black', markeredgewidth=1.5, label='($t_{50}$, $p_{50}$)')
    
       
    ax.grid(True)
    
    #y axis limits
    #ax.set_ylim([0,500])
    plt.ylabel('Discharge [m3 s-1]')
    #x axis ticks and limits
    days = mdates.DayLocator()
    hours = mdates.HourLocator()
    yearsFmt = mdates.DateFormatter('%Y-%m-%d') # %H:%M')
    
    ax.xaxis.set_major_locator(days)
    ax.xaxis.set_major_formatter(yearsFmt)
    ax.xaxis.set_minor_locator(hours)
    # min and max on x axis
    datemin = np.datetime64(ens.date[0], 'm') - np.timedelta64(60, 'm')
    datemax = np.datetime64(ens.date[119], 'm') + np.timedelta64(25, 'm')
    ax.set_xlim(datemin, datemax)
    
    #Legend:
    fig.legend(handles=[runoff_member[0], l2[0], peak_member[0], median_value[0], peak_obs[0]], ncol=1, loc=(0.83,0.66), numpoints = 1,
           labels=['Runoff member', 'Runoff obs', '($t_i$, $p_i$), $i \in $#realizations', '($t_{50}$, $p_{50}$)',
                   '($t_{obs}$, $p_{obs}$)']); #loc=(0.66,0.66)
        
    #Text box to show the tuneable parameters:
    tuneable_params = '\n'.join((f'$\Delta_t$={delta_t} h',  f'$\gamma$={gamma:.3}', f'decreas_hours={decreas_hours} h', f'$\\beta$={beta:.2}'))
    ax.text(1.05, 0.65, tuneable_params, transform=ax.transAxes, fontsize=11, 
           verticalalignment='top', bbox=dict(facecolor='none', edgecolor='black', boxstyle='round')) #0.7, 0.65
    
    
    plt.rcParams.update({'font.size': 10});
    
    return plt.show()
    
    
    
    
    
    
    