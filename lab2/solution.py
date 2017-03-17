import matplotlib
matplotlib.use('Agg')

import pandas as pd
from scipy.stats import norm
import random
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
#calclulation of std deviation
df = pd.read_csv('./vehicles.csv')
print (df.columns)
current_fleet_std = np.std(df['Current fleet'])
print ("Current Fleet Std : ", current_fleet_std)

new_fleet_std = np.std(df['New Fleet'])
print ("New Fleet Std : ", new_fleet_std)

def boostrap(statistic_func, iterations, data):
	samples  = np.random.choice(data,replace = True, size = [iterations, len(data)])
	#print samples.shape
	data_std = data.std()
	vals = []
	for sample in samples:
		sta = statistic_func(sample)
		#print sta
		vals.append(sta)
	b = np.array(vals)
	#print b
	lower, upper = np.percentile(b, [2.5, 97.5])
	return data_std,lower, upper

#Finding the upper and lower bound of current fleet
boots = []
for i in range(10,100000,1000):
	boot = boostrap(np.std, i, df['Current fleet'].T)
	boots.append([i,boot[0], "std"])
	boots.append([i,boot[1], "lower"])
	boots.append([i,boot[2], "upper"])

df_boot = pd.DataFrame(boots,  columns=['Boostrap Iterations','std',"Value"])
sns_plot = sns.lmplot(df_boot.columns[0],df_boot.columns[1], data=df_boot, fit_reg=False,  hue="Value")

sns_plot.axes[0,0].set_ylim(0,)
sns_plot.axes[0,0].set_xlim(0,100000)

sns_plot.savefig("bootstrap_current_fleet.png",bbox_inches='tight')
sns_plot.savefig("bootstrap_current_fleet.pdf",bbox_inches='tight')

#Finding the upper and lower bound of new fleet
#print df['New Fleet'].T
data_nf = df['New Fleet'].dropna()
print( data_nf)
boots = []
for i in range(10,100000,1000):
	boot = boostrap(np.std, i, data_nf)
	boots.append([i,boot[0], "std"])
	boots.append([i,boot[1], "lower"])
	boots.append([i,boot[2], "upper"])

df_boot = pd.DataFrame(boots,  columns=['Boostrap Iterations','std',"Value"])
sns_plot = sns.lmplot(df_boot.columns[0],df_boot.columns[1], data=df_boot, fit_reg=False,  hue="Value")

sns_plot.axes[0,0].set_ylim(0,)
sns_plot.axes[0,0].set_xlim(0,100000)

sns_plot.savefig("bootstrap_new_fleet.png",bbox_inches='tight')
sns_plot.savefig("bootstrap_new_fleet.pdf",bbox_inches='tight')
