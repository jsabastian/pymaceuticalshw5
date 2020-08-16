#!/usr/bin/env python
# coding: utf-8

# ## Analysis
# 
# ### 1: What seems very obvious is the decreased tumor burden in mice treated with Capomulin over the 45 days.
# 
# ### 2: Looking at the boxplot data, it appears that Ramicane may have a similar effect on tumor growth. 
# 
# ### 3: The pie chart indicates that there is an equal number of each sex in the study, which can further allow for  gender effect of treatment.
# 
# ### 4: I would like to have seen survival data...

# In[1]:


#import dependencies
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as st
import numpy as np
from scipy.stats import linregress


# In[2]:


# read csv files from working directory
mus_meta_df = "data/Mouse_metadata.csv"
trial_result_df = "data/Study_results.csv"


# In[3]:


# Read the file into dataframe
mus_meta_df = pd.read_csv(mus_meta_df)
trial_result_df = pd.read_csv(trial_result_df)


# In[4]:


# View the first 10 rows of data
mus_meta_df.head(10)


# In[5]:


# View the first 10 rows of data
trial_result_df.head(10)


# In[6]:


#number of rows in the df, for reference after removal of data
count_row = trial_result_df.shape[0]
count_row


# In[7]:


#id the duplicate id - timepoint data
pd.concat(g for _, g in trial_result_df.groupby(["Mouse ID","Timepoint"]) if len(g) > 1)


# In[8]:


# remove duplicate data for g989 data,which should be 10 rows
trial_result_df.drop_duplicates(subset=["Mouse ID","Timepoint"], keep= False)


# In[9]:


#merge the cleaned dataframes
merged_df = pd.merge(mus_meta_df, trial_result_df, on="Mouse ID")
merged_df.info()


# In[10]:


#generate a table of summary stats for tumor volume by drug regimen columns
# quick aggregation of the data for each of the stats
trial_summary = merged_df.groupby('Drug Regimen').agg(        tum_vol_avg=('Tumor Volume (mm3)', np.mean),                                                      
        tum_vol_median=('Tumor Volume (mm3)', np.median),\
                                                      
        tum_vol_var=('Tumor Volume (mm3)', np.var),\
                                                      
        tum_vol_stdev=('Tumor Volume (mm3)', np.std),\
        # there is no numpy call for sem                                              
        tum_vol_sem =('Tumor Volume (mm3)', st.sem)\
).round(3)


# In[11]:


trial_summary


# In[12]:


#use pyplot to gen a barplot of total mice per regimen
mice_per_trial = merged_df.groupby('Drug Regimen')


# In[13]:


#grab the number of data points ('mice') per trial
mpt_count = pd.DataFrame(mice_per_trial['Drug Regimen'].count())


# In[14]:


#set axis and ticks
x_axis = np.arange(len(mpt_count))
ticks = [i for i in x_axis]


# In[15]:


# set plot size
plt.figure(figsize=(9,8))
plt.bar(x_axis, mpt_count["Drug Regimen"], color='black', align="center", width = 0.52)
# list of x labels
plt.xticks(ticks, list(mpt_count.index), rotation="vertical")

# Set axis limits
plt.xlim(-0.7, len(x_axis)-0.3)
plt.ylim(0, max(mpt_count["Drug Regimen"])*1.4)

# titles and axis labels
plt.title("Total # of Mice per Regimen")
plt.xlabel("Drug Regimen")
plt.ylabel("Count")

column_name = ["Drug Regimen"]
plt.legend(column_name,loc="best")


# In[16]:


# use pandas to create the same plot

mice_per_trial = mpt_count.plot(kind='bar', title="Total # of Mice per Regimen", color="black")

# class method for labeling
mice_per_trial.set_xlabel("Drug Regimen")
mice_per_trial.set_ylabel("Count")


# In[17]:


# produce a pieplot of mouse gender makeup
gender_df = merged_df.groupby('Sex')

#grab the count
gender_df_count = pd.DataFrame(gender_df['Sex'].count())


# In[18]:


gender_df_count


# In[19]:


#use pyplot to produce the piechart

#use index of the counts df for labels
mouse_gender_df = list(gender_df_count.index.values)


# In[20]:


# each section of the chart will have sex as a value
gender_df_count = gender_df_count['Sex']


# In[21]:


# format the color of the chart
colors = ["green", "yellow"]


# In[22]:


# calculation and format
plt.pie(gender_df_count, labels = mouse_gender_df, colors=colors,
        autopct="%1.1f%%", shadow = True, startangle = 140)
plt.rcParams['font.size'] = 16
plt.title("Gender Makeup of the Study")
plt.ylabel('Sex')


# In[23]:


# use pandas to make the same piechart with the same dataframe
gender_df_count.plot(kind = 'pie',                         title = "Gender Makeup of the Study",startangle = 140,                        autopct = '%1.1f%%',shadow = True, fontsize = 16, colors  = ["green","yellow"],legend = False)


# In[24]:


merged_df.head()


# In[25]:


# Create a list of the four drugs to examine
drugs_of_interest = ['Capomulin', 'Ramicane', 'Infubinol', 'Ceftamin']

# look for the chosen drugs by slicing throug the merged df
drugs_df = merged_df[merged_df['Drug Regimen'].isin(drugs_of_interest)]


# In[26]:


drugs_df.head(10)


# In[27]:


# groupby Mouse ID and quick aggregation
# grab the final tumor volume
# by locating -1 element using lamda anonymous function
final_vol_list = drugs_df.groupby(['Drug Regimen','Mouse ID']).agg(        # Get the last value of the 'Tumor Volume (mm3)' column using a lambda function selecting the element in -1 position
        final_size=('Tumor Volume (mm3)',lambda x: x.iloc[-1])).round(3)


# In[28]:


# check the df
final_vol_list


# In[29]:


#transpose above df
reshape_df = final_vol_list.stack(level = 0).unstack(level = 0)


# In[30]:


reshape_df.head()


# In[31]:


#Calculate the IQR and quantitatively determine if there are any potential outliers.

counter = 0


# Do quartile calculations for each drug
for drug in drugs_of_interest:
    quarters  =  reshape_df[drug].quantile([.25,.5,.75]).round(2)
    lower_quarter  =  quarters[0.25].round(2)
    upper_quarter  =  quarters[0.75].round(2)
    iqr  =  round(upper_quarter-lower_quarter,2)
    lower_bound  =  round(lower_quarter - (1.5*iqr),2)
    upper_bound  =  round(upper_quarter + (1.5*iqr),2)
    
    # print an escape line for each new loop
    if counter == 0:
        print(f'\n')
    print(f"{drug} IQR data is:")
    print(f"The lower quartile of {drug} is: {lower_quarter}")
    print(f"The upper quartile of {drug} is: {upper_quarter}")
    print(f"The interquartile range of {drug} is: {iqr}")
    print(f"The the median of {drug} is: {quarters[0.5]} ")
    print(f"Values below {lower_bound} for {drug} could be outliers.")
    print(f"Values above {upper_bound} for {drug} could be outliers.")
    print(f'\n')
    counter += 1


# In[32]:


#box and whisker plot for final tum volumes, and identify outliers

# Create an empty list to populate
boxplot_list  =  []

# loop through the list of the four drugs and populate the boxplot list (list of lists), removing na values
for i in drugs_of_interest:
    boxplot_list.append(list(reshape_df[i].dropna()))


# In[33]:


# Plot the list of lists and id outlier with a blue diamond
fig1, ax  =  plt.subplots(figsize = (9,7))
ax.set_title('Final Tumor Volume per each Drug')
ax.set_xlabel('Drug Regimen')
ax.set_ylabel('Tumor Vol (mm3)')
ax.boxplot(boxplot_list,notch = 0,sym = 'bD')
plt.xticks([1,2,3,4],drugs_of_interest)


# In[34]:


#df for single mouse treated with Capomulin
capo_mus = merged_df.loc[merged_df['Drug Regimen'] == 'Capomulin']


# In[35]:


capo_mus.head()


# In[36]:


# group by timepoints and calc tum_vol_avg

cap_tum_timepoint = capo_mus.groupby(['Timepoint']).agg(        # Get the mean of the 'Tumor Volume (mm3)' column\
        Tum_Vol_Avg=('Tumor Volume (mm3)', np.mean),\
).round(3)


# In[37]:


cap_tum_timepoint.head(10)


# In[38]:


# Plot
# store x values
x_time_points = list(cap_tum_timepoint.index.values)

plt.plot(
    x_time_points,
    cap_tum_timepoint['Tum_Vol_Avg'],
    label="Timepoints",
    linewidth=3  # width of plot line
    )
# Add the descriptive title, x labels and y labels
plt.title("Time Series of Tumor Volume for Capomulin")
plt.xlabel("Time (days)")
plt.ylabel("Tumor Volume (mm3)")

# Set x and y limits 
plt.xlim(min(x_time_points)-max(x_time_points)*0.05, max(x_time_points)*1.05)
plt.ylim(min(cap_tum_timepoint['Tum_Vol_Avg'])*0.95, max(cap_tum_timepoint['Tum_Vol_Avg'])*1.05)
plt.rcParams["figure.figsize"] = [8,7]


# In[39]:


# Scatter Plot of mouse weight by tumor volume avg
capo_mus.head()


# In[40]:


# Groupby Mouse ID aggregation
mouse_ID  =  capo_mus.groupby(['Mouse ID']).agg(        # mean
        weight = ('Weight (g)', np.mean),\
        tum_vol_avg = ('Tumor Volume (mm3)', np.mean)\
).round(3)


# In[41]:


mouse_ID.head(30)


# In[42]:


#create scatter plot from the above dataframe
plt.scatter(
    mouse_ID['weight'],
    mouse_ID['tum_vol_avg'],
    marker = 'o',
    facecolors = 'gray',
    edgecolors = 'black',
    s = mouse_ID['tum_vol_avg'],
    alpha = .75)

# Create a title, x label, and y label for our chart
plt.title("Mouse weight vs. Avg. Tumor Volume")
plt.xlabel("Mouse weight (g)")
plt.ylabel("Tumor Volume (mm3)")


# In[43]:


#correlation coefficient and linear regression model weight and average tumor volume under Capo treatment
correlation  =  st.pearsonr(mouse_ID['weight'],mouse_ID['tum_vol_avg'])
print(f"Correlation is:  {round(correlation[0],3)}")


# In[45]:


# r-squared analysis
x_values  =  mouse_ID['weight']
y_values  =  mouse_ID['tum_vol_avg']
(slope, intercept, rvalue, pvalue, stderr)  =  linregress(x_values, y_values)
regress_values  =  x_values * slope + intercept
#line_eq  =  "y  =  " + str(round(slope,2)) + "x + " + str(round(intercept,2))
line_eq  =  f'y  =  {str(round(slope,2))}x + {str(round(intercept,2))}'
plt.scatter(x_values,y_values)
plt.plot(x_values,regress_values,"r-")
plt.annotate(line_eq,(17,37),fontsize = 15,color = "black")
plt.title("Weight by Average Tumor Volume")
plt.xlabel("Weight (g)")
plt.ylabel("Tumor Volume (mm3)")

plt.show()


# In[ ]:




