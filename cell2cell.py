#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas
import mba263
import matplotlib
import matplotlib.pyplot as plt


# In[2]:


data=pandas.read_csv('cell2cell.csv')


# In[3]:


data


# In[4]:


data_calibrat=data[data['calibrat']==1]


# In[5]:


data_validation=data[data['calibrat']==0]


# In[6]:


data_calibrat


# In[ ]:





# In[7]:


res=mba263.logit(data_calibrat['churndep'],data_calibrat.loc[:,'revenue':'retcall'])


# In[8]:


analysis=mba263.odds_ratios(res)


# In[54]:


analysis=mba263.odds_ratios(res)
pandas.set_option('display.max_rows', None)
analysis.loc['revenue','Odds ratios']


# In[10]:


analysis.to_csv('Variable_Odds_Analysis.csv')


# In[11]:


SignVar=analysis.loc[analysis['P>|z|']<0.05,:]


# In[12]:


pandas.set_option('display.max_rows', None)
SignVar


# In[13]:


SignVar.index


# In[14]:


Varlist=['revenue', 'mou', 'recchrge', 'overage', 'roam', 'changem', 'changer',
       'unansvce', 'custcare', 'threeway', 'incalls', 'peakvce', 'months',
       'uniqsubs', 'actvsubs', 'phones', 'eqpdays', 'age1', 'children',
       'credita', 'creditaa', 'refurb', 'webcap', 'marryun','newcelly',
       'income', 'setprcm', 'setprc', 'retcall']
Varlist1=['revenue', 'mou', 'recchrge', 'overage', 'roam', 'changem', 'changer',
       'unansvce', 'custcare', 'threeway', 'incalls', 'peakvce', 'months',
       'uniqsubs', 'actvsubs', 'phones', 'eqpdays', 'age1', 'children',
       'credita', 'creditaa', 'refurb', 'webcap', 'marryun','newcelly',
       'income', 'setprcm', 'setprc']


# In[15]:


res2=mba263.logit(data_calibrat['churndep'],data_calibrat[Varlist])


# 

# In[16]:


pandas.set_option('display.max_rows', None)
mba263.odds_ratios(res2)


# In[17]:


SignVar_bad=SignVar[SignVar['Odds ratios']>1.01]


# Sort out high-influential factors that will increase churn rate

# In[18]:


SignVar_bad


# Look at the results, whether the customer made retention call are a major influencer of the churn rate

# Whether the customer have refurbished phone is also a strong influencer. That means if the customer does not have a good phone (i.e. refurbished phone), he/she is likely to switch providers, cause many providers will use new phone as a method for customer acquisition.

# In[19]:


SignVar_good=SignVar[SignVar['Odds ratios']<0.99]
print(SignVar_good)


# Sort out high-influential factors that will help to decrease churn rate

# Based on results, actvsubs is a strong influencer on decerase churn rate, which also echo the uniqsubs increase churn rate. This means, within an account, more active users will decrease churn rate, and more static users (dead users) will increase the churn rate.

# Credita and Creditaa users will have lower churn rate. Naturally true, since users with higher credit score are not so price sensitive, and not likely want to spend time and hussle to change providers.
# 
# Webcap also a strong influencer. This means users using data plan (who also high-value users) are less likely to churn.
# 
# NewCelly, new cell phone users, who are likely teenagers, have less churn rate.
# 
# Setprcm, missing data on handset price is negative influencer on churn rate. Somewhat surprising. What is major reason for missing the handset price? It could be users used non-mainstream phones, certainly not purchased through Cell2Cell, or simply because a phone number in the account not linked to a phone. We could dig deeper to find major reasons:

# In[39]:


data_calibrat_setprcm=data_calibrat[data_calibrat['setprcm']==1]
data_calibrat_setprcm['phones'].mean()


# In[40]:


data_calibrat_setprcm['uniqsubs'].mean()


# The mean of "phones" on the full dataset is 1.8, while with setprcm = 1, the "phones" mean drops significantly to 1.0. However, the mean of 'uniqsubs' doesn't change much (1.529 vs. 1.55). This means, for setprcm=1, number of users within account doesn't change much, but number of phones changes a lot. Which means, for setprcm=1 accounts, many user numbers not linked to a phone, or linked to a phone not count as "issued phone", which highly likely to be 1G or 2G phones.

# In[55]:


odds_processed=pandas.read_csv('odds_ratio.csv')


# In[57]:


odds_processed


# 

# Below commented codes were to study interaction term of actvsubs and uniqsubs

# In[20]:


#See interaction between uniqsubs and actvsubs, create interation terms
#data_calibrat[['actvsubs','uniqsubs']]=data_calibrat[['actvsubs','uniqsubs']].replace(0,0.1)
#data_validation[['actvsubs','uniqsubs']]=data_validation[['actvsubs','uniqsubs']].replace(0,0.1)

#data_calibrat['uniq_div_act']=data_calibrat['uniqsubs']/data_calibrat['actvsubs']
#data_validation['uniq_div_act']=data_validation['uniqsubs']/data_validation['actvsubs']

#data_calibarate_interactive_predictor = data_calibrat.loc[:,'revenue':'retcall']
#data_calibarate_interactive_predictor['uniq_div_act']=data_calibrat['uniq_div_act']
#data_validation_interactive_predictor = data_validation.loc[:,'revenue':'retcall']
#data_validation_interactive_predictor['uniq_div_act']=data_validation['uniq_div_act']


# In[21]:


#pandas.set_option("display.max_rows", 20)
#data_calibarate_interactive_predictor['uniq_div_act'].sum()


# In[22]:


#res_interaction=mba263.logit(data_calibrat['churndep'],data_calibarate_interactive_predictor)


# In[23]:



#data_calibarate_interactive_predictor['uniq_div_act'].head(50)


# In[24]:


#pandas.set_option("display.max_rows", 80)
#analysis_inter = mba263.odds_ratios(res_interaction)
#analysis_inter


# In[25]:


res_retcall=mba263.logit(data_calibrat['retcall'],data_calibrat[Varlist1])


# Study the logitic model for retcall, since it is the strongest influencer to churn rate, then let's investigate what will increase the chance of retention call.

# In[26]:


analysis_retcall=mba263.odds_ratios(res_retcall)
analysis_retcall_good=analysis_retcall.loc[analysis_retcall['P>|z|']<0.05,:]
#pandas.set_option('display.max_rows', None)
analysis_retcall_good


# Results show that actvsubs and setprcm will be most effective factors to decrease retention call. While refurbished phone and uniqsubs are the worst factors for increases retention call.

# In[27]:


data_calibrat['uniq_minus_actv']=data_calibrat['uniqsubs']-data_calibrat['actvsubs']
data_validation['uniq_minus_actv']=data_validation['uniqsubs']-data_validation['actvsubs']
data_calibrat_setprcm=data_calibrat[data_calibrat['setprcm']==1]
data_calibrat_setprcm['uniq_minus_actv'].mean()


# In[28]:


data_calibrat['uniq_minus_actv'].mean()


# In[29]:


Varlist2=['revenue', 'mou', 'recchrge', 'overage', 'roam', 'changem', 'changer',
       'unansvce', 'custcare', 'threeway', 'incalls', 'peakvce', 'months',
       'actvsubs', 'phones', 'eqpdays', 'age1', 'children',
       'credita', 'creditaa', 'refurb', 'webcap', 'marryun','newcelly',
       'income', 'setprcm', 'setprc', 'retcall','uniq_minus_actv']

#res3=mba263.logit(data_calibrat['churndep'],data_calibrat[Varlist2])
res3=mba263.logit(data_calibrat['churndep'],data_calibrat.loc[:,'revenue':'uniq_minus_actv'])


# In[64]:


data_calibrat['uniq_minus_actv'].to_frame().std()


# In[65]:


mba263.odds_ratios(res3)


# In[31]:


#Evaluate validation data with models
data_validation['p_res']=res.predict(data_validation.loc[:,'revenue':'retcall'])
data_validation['p_res2']=res2.predict(data_validation[Varlist])
data_validation['p_res3']=res3.predict(data_validation.loc[:,'revenue':'uniq_minus_actv'])
#data_validation['p_interaction']=res_interaction.predict(data_validation_interactive_predictor)


# In[43]:


lift_res=mba263.lift(data_validation['churn'],data_validation['p_res'],bins=10)
lift_res2=mba263.lift(data_validation['churn'],data_validation['p_res2'],bins=10)
lift_res3=mba263.lift(data_validation['churn'],data_validation['p_res3'],bins=10)
#lift_interaction=mba263.lift(data_validation['churn'],data_validation['p_interaction'],bins=10)

plt.plot(lift_res,label='full variables')
#plt.plot(lift_res2,label='stats meaningful')
#plt.plot(lift_res3,label='added inactive users')
#plt.plot(lift_interaction,label='interaction')
plt.legend()


# In[33]:


data_validation.loc[data_validation['p_res2'].isna(),'churn'].sum()

