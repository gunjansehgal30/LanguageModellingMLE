#Creating a pandas cheat sheet

#Importing Data
import pandas as pd
import sqlite3
import numpy as np
#######################  Importing Data (Similarly we can export data in similar formats               #####################################

#read from csv
df1=pd.read_csv("train.csv")

#read from a delimeter seperated file likr '\t' or ','
df1=pd.read_table("train.csv",delimiter=",")

#read from a sqlite database
conn = sqlite3.connect('database.db')
print ("Opened database successfully")
df1=pd.read_sql("select * from modelData;", conn)

#read from a json
df1=pd.read_json("screen4.txt")



###################     Creating Data frames          ##############################

dict={"A":[1,2,3,4,5,np.nan],"B":["A","B","C","E","F","G"]}
df1=pd.DataFrame(data=dict)


#################       Viewing Data    ###########################################

#Gives first n rows
df1.head(10)
# Gives last n rows
df1.tail(5)
#get the shape of the dataframe
df1.shape
# get the information about columns, datatypes and memory usage
df1.info()
#get the summary statistics about the numerical attributes of the dataframe
df1.describe()

#########################  Selection ######################################################################
df1["Pressure"]   # Rerurns the single column as a series
df1[["Pressure","Rotation"]]  #Returns a set of columns as a dataframe
df1.iloc[10]   # to select aparticular row  iloc
df1.iloc[0:10,:]  # to select a subset of data  #subselection by indexes
df1.iloc[[0,1,2],[0,1,2,3]]  #explicitely mention the row and column indices to be selected
df1.loc[0:10,'store':'sales']  # subselection by names where the row part represents the row index no and column part represents the column names
'''
#difference between numpy and pandas
import numpy as np
np1= np.random.randn(3,3)
np1[0,:]   # no requirement of iloc functions
'''
# putting conditions for subselection
df1[df1['store']==2]
df1.loc[df1['store']>2,['store','item']]

#Random Sampling
df1.sample(n=10,replace=True )  #random sampling n= no of samples to be chosen replace= sampling with or without replacement
df1.sample(frac=0.5, replace=False)  # to sample 50 percent of the data axis=1=columns

# sort by a column/ few columns and return the top few enteries
df1.nlargest(10,['item','sales'])
df1.nsmallest(1,'sales')

####################################   Data Cleaning    ##################################################

#Renaming all Columns
df1.columns=['A','B','C','D']
df1.rename(columns={'store': 'Store'})

#handling null values
df1.isnull()
df1.notnull()

df1.dropna(axis=0)  #remove rows with null values
df1.dropna(axis=1)  #removes columns with null values

df1.fillna(0)  #filling all the null values by a value
df1.fillna(df1.mean()) # filling the numerical column missing value by mean, mode, meadin

#replaceing a value or a set of values in the dataframe
df1.replace("A","Apple")
df1.replace(["A","B"],["Apple","Ball"])
df1.replace(np.nan,0)


#############################    Reshaping the data   #########################################

#concat two dataframes
dict={"col1":[1,2,3,4,5,np.nan],"col2":["A","B","C","E","F","G"]}
df1=pd.DataFrame(data=dict)

dict={"col1":[11,22,33,44,55,np.nan],"col2":["AA","BB","CC","EE","FF","GG"]}
df2=pd.DataFrame(data=dict)

pd.concat([df1,df2],axis=0)  # concat two dataframes by placing then one on top of other (inc no of rows)
pd.concat([df1,df2], axis=1) # concat by placing them on the side (inc no of columns)

#converting the multiple columns to a lesser columns

dict={"col1":[1,2,3,4,5,np.nan],"col2":["A","B","C","E","F","G"],"col3":["apple","orange","guava","pineapple",np.nan,"litchi"]}
df=pd.DataFrame(data=dict)

df=df.dropna()
x=pd.melt(df,id_vars=["col1"], value_vars=["col2","col3"])


#reverse of melt is pivot Converting rows to columns
x=x.pivot(index="col1",columns='variable', values='value') #pivot cannot handle dupicacy pivot_table can do that
x=x.reset_index()

#dropping columns from the dataframe
m=x.drop(columns=['col1'])



#pivot table

df = pd.DataFrame({"A": ["foo", "foo", "foo", "foo", "foo",
                       "bar", "bar", "bar", "bar"],
                    "B": ["one", "one", "one", "two", "two",
                         "one", "one", "two", "two"],
                   "C": ["small", "large", "large", "small",
                        "small", "large", "small", "small",
                        "large"],
                  "D": [1, 2, 2, 3, 3, 4, 5, 6, 7],
                  "E": [2, 4, 5, 5, 6, 6, 8, 9, 9]})

df.pivot_table(index=["A"],columns=["C","B"],values=["D","E"],aggfunc={'D':np.max, 'E':np.min})


#sorting
df1=pd.read_json("screen4.txt")
#sort by single column
df1.sort_values('Pressure', ascending=False)
#sort by multiple columns
df1.sort_values(['Pressure','Rotation'],ascending=[True,False])

#sort the index values
df1.sort_index()

#reset the insdex value starting from 0 for a dataframe.Also the current index value becomes a column
df1.reset_index()
df1.reset_index(drop=True)  #if we apply drop the old index column gets dropped

#grouping

df = pd.DataFrame({"A": ["foo", "foo", "foo", "foo", "foo",
                     "bar", "bar", "bar", "bar"],
               "B": ["one", "one", "one", "two", "two",
                        "one", "one", "two", "two"],
                    "C": ["small", "large", "large", "small",
                          "small", "large", "small", "small",
                          "large"],
                    "D": [1, 2, 2, 3, 3, 4, 5, 6, 7],
                    "E": [2, 4, 5, 5, 6, 6, 8, 9, 9]})

x=df.groupby(["A","B","C"])["D","E"].mean()
# how to use the dataframe returned by group by  Just reset index/
x.reset_index()

################################   Join/Merge #################################################################
df1=pd.DataFrame({"x1":["A","B","C"],"x2":[1,2,3]})
df2=pd.DataFrame({"x1":["A","B","D"],"x3":["T","F","T"]})

#left
pd.merge(df1,df2,on="x1",how="left")

#right
pd.merge(df1,df2,on="x1",how="right")

#outer
pd.merge(df1,df2,on="x1",how="outer")

#right
pd.merge(df1,df2,on="x1",how="inner")


###################################     Summarize data      ###########################################################
df2=pd.DataFrame({"x1":["A","B","C","D"],"x3":["T","F","T",np.nan],"x2":[1,3,3,np.nan]})
df2["x2"].value_counts()

df2["x3"].nunique()

df2["x3"].unique()


