#In[]

import findspark 
import pyspark
import math
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import statistics as stats
from sklearn import neural_network
from sklearn.neural_network import MLPClassifier
from pyspark.sql.functions import max , min , mean 
import pyspark.sql.functions as func 
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()
findspark.init()



# In[]
df = spark.read.csv("nuclear_plants_small_dataset.csv", inferSchema=True , header = True )
df.printSchema()


#In[]
#Task1
#dealing with missing data


status_count = df.groupBy("Status").count()
status_count.show()

rows = status_count.select("count").collect()


normalCount=rows[0][0]
abnormalCount=rows[1][0]

total = normalCount + abnormalCount

normalPercentage = round((normalCount/total)*100 )
abnormalPercentage = round((abnormalCount/total)*100)



print('{0:0d} % normal : {1:0d} % abnormal'.format(normalPercentage,abnormalPercentage))
print("total" , total)

#In[]
#Task1
# Get count of both null and missing values in the Dataset

from pyspark.sql.functions import isnan, when, count, col
df.select("Status").show(1)
df1 = df.select("Status","Power_range_sensor_1","Power_range_sensor_2","Power_range_sensor_3 ","Power_range_sensor_4")
df2 = df.select("Pressure _sensor_1","Pressure _sensor_2","Pressure _sensor_3","Pressure _sensor_4")
df3 = df.select("Vibration_sensor_1","Vibration_sensor_2","Vibration_sensor_3","Vibration_sensor_4")
print("The count of missing data in the dataset")
print("----------------------------------------")
print(" ")


x = df1.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df1.columns]).show(truncate = 10)
y = df2.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df2.columns]).show(truncate = 10)
z = df3.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df3.columns]).show(truncate = 10)

#In[]
#task 2



#the power range sensors staticstics 

# #The maximum values power range sensors (Normal)
max_pr_1 = df.groupBy("status").max("Power_range_sensor_1").withColumnRenamed("max(Power_range_sensor_1)","max of power range 1")
max_pr_2 = df.groupBy("status").max("Power_range_sensor_2").withColumnRenamed("max(Power_range_sensor_2)","max of power range 2")
max_pr_3 = df.groupBy("status").max("Power_range_sensor_3 ").withColumnRenamed("max(Power_range_sensor_3 )","max of power range 3")
max_pr_4 = df.groupBy("status").max("Power_range_sensor_4").withColumnRenamed("max(Power_range_sensor_4)","max of power range 4")

max_pr_1.show()
max_pr_2.show()
max_pr_3.show()
max_pr_4.show()


# #the minimum values of power range sensors (Normal)
min_pr_1 = df.groupBy("status").min("Power_range_sensor_1").withColumnRenamed("min(Power_range_sensor_1)","minimum of power range 1")
min_pr_2 = df.groupBy("status").min("Power_range_sensor_2").withColumnRenamed("min(Power_range_sensor_2)","minimum of power range 2")
min_pr_3 = df.groupBy("status").min("Power_range_sensor_3 ").withColumnRenamed("min(Power_range_sensor_3 )","minimum of power range 3")
min_pr_4 = df.groupBy("status").min("Power_range_sensor_4").withColumnRenamed("min(Power_range_sensor_4)","minimum of power range 4")

min_pr_1.show()
min_pr_2.show()
min_pr_3.show()
min_pr_4.show()

# #The mean of power range sensors  (Normal)
average_pr_1 = df.groupBy("status").mean("Power_range_sensor_1").withColumnRenamed("avg(Power_range_sensor_1)","average power range sensor 1")
average_pr_2 = df.groupBy("status").mean("Power_range_sensor_2").withColumnRenamed("avg(Power_range_sensor_2)","average power range sensor 2")
average_pr_3 = df.groupBy("status").mean("Power_range_sensor_3 ").withColumnRenamed("avg(Power_range_sensor_3 )","average power range sensor 3")
average_pr_4 = df.groupBy("status").mean("Power_range_sensor_4").withColumnRenamed("avg(Power_range_sensor_4)","average power range sensor 4")


average_pr_1.show()
average_pr_2.show()
average_pr_3.show()
average_pr_4.show()

#the medians of power range sensors 
median1 = df.groupBy("status").agg(func.percentile_approx("Power_range_sensor_1",0.5).alias("median of power range 1"))
median2 = df.groupBy("status").agg(func.percentile_approx("Power_range_sensor_2",0.5).alias("median of power range 2"))
median3 = df.groupBy("status").agg(func.percentile_approx("Power_range_sensor_3 ",0.5).alias("median of power range 3"))
median4 = df.groupBy("status").agg(func.percentile_approx("Power_range_sensor_4",0.5).alias("median of power range 4"))

print("The median of power range sensors (Normal and Abnormal status)")
median1.show()
median2.show()
median3.show()
median4.show()

#the mode of power range sensors 
mode_prs = []
mode_prs.insert(0,df.groupby("Power_range_sensor_1").count().orderBy("count", ascending=False).first()[0])
mode_prs.insert(1,df.groupby("Power_range_sensor_2").count().orderBy("count", ascending=False).first()[0])
mode_prs.insert(2,df.groupby("Power_range_sensor_3 ").count().orderBy("count", ascending=False).first()[0])
mode_prs.insert(2,df.groupby("Power_range_sensor_4").count().orderBy("count", ascending=False).first()[0])

i = 1
for x in mode_prs:
  print("the mode of power range sensors",i, "=" , x)
  i+=1

df.agg({'Power_range_sensor_1': 'variance'}).show()
df.agg({'Power_range_sensor_2': 'variance'}).show()
df.agg({'Power_range_sensor_3 ': 'variance'}).show()
df.agg({'Power_range_sensor_4': 'variance'}).show()

print("###########################################################")

# In[]


#the pressure sensors staticstics 


#The maximum values pressure sensors 
max_ps_1 = df.groupBy("status").max("Pressure _sensor_1").withColumnRenamed("max(Pressure _sensor_1)","max of pressure sensor 1")
max_ps_2 = df.groupBy("status").max("Pressure _sensor_2").withColumnRenamed("max(Pressure _sensor_2)","max of pressure sensor 2")
max_ps_3 = df.groupBy("status").max("Pressure _sensor_3").withColumnRenamed("max(Pressure _sensor_3)","max of pressure sensor 3")
max_ps_4 = df.groupBy("status").max("Pressure _sensor_4").withColumnRenamed("max(Pressure _sensor_4)","max of pressure sensor 4")

max_ps_1.show()
max_ps_2.show()
max_ps_3.show()
max_ps_4.show()


# #the minimum values of pressure sensors 
min_ps_1 = df.groupBy("status").min("Pressure _sensor_1").withColumnRenamed("min(Pressure _sensor_1)","minimum of pressure sensor 1")
min_ps_2 = df.groupBy("status").min("Pressure _sensor_2").withColumnRenamed("min(Pressure _sensor_2)","minimum of pressure sensor 2")
min_ps_3 = df.groupBy("status").min("Pressure _sensor_3").withColumnRenamed("min(Pressure _sensor_3)","minimum of pressure sensor 3")
min_ps_4 = df.groupBy("status").min("Pressure _sensor_4").withColumnRenamed("min(Pressure _sensor_4)","minimum of pressure sensor 4")

min_ps_1.show()
min_ps_2.show()
min_ps_3.show()
min_ps_4.show()

#The mean of Pressure sensors   
average_ps_1 = df.groupBy("status").mean("Pressure _sensor_1").withColumnRenamed("avg(Pressure _sensor_1)","average pressure sensor 1")
average_ps_2 = df.groupBy("status").mean("Pressure _sensor_2").withColumnRenamed("avg(Pressure _sensor_2)","average pressure sensor 2")
average_ps_3 = df.groupBy("status").mean("Pressure _sensor_3").withColumnRenamed("avg(Pressure _sensor_3)","average pressure sensor 3")
average_ps_4 = df.groupBy("status").mean("Pressure _sensor_4").withColumnRenamed("avg(Pressure _sensor_4)","average pressure sensor 4")


average_ps_1.show()
average_ps_2.show()
average_ps_3.show()
average_ps_4.show()

#the medians of Pressure sensors 
median12 = df.groupBy("status").agg(func.percentile_approx("Pressure _sensor_1",0.5).alias("median of pressure sensor 1"))
median22 = df.groupBy("status").agg(func.percentile_approx("Pressure _sensor_2",0.5).alias("median of pressure sensor 2"))
median32 = df.groupBy("status").agg(func.percentile_approx("Pressure _sensor_3",0.5).alias("median of pressure sensor 3"))
median42 = df.groupBy("status").agg(func.percentile_approx("Pressure _sensor_4",0.5).alias("median of pressure sensor 4"))

print("The median of pressure sensors (Normal and Abnormal status)")
median12.show()
median22.show()
median32.show()
median42.show()
#the mode of Pressure sensors 
mode_ps = []
mode_ps.insert(0,df.groupby("Pressure _sensor_1").count().orderBy("count", ascending=False).first()[0])
mode_ps.insert(1,df.groupby("Pressure _sensor_2").count().orderBy("count", ascending=False).first()[0])
mode_ps.insert(2,df.groupby("Pressure _sensor_3").count().orderBy("count", ascending=False).first()[0])
mode_ps.insert(2,df.groupby("Pressure _sensor_4").count().orderBy("count", ascending=False).first()[0])

i = 1
for x in mode_ps:
  print("the mode of Pressure sensors",i, "=" , x)
  i+=1


print("###########################################################")
print(" ")

#the variance of pressure sensors 

df.agg({'Pressure _sensor_1': 'variance'}).show()
df.agg({'Pressure _sensor_2': 'variance'}).show()
df.agg({'Pressure _sensor_3': 'variance'}).show()
df.agg({'Pressure _sensor_4': 'variance'}).show()

print("###########################################################")


# In[]

#the Vibration sensors staticstics 



#The maximum values Vibration sensors 
max_vs_1 = df.groupBy("status").max("Vibration_sensor_1").withColumnRenamed("max(Vibration_sensor_1)","max of Vibration sensor 1")
max_vs_2 = df.groupBy("status").max("Vibration_sensor_2").withColumnRenamed("max(Vibration_sensor_2)","max of Vibration sensor 2")
max_vs_3 = df.groupBy("status").max("Vibration_sensor_3").withColumnRenamed("max(Vibration_sensor_3)","max of Vibration sensor 3")
max_vs_4 = df.groupBy("status").max("Vibration_sensor_4").withColumnRenamed("max(Vibration_sensor_4)","max of Vibration sensor 4")

max_vs_1.show()
max_vs_2.show()
max_vs_3.show()
max_vs_4.show()


# #the minimum values of Vibration sensors (Normal)
min_vs_1 = df.groupBy("status").min("Vibration_sensor_1").withColumnRenamed("min(Vibration_sensor_1)","minimum of Vibration sensor 1")
min_vs_2 = df.groupBy("status").min("Vibration_sensor_2").withColumnRenamed("min(Vibration_sensor_2)","minimum of Vibration sensor 2")
min_vs_3 = df.groupBy("status").min("Vibration_sensor_3").withColumnRenamed("min(Vibration_sensor_3)","minimum of Vibration sensor 3")
min_vs_4 = df.groupBy("status").min("Vibration_sensor_4").withColumnRenamed("min(Vibration_sensor_4)","minimum of Vibration sensor 4")

min_vs_1.show()
min_vs_2.show()
min_vs_3.show()
min_vs_4.show()

#The mean of Vibration sensors   
average_vs_1 = df.groupBy("status").mean("Vibration_sensor_1").withColumnRenamed("avg(Vibration_sensor_1)","average Vibration sensor 1")
average_vs_2 = df.groupBy("status").mean("Vibration_sensor_2").withColumnRenamed("avg(Vibration_sensor_2)","average Vibration sensor 2")
average_vs_3 = df.groupBy("status").mean("Vibration_sensor_3").withColumnRenamed("avg(Vibration_sensor_3)","average Vibration sensor 3")
average_vs_4 = df.groupBy("status").mean("Vibration_sensor_4").withColumnRenamed("avg(Vibration_sensor_4)","average Vibration sensor 4")


average_vs_1.show()
average_vs_2.show()
average_vs_3.show()
average_vs_4.show()

#the medians of Vibration sensors 
median13 = df.groupBy("status").agg(func.percentile_approx("Vibration_sensor_1",0.5).alias("median of Vibration sensor 1"))
median23 = df.groupBy("status").agg(func.percentile_approx("Vibration_sensor_2",0.5).alias("median of Vibration sensor 2"))
median33 = df.groupBy("status").agg(func.percentile_approx("Vibration_sensor_3",0.5).alias("median of Vibration sensor 3"))
median43 = df.groupBy("status").agg(func.percentile_approx("Vibration_sensor_4",0.5).alias("median of Vibration sensor 4"))

print("The median of pressure sensors (Normal and Abnormal status)")
median13.show()
median23.show()
median33.show()
median43.show()

#the mode of Vibration sensors 
mode_vs = []

mode_vs.insert(0,df.groupby("Vibration_sensor_1").count().orderBy("count", ascending=False).first()[0])
mode_vs.insert(1,df.groupby("Vibration_sensor_2").count().orderBy("count", ascending=False).first()[0])
mode_vs.insert(2,df.groupby("Vibration_sensor_3").count().orderBy("count", ascending=False).first()[0])
mode_vs.insert(3,df.groupby("Vibration_sensor_4").count().orderBy("count", ascending=False).first()[0])

i = 1
for x in mode_vs:
  print("the mode of vibration sensors",i, "=" , x)
  i+=1
    

#the variance of Vibration sensors 

df.agg({'Vibration_sensor_1': 'variance'}).show()
df.agg({'Vibration_sensor_2': 'variance'}).show()
df.agg({'Vibration_sensor_3': 'variance'}).show()
df.agg({'Vibration_sensor_4': 'variance'}).show()


#In[]
#Task2 
#the box plot for each features 
df_pandas = df.toPandas()
boxplot1 = df_pandas.where(df_pandas.Status == "Normal").boxplot(grid=False, fontsize=15, column=['Power_range_sensor_1'])
#In[]
boxplot2 = df_pandas.where(df_pandas.Status == "Abnormal").boxplot(grid=False, fontsize=15, column=['Power_range_sensor_1'])
#In[]
boxplot3 = df_pandas.where(df_pandas.Status == "Normal").boxplot(grid=False, fontsize=15, column=['Power_range_sensor_2'])

#In[]
boxplot4 = df_pandas.where(df_pandas.Status == "Abnormal").boxplot(grid=False, fontsize=15, column=['Power_range_sensor_2'])

#In[]
boxplot5 = df_pandas.where(df_pandas.Status == "Normal").boxplot(grid=False, fontsize=15, column=['Power_range_sensor_3 '])

#In[]
boxplot6 = df_pandas.where(df_pandas.Status == "Abnormal").boxplot(grid=False, fontsize=15, column=['Power_range_sensor_3 '])

#In[]
boxplot7 = df_pandas.where(df_pandas.Status == "Normal").boxplot(grid=False, fontsize=15, column=['Power_range_sensor_4'])

#In[]
boxplot8 = df_pandas.where(df_pandas.Status == "Abnormal").boxplot(grid=False, fontsize=15, column=['Power_range_sensor_4'])
#In[]
boxplot9 = df_pandas.where(df_pandas.Status == "Normal").boxplot(grid=False, fontsize=15, column=['Pressure _sensor_1'])

#In[]
boxplot10 = df_pandas.where(df_pandas.Status == "Abnormal").boxplot(grid=False, fontsize=15, column=['Pressure _sensor_1'])

#In[]
boxplot11 = df_pandas.where(df_pandas.Status == "Normal").boxplot(grid=False, fontsize=15, column=['Pressure _sensor_2'])

#In[]
boxplot12 = df_pandas.where(df_pandas.Status == "Abnormal").boxplot(grid=False, fontsize=15, column=['Pressure _sensor_2'])

#In[]
boxplot13 = df_pandas.where(df_pandas.Status == "Normal").boxplot(grid=False, fontsize=15, column=['Pressure _sensor_3'])

#In[]
boxplot14 = df_pandas.where(df_pandas.Status == "Abnormal").boxplot(grid=False, fontsize=15, column=['Pressure _sensor_3'])

#In[]
boxplot15 = df_pandas.where(df_pandas.Status == "Normal").boxplot(grid=False, fontsize=15, column=['Pressure _sensor_4'])


#In[]
boxplot16 = df_pandas.where(df_pandas.Status == "Abnormal").boxplot(grid=False, fontsize=15, column=['Pressure _sensor_4'])
#In[]
boxplot17 = df_pandas.where(df_pandas.Status == "Normal").boxplot(grid=False, fontsize=15, column=['Vibration_sensor_1'])
#In[]
boxplot18 = df_pandas.where(df_pandas.Status == "Abnormal").boxplot(grid=False, fontsize=15, column=['Vibration_sensor_1'])

#In[]
boxplot19 = df_pandas.where(df_pandas.Status == "Normal").boxplot(grid=False, fontsize=15, column=['Vibration_sensor_2'])
#In[]
boxplot22 = df_pandas.where(df_pandas.Status == "Abnormal").boxplot(grid=False, fontsize=15, column=['Vibration_sensor_2'])

#In[]
boxplot20 = df_pandas.where(df_pandas.Status == "Normal").boxplot(grid=False, fontsize=15, column=['Vibration_sensor_3'])

#In[]
boxplot21 = df_pandas.where(df_pandas.Status == "Abnormal").boxplot(grid=False, fontsize=15, column=['Vibration_sensor_3'])

#In[]

boxplot23 = df_pandas.where(df_pandas.Status == "Normal").boxplot(grid=False, fontsize=15, column=['Vibration_sensor_4'])

#In[]
boxplot24 = df_pandas.where(df_pandas.Status == "Abnormal").boxplot(grid=False, fontsize=15, column=['Vibration_sensor_4'])

#In[]
#Task3

#In[]
#the correlation between power range sensors 
print(df.stat.corr("Power_range_sensor_1","Power_range_sensor_2"))
print(df.stat.corr("Power_range_sensor_1","Power_range_sensor_3 "))
print(df.stat.corr("Power_range_sensor_1","Power_range_sensor_4"))
print("----------------")
print(df.stat.corr("Power_range_sensor_2","Power_range_sensor_3 "))
print(df.stat.corr("Power_range_sensor_2","Power_range_sensor_4"))
print("----------------")
print(df.stat.corr("Power_range_sensor_3 ","Power_range_sensor_4"))
print("----------------")

print("the correlation betwen sensor 2 and 3 are the highest ")

#In[]
#the correlation between pressure sensors 
print(df.stat.corr("Pressure _sensor_1","Pressure _sensor_2"))
print(df.stat.corr("Pressure _sensor_1","Pressure _sensor_3"))
print(df.stat.corr("Pressure _sensor_1","Pressure _sensor_4"))
print("-------------------")
print(df.stat.corr("Pressure _sensor_2","Pressure _sensor_3"))
print(df.stat.corr("Pressure _sensor_2","Pressure _sensor_4"))
print("-------------------")
print(df.stat.corr("Pressure _sensor_3","Pressure _sensor_4"))

print("the correlation between sensor 2 and 4 is the highest")

#In[]
#the correlation between Vibration sensors 

print(df.stat.corr("Vibration_sensor_1","Vibration_sensor_2"))
print(df.stat.corr("Vibration_sensor_1","Vibration_sensor_3"))
print(df.stat.corr("Vibration_sensor_1","Vibration_sensor_4"))
print("----------------")
print(df.stat.corr("Vibration_sensor_2","Vibration_sensor_3"))
print(df.stat.corr("Vibration_sensor_2","Vibration_sensor_4"))
print("----------------")
print(df.stat.corr("Vibration_sensor_3","Vibration_sensor_4"))
print("----------------")

print("there is no strong correlation between vibration sensors ")


#In[]
#Section II: Classification & Big data analysis 

#Task 4 & 5 

from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# convert the Status column to a numerical values 
indexer = StringIndexer(inputCol="Status", outputCol="Status_index").fit(df)
df_ind = indexer.transform(df)
df_ind.select("Status_index").show(2)#test the output

#create assembler to add all features to single column 
assembler = VectorAssembler(inputCols = ['Power_range_sensor_1','Power_range_sensor_2','Power_range_sensor_3 ','Power_range_sensor_4','Pressure _sensor_1','Pressure _sensor_2','Pressure _sensor_3','Pressure _sensor_4','Vibration_sensor_1','Vibration_sensor_2','Vibration_sensor_3','Vibration_sensor_4'],outputCol="features")

output = assembler.transform(df_ind)


model_df = output.select("features","Status_index") #create a new dataframe to use it through the decision tree 
model_df.show(3,truncate = 50)

#In[]

#decision tree  


training_ds , testing_ds = model_df.randomSplit([0.7,0.3])#Task4 --- make a random split to model_df (the selected model for desision tree)

#In[]

#check the count of the training dataset
training_ds.count()
#In[]

#check the count of the test dataset

testing_ds.count()
#In[]

from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


#In[]
classifier_ds = DecisionTreeClassifier(labelCol= "Status_index").fit(training_ds) # define a dataset and fit the training data thourgh the model


#In[]
predictions_ds = classifier_ds.transform(testing_ds) #define a prediction dataset and using the trained data to predict the testing data 


#In[]
predictions_ds.show(100,truncate = 10)
# "Status_index","prediction"

#In[]
accuracy_ds = MulticlassClassificationEvaluator(
    labelCol = "Status_index" , metricName = "accuracy").evaluate(predictions_ds) #evaluate the accuracy 

accuracy_percentage = accuracy_ds * 100
print(accuracy_percentage,"%")

#In[]

precision_ds = MulticlassClassificationEvaluator(
    labelCol = "Status_index" , metricName = "weightedPrecision").evaluate(predictions_ds)

precision_percentage = precision_ds * 100
print(precision_percentage,"%")

#In[]
predictions_ds.groupby(['Status_index','prediction']).count().show()
print("Total ",predictions_ds.count())
classified_sample = 233
incorrect_classified_sample = 66
print("Incorrectly classified Samples ", 39+27)
print("classified Samples ", 122+111)
print("error rate = ", incorrect_classified_sample/classified_sample)

#In[]
#Task 5 (Support vector machine)

#SVM 

indexer = StringIndexer(inputCol="Status", outputCol="Status_index").fit(df) #using stringIndexer to convert Status to Numerical Values 
df_ind = indexer.transform(df) #Added to a new dataframe 
df_ind.select("Status_index").show(2)
df_pandas = df_ind.toPandas() #convert the dataframe to Pandas dataframe 

#In[]
#select the features and the predicated values and alocate them 
X = df_pandas.iloc[:, [1, 12]].values #
y = df_pandas.iloc[:, -1].values
df_pandas

#In[]
#split the data in another library (sklearn) 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#In[]
#Apply Standard Scaler 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#In[]
#apply SVM and fit the data
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

#In[]
y_pred = classifier.predict(X_test)

#In[]
#calculate the accuracy score and the error rate 
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
# accuracy_score(y_test,y_pred)
accuracy = accuracy_score(y_test,y_pred)*100

print("the accuracy of SVM = ",accuracy_score(y_test,y_pred))
error_rate = 1 - accuracy_score(y_test,y_pred)
print("the error of of SVM = ", error_rate)

#In[]
from sklearn.metrics import recall_score
# recall_sensitivity = recall_score(y_test, y_pred)
recall_specificity = recall_score(y_test, y_pred,pos_label=1)
recall_sensitivity = recall_score(y_test, y_pred,pos_label=0)

# recall_sensitivity, recall_specificity 
accuracy = accuracy_score(y_test, y_pred)
print(recall_sensitivity)
print(recall_specificity)



#In[]
#ANN with two different hidden layers 

from sklearn import neural_network
from sklearn.neural_network import MLPClassifier
import tensorflow as tf
#ANN

ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=6, activation='sigmoid'))
ann.add(tf.keras.layers.Dense(units=6, activation='sigmoid'))
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

ann.fit(X_train, y_train, batch_size = 32, epochs = 500)


#In[]
#ANN with two different hidden layers 


ann = neural_network.MLPClassifier(solver = 'sgd', activation = 'tanh', max_iter = 5000
                                   , hidden_layer_sizes = (50,50,50), alpha=0, random_state=0)
ann=ann.fit( X_train, y_train)
x_test_predict = ann.predict(X_test) 
accuracy = accuracy_score(y_test, x_test_predict) 
accuracy

#In[]
import pandas as pd
df_large_data = pd.read_csv('nuclear_plants_big_dataset.csv')
df_large_data.describe()
