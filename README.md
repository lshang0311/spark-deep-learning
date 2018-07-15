# spark-deep-learning
Examples of Deep Learning Pipelines for Apache Spark

Setup
=====
* Ubuntu 16.04.1
* Python 3.6.3
* [Spark 2.3.1](https://spark.apache.org/downloads.html)
* [Deep Learning Pipelines for Apache Spark](https://github.com/databricks/spark-deep-learning)
* [spark-deep-learning release 1.1.0-spark2.3-s2.11](https://spark-packages.org/package/databricks/spark-deep-learning)

Summary of Results
=================

| Example | Description  |  
|:------: |:---: |
| [Image classification](https://medium.com/linagora-engineering/making-image-classification-simple-with-spark-deep-learning-f654a8b876b8)  | Classify images of two persons (S. Jobs and M. Zuckerberg)  | 

Example Output
=============

 * Start pyspark
```
export SPARK_HOME=/home/lshang/Downloads/spark-2.3.1-bin-hadoop2.7
export set JAVA_OPTS="-Xmx9G -XX:MaxPermSize=2G -XX:+UseCompressedOops -XX:MaxMetaspaceSize=512m"export set JAVA_OPTS="-Xmx9G -XX:MaxPermSize=2G -XX:+UseCompressedOops -XX:MaxMetaspaceSize=512m"export set JAVA_OPTS="-Xmx9G -XX:MaxPermSize=2G -XX:+UseCompressedOops -XX:MaxMetaspaceSize=512m"export set JAVA_OPTS="-Xmx9G -XX:MaxPermSize=2G -XX:+UseCompressedOops -XX:MaxMetaspaceSize=512m"export set JAVA_OPTS="-Xmx9G -XX:MaxPermSize=2G -XX:+UseCompressedOops -XX:MaxMetaspaceSize=512m"export set JAVA_OPTS="-Xmx9G -XX:MaxPermSize=2G -XX:+UseCompressedOops -XX:MaxMetaspaceSize=512m"
$SPARK_HOME/bin/pyspark --packages databricks:spark-deep-learning:1.1.0-spark2.3-s_2.11 --driver-memory 5g
```
    
    Console output:
```
   Welcome to
      ____              __
     / __/__  ___ _____/ /__
    _\ \/ _ \/ _ `/ __/  '_/
   /__ / .__/\_,_/_/ /_/\_\   version 2.3.1
      /_/

Using Python version 3.6.3 (default, Oct 13 2017 12:02:49)
SparkSession available as 'spark'.
>>> 
```

 * Run [code](https://github.com/lshang0311/spark-deep-learning/blob/master/pyspark_shell_classify_images.py) in shell.
```
 ...
 
>>> predictionAndLabels = df.select("prediction", "label")
>>> evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
>>> print("Training set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))
[Stage 168:===================================================>   (15 + 1) / 16]Using TensorFlow backend.
Training set accuracy = 0.5714285714285714   >>> predictionAndLabels = df.select("prediction", "label")
>>> evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
>>> print("Training set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))
[Stage 168:===================================================>   (15 + 1) / 16]Using TensorFlow backend.
Training set accuracy = 0.5714285714285714   
from pyspark.sql.functions import lit
from sparkdl.image import imageIO
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from sparkdl import DeepImageFeaturizer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

img_dir = "/home/lshang/Downloads/personalities"
jobs_df = imageIO.readImagesWithCustomFn(img_dir + "/jobs", decode_f=imageIO.PIL_decode).withColumn("label", lit(1))
zuckerberg_df = imageIO.readImagesWithCustomFn(img_dir + "/zuckerberg", decode_f=imageIO.PIL_decode).withColumn("label", lit(0))

jobs_train, jobs_test = jobs_df.randomSplit([0.6, 0.4])
zuckerberg_train, zuckerberg_test = zuckerberg_df.randomSplit([0.6, 0.4])

#dataframe for training a classification model
train_df = jobs_train.unionAll(zuckerberg_train)

#dataframe for testing the classification model
test_df = jobs_test.unionAll(zuckerberg_test)

featurizer = DeepImageFeaturizer(inputCol="image", outputCol="features", modelName="InceptionV3")
lr = LogisticRegression(maxIter=20, regParam=0.05, elasticNetParam=0.3, labelCol="label")
p = Pipeline(stages=[featurizer, lr])
p_model = p.fit(train_df)

predictions = p_model.transform(test_df)

predictions.select("image", "prediction").show()

df = p_model.transform(test_df)
df.show()

predictionAndLabels = df.select("prediction", "label")
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
print("Training set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))```

