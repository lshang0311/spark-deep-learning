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
 ```

