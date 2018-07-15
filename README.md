# spark-deep-learning
Examples of Deep Learning for Apache Spark

Setup
=====
* Ubuntu 16.04.1
* Spark 2.3.1
* Python 3.6.3
* [Deep Learning Pipelines for Apache Spark 1.1.0-spark2.3-s2.11](https://spark-packages.org/package/databricks/spark-deep-learning)

Summary of Results
=================
| Example | Description  |  
|:------: |:---: |
| [Image classification](https://medium.com/linagora-engineering/making-image-classification-simple-with-spark-deep-learning-f654a8b876b8)  | Classify images of two persons (S. Jobs and M. Zuckerberg)  | 

Example Output
=============

 1) Start pyspark
    ```
    $SPARK_HOME/bin/pyspark --packages databricks:spark-deep-learning:0.1.0-spark2.1-s_2.11 --driver-memory 5g
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

