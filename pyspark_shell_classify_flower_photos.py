from pyspark.sql.functions import lit
from sparkdl.image import imageIO
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from sparkdl import DeepImageFeaturizer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

img_dir = "/home/lshang/Downloads/flower_photos"
tulips_df = imageIO.readImagesWithCustomFn(img_dir + "/tulips", decode_f=imageIO.PIL_decode).withColumn("label", lit(1))
daisy_df = imageIO.readImagesWithCustomFn(img_dir + "/daisy", decode_f=imageIO.PIL_decode).withColumn("label", lit(0))

tulips_train, tulips_test = tulips_df.randomSplit([0.6, 0.4])
daisy_train, daisy_test = daisy_df.randomSplit([0.6, 0.4])

#dataframe for training a classification model
train_df = tulips_train.unionAll(daisy_train)

#dataframe for testing the classification model
test_df = tulips_test.unionAll(daisy_test)

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
print("Training set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))