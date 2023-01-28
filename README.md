## Run

### python
```bash
python main.py
```

### spark-submit
```bash
$ ./bin/spark-submit ~/main.py
```


```python
import os
os.environ["JAVA_HOME"] = "/home/opt/java8"
os.environ["SPARK_HOME"] = "/home/opt/spark-3.3.1-bin-hadoop3"

from pyspark.ml.recommendation import ALSModel
from pyspark.sql.functions import col, explode
from pyspark.sql import SparkSession
import pyspark


class prediction:
    def __init__(self, pathModel: str, output: str):
        self.spark = SparkSession.builder\
            .master("local[*]")\
            .appName("prediction")\
            .config("spark.driver.memory", "1g")\
            .config("spark.ui.port", "4050")\
            .getOrCreate()

        self.model = ALSModel.load(
            pathModel
        )

        self.output = output

    def dataFrameWriter(self, df: pyspark.sql.dataframe.DataFrame, type : str):
        df.coalesce(1)\
            .write.mode("overwrite")\
            .csv(f"{self.output}/output_csv_{type}", header="false")

        df.coalesce(1)\
            .write.mode("overwrite")\
            .json(f"{self.output}/output_json_{type}")

        df.coalesce(1)\
            .write.mode("overwrite")\
            .parquet(f"{self.output}/output_parquet_{type}")

        path = self.output+"/output_easy_json_"+type+".json"
        df.toPandas()\
            .to_json(orient="records", path_or_buf=path)

    def formatter(self, df: pyspark.sql.dataframe.DataFrame) -> pyspark.sql.dataframe.DataFrame:
        return df.withColumn("r", explode("recommendations"))\
            .select("user_id", col("r.movie_id"), col("r.rating"))

    def recommendForAllUsers(self, topN: int):
        df = self.model.recommendForAllUsers(topN)
        print(df.show())
        self.dataFrameWriter(
            self.formatter(df),
            "all_user"
        )

    def recommendForUser(self, userId, topN: int):
        user = self.spark.createDataFrame([([topN])], ["user_id"])
        df = self.model.recommendForUserSubset(user, topN)
        print(df.show())
        self.dataFrameWriter(
            self.formatter(df),
            "user"
        )

    def stopSpark(self):
        self.spark.sparkContext.stop()
        self.spark.stop()


if __name__ == "__main__":
    p = prediction(
        "./ranker_best_spark331", ".")
    p.recommendForAllUsers(5)
    p.recommendForUser(5, 5)
    p.stopSpark()
```
