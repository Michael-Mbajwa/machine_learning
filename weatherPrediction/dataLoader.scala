import org.apache.spark.rdd._
import org.apache.spark.sql.types._
import org.apache.spark.sql.Row
import java.io.{BufferedWriter, FileWriter, File}

val schema = StructType(
    Array(
        StructField("year", IntegerType, false),
        StructField("month", IntegerType, false),
        StructField("day", IntegerType, false),
        StructField("hour",IntegerType, false),
        StructField("latitude", DoubleType, false),
        StructField("longitude", DoubleType, false),
        StructField("elevationDimension", IntegerType, false),
        StructField("directionAngle", IntegerType, false),
        StructField("speedRate", DoubleType, false),
        StructField("ceilingHeightDimension", IntegerType, false),
        StructField("distanceDimension", IntegerType, false),
        StructField("dewPointTemperature", DoubleType, false),
        StructField("airTemperature", DoubleType, false)
    )
)

def parseNOAA(rawData: RDD[String]) = {
    rawData
      .filter(line => line.substring(87, 92) != "+9999") // filter out missing temperature labels
      .map { line =>
          val year = line.substring(15, 19).toInt
          val month = line.substring(19, 21).toInt
          val day = line.substring(21, 23).toInt
          val hour = line.substring(23, 25).toInt
          val latitude = line.substring(28, 34).toDouble / 1000
          val longitude = line.substring(34, 41).toDouble / 1000
          val elevationDimension = line.substring(46, 51).toInt
          val directionAngle = line.substring(60, 63).toInt
          val speedRate = line.substring(65, 69).toDouble / 10
          val ceilingHeightDimension = line.substring(70, 75).toInt
          val distanceDimension = line.substring(78, 84).toInt
          val dewPointTemperature = line.substring(93, 98).toDouble / 10
          val airTemperature = line.substring(87, 92).toDouble / 10

          Row(year, month, day, hour, latitude, longitude, elevationDimension, directionAngle, speedRate, ceilingHeightDimension, distanceDimension, dewPointTemperature, airTemperature)
      }
}

val rawNOAA = sc.textFile("/home/users/mmbajwa/NOAA-065900/065900*")

val parsedNOAA = parseNOAA(rawNOAA)

// Transform RDD to a dataframe
val df = spark.createDataFrame(parsedNOAA, schema)

//save the dataframe
df.write.format("csv").option("header", true).save("/home/users/mmbajwa/problem2A_data.csv")