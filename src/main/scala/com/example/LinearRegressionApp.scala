// LinearRegressionApp.scala
package com.example

import breeze.linalg._
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
//import breeze.numerics.pinv // Explicitly import pinv from breeze.numerics

object LinearRegressionApp {

  def main(args: Array[String]): Unit = {

    // Initialize SparkSession for local execution
    implicit val spark: SparkSession = SparkSession.builder
      .appName("LocalLinearRegressionWithBreeze")
      .master("local[*]") // Use all available cores on the local machine
      .getOrCreate()

    import spark.implicits._

    // Linear regression using closed-form solution and outer product for X^T X
    def linearRegressionClosedForm_OuterProduct(X: DataFrame, y: DataFrame)(implicit spark: SparkSession): DenseVector[Double] = {
      import spark.implicits._

      X.createOrReplaceTempView("X")
      y.createOrReplaceTempView("y")

      // Aggregate values row-wise into vector (j, value) pairs
      val rowVectors = spark.sql(
        """
          SELECT rowIndex, collect_list(array(colIndex, value)) AS rowVec
          FROM X
          GROUP BY rowIndex
        """
      ).createOrReplaceTempView("rowVectors")

      val outerProductUDF = udf { row: Seq[Seq[Double]] =>
        row.flatMap { v1 =>
          val i = v1(0).toInt
          val val1 = v1(1)
          row.map { v2 =>
            val j = v2(0).toInt
            val val2 = v2(1)
            ((i, j), val1 * val2)
          }
        }
      }

      spark.udf.register("outerProductUDF", outerProductUDF)

      val xtx = spark.sql(
        """
          SELECT 
            outerProduct._1._1 AS i,
            outerProduct._1._2 AS j,
            SUM(outerProduct._2) AS value
          FROM rowVectors
          LATERAL VIEW explode(outerProductUDF(rowVec)) AS outerProduct
          GROUP BY outerProduct._1._1, outerProduct._1._2
          ORDER BY i, j
        """
      )

      xtx.createOrReplaceTempView("xtx")

      val xtxArray = xtx.collect()
      
      // Handle potential empty DataFrame for maxIndex
      val maxIndexResult = spark.sql("SELECT GREATEST(IFNULL(MAX(i), -1), IFNULL(MAX(j), -1)) AS max_index FROM xtx").first()
      val maxIndex = if (maxIndexResult != null && !maxIndexResult.isNullAt(0)) maxIndexResult.getInt(0) else -1

      if (maxIndex == -1) {
          println("Warning: XTX matrix is empty or has no valid indices, returning empty weights.")
          // Handle this case appropriately, e.g., return an empty vector or throw an error
          return DenseVector.zeros[Double](0) // Return an empty vector
      }

      val xtxMatrix = DenseMatrix.zeros[Double](maxIndex + 1, maxIndex + 1)

      xtxArray.foreach { row =>
        val i = row.getAs[Int]("i")
        val j = row.getAs[Int]("j")
        val value = row.getAs[Double]("value")
        xtxMatrix(i, j) = value
      }

      val XTX_inv = pinv(xtxMatrix) // pinv is imported from breeze.numerics

      // Compute X^Ty
      spark.sql(
        """
          SELECT X.colIndex AS i, SUM(X.value * y.value) AS value
          FROM X JOIN y ON X.rowIndex = y.rowIndex
          GROUP BY X.colIndex
        """
      ).createOrReplaceTempView("XTy")

      val xtyArray = spark.sql("SELECT * FROM XTy").collect()
      val xtyVector = DenseVector.zeros[Double](maxIndex + 1)

      xtyArray.foreach { row =>
        val i = row.getAs[Long]("i").toInt // Ensure correct type casting if 'i' is Long in SQL
        val value = row.getAs[Double]("value")
        xtyVector(i) = value
      }

      val weights = XTX_inv * xtyVector

      weights
    }


    // --- Main application logic starts here ---

    // sample X and y values
    val X = Seq((0L, 0L, 1.0), (0L, 1L, 2.0),(1L, 0L, 3.0), (1L, 1L, 4.0)).toDF("rowIndex", "colIndex", "value")
    val y = Seq((0L, 4.0), (1L, 5.0)).toDF("rowIndex", "value")

    // Partitioning in the beginning before the outer product calculation to avoid further shuffles
    val X_partitioned = X.repartition($"rowIndex")

    // Call your linear regression function
    val weights = linearRegressionClosedForm_OuterProduct(X_partitioned, y)(spark)
    
    println("------------------------------------")
    println("Final Calculated Weights:")
    println(weights)
    println("------------------------------------")

    // Stop the SparkSession when done
    spark.stop()
  }
}