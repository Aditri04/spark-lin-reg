// LinearRegressionApp.scala
package com.example

import breeze.linalg._
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.storage.StorageLevel
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._


object LinearRegressionApp {

  def main(args: Array[String]): Unit = {

    def initializa_spark() = {
       SparkSession.builder
      .appName("LocalLinearRegressionWithBreeze")
      .master("local[*]") // Use all available cores on the local machine
      .getOrCreate()
    }

    def linearRegressionClosedForm_InnerProduct(X: DataFrame, y: DataFrame) (implicit spark: SparkSession): DenseVector[Double] = {
      X.createOrReplaceTempView("X")
      y.createOrReplaceTempView("y")

      val row_count = y.count().toInt //returns a Long type //check slides for dims
      val col_cnt = spark.sql("""Select count(*) from X group by rowIndex limit 1""").first().getLong(0).toInt
      
     
      // val xt = spark.sql("""
      // Select x.colIndex as rowIndex,
      //       x.rowIndex as colIndex,
      //       x.value as value
      //       FROM X x
      // """)
      // xt.show()
      // xt.createOrReplaceTempView("XT")

      

      // val xtx = spark.sql(
      //   """
      //     SELECT
      //       xt.rowIndex AS i,
      //       x.colIndex AS j,
      //       SUM(xt.value * x.value) AS value
      //     FROM XT xt INNER JOIN X x  on xt.colIndex = x.rowIndex
      //     GROUP BY  x.colIndex, xt.rowIndex
          
      //   """
      // )

      // xtx.show()

      val xtx = spark.sql(
        """
          SELECT
            xt.colIndex AS i,
            x.colIndex AS j,
            SUM(x.value * xt.value) AS value
          FROM X x INNER JOIN X xt on x.rowIndex = xt.rowIndex 
          GROUP BY x.colIndex, xt.colIndex
          
        """
      )
      xtx.createOrReplaceTempView("xtx")

      val xtxArray = xtx.collect() //brings to driver memory


      val xtxMatrix = DenseMatrix.zeros[Double](col_cnt + 1, col_cnt + 1)
      xtxArray.foreach { row =>
        val i = row.getAs[Long]("i").toInt
        val j = row.getAs[Long]("j").toInt
        val value = row.getAs[Double]("value")
        xtxMatrix(i, j) = value
      }
      //computes the (XTX)-1
      val XTX_inv = pinv(xtxMatrix)

      val xty = spark.sql(
          """
            SELECT X.colIndex AS i, SUM(X.value * y.value) AS value
            FROM X JOIN y ON X.rowIndex = y.rowIndex
            GROUP BY X.colIndex
          """).collect()

      // xty.createOrReplaceTempView("XTy")
      
      // val xtyArray = spark.sql("SELECT * FROM XTy").collect() //driver mem
      val xtyVector = DenseVector.zeros[Double](col_cnt + 1)
      
      xty.foreach { row =>
        val i = row.getAs[Long]("i").toInt
        val value = row.getAs[Double]("value")
        xtyVector(i) = value
      }

      val weights = XTX_inv * xtyVector

      weights

    }
    
    // Linear regression using closed-form solution and outer product for X^T X
    def linearRegressionClosedForm_OuterProduct(X: DataFrame, y: DataFrame)(implicit spark: SparkSession): DenseVector[Double] = {
      

      X.createOrReplaceTempView("X")
      y.createOrReplaceTempView("y")

      // Aggregate values row-wise into vector (j, value) pairs
      // collect_list bring to driver mem?? separate .createOR
      val rowVectors = spark.sql(
        """
          SELECT rowIndex, collect_list(array(colIndex, value)) AS rowVec
          FROM X
          GROUP BY rowIndex
        """
      )
      
      rowVectors.createOrReplaceTempView("rowVectors")


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

      //seq-> list, 

      // Compute X^Ty
      val xty = spark.sql(
        """
          SELECT X.colIndex AS i, SUM(X.value * y.value) AS value
          FROM X JOIN y ON X.rowIndex = y.rowIndex
          GROUP BY X.colIndex
        """)

        xty.createOrReplaceTempView("XTy")
      
      

      val xtxArray = xtx.collect() //brings to driver memory


      val maxIndex = spark.sql("SELECT COUNT(DISTINCT i) AS num_rows FROM xtx").first().getLong(0).toInt
      //val maxIndex = spark.sql("SELECT GREATEST(MAX(i), MAX(j)) AS max_index FROM xtx").first().getInt(0)
      val xtxMatrix = DenseMatrix.zeros[Double](maxIndex + 1, maxIndex + 1)

      xtxArray.foreach { row =>
        val i = row.getAs[Int]("i")
        val j = row.getAs[Int]("j")
        val value = row.getAs[Double]("value")
        xtxMatrix(i, j) = value
      }


      //computes the (XTX)-1
      val XTX_inv = pinv(xtxMatrix)

      
      val xtyArray = spark.sql("SELECT * FROM XTy").collect() //driver mem
      val xtyVector = DenseVector.zeros[Double](maxIndex + 1)
      
      xtyArray.foreach { row =>
        val i = row.getAs[Long]("i").toInt
        val value = row.getAs[Double]("value")
        xtyVector(i) = value
      }

      val weights = XTX_inv * xtyVector


      weights
    }

    // def linearRegressionGradientDescent(X: DataFrame, y: DataFrame, numIters: Int, lr: Double) (implicit spark: SparkSession): DenseVector[Double] = {
    //     val row_count = y.count().toInt
    //     val feature_count = X.count(1)
    //     val weights = DenseVector.zeroes[Double](feature_count+1)


    //     for (_ <- 1 to numIters) {
          
    //     }

    //    weights

    // }


    
    implicit val spark: SparkSession = initializa_spark()
    import spark.implicits._


    val X = List((0L, 0L, 9.0), (0L, 1L, 3.0),(0L, 2L, 5.0),(1L, 0L, 4.0), (1L, 1L, 1.0),  (1L, 2L, 2.0)).toDF("rowIndex", "colIndex", "value")
    val y = List((0L, 4.0), (1L, 5.0)).toDF("rowIndex", "value")

    val X_partitioned = X.repartition($"rowIndex")

    //Analysis
    // partition not help inner product 

    var t1 = 0.0
    var t2 = 0.0


    // t1 = System.nanoTime
    // val weights_innerProduct = linearRegressionClosedForm_InnerProduct(X, y)(spark)
    // t2 =(System.nanoTime - t1) / 1e9d
    
    // println("------------------------------------")
    // println("Final Calculated Weights using Inner Product without input partitioned:")
    // println(weights_innerProduct)
    
    // println("Time taken")
    // println(t2)
    // println("------------------------------------")

    

    t1 = System.nanoTime
    val weights_innerProduct_partitioned = linearRegressionClosedForm_InnerProduct(X_partitioned, y)(spark)
    t2 =(System.nanoTime - t1) / 1e9d
    
    println("------------------------------------")
    println("Final Calculated Weights using inner product with input partitioned:")
    println(weights_innerProduct_partitioned)
    println("Time taken")
    println(t2)
    println("------------------------------------")

    // t1 = System.nanoTime
    // val weights_outerProduct = linearRegressionClosedForm_OuterProduct(X, y)(spark)
    // t2 =(System.nanoTime - t1) / 1e9d
    
    // println("------------------------------------")
    // println("Final Calculated Weights using Outer product without input partitioned:")
    // println(weights_outerProduct)
    // println("Time taken")
    // println(t2)
    // println("------------------------------------")

    t1 = System.nanoTime
    val weights_outerProduct_partitioned = linearRegressionClosedForm_OuterProduct(X_partitioned, y)(spark)
    t2 =(System.nanoTime - t1) / 1e9d
    
    println("------------------------------------")
    println("Final Calculated Weights using Outer product with input partitioned:")
    println(weights_outerProduct_partitioned)
    println("Time taken")
    println(t2)
    println("------------------------------------")
    

    // val weights_gradientDescent = linearRegressionGradientDescent(X_partitioned, y, 100, 0.01)(spark)
    
    // println("------------------------------------")
    // println("Final Calculated Weights using gradient descent:")
    // println(weights_gradientDescent)
    // println("------------------------------------")

    // Stop the SparkSession when done
    spark.stop()
  }
}