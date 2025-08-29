// LinearRegressionApp.scala
package com.example

import breeze.linalg._
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.storage.StorageLevel
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.sql.Row
import dev.ludovic.netlib.blas.BLAS
import dev.ludovic.netlib.lapack.LAPACK

import dev.ludovic.netlib.arpack.ARPACK



object LinearRegressionApp {

  def main(args: Array[String]): Unit = {

    println(BLAS.getInstance.getClass)
    println(LAPACK.getInstance.getClass)


    def linearRegressionClosedForm_InnerProduct(X: DataFrame, y: DataFrame, col_cnt: Int, spark: SparkSession) : DenseVector[Double] = {
      X.createOrReplaceTempView("X")
      y.createOrReplaceTempView("y")

      val row_count = y.count().toInt //returns a Long type //check slides for dims
      
      
     
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


      val xtxMatrix = DenseMatrix.zeros[Double](col_cnt, col_cnt)
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
      val xtyVector = DenseVector.zeros[Double](col_cnt)

      
      xty.foreach { row =>
        val i = row.getAs[Long]("i").toInt
        val value = row.getAs[Double]("value")
        xtyVector(i) = value
      }

      val weights = XTX_inv * xtyVector

      weights

    }
    
    // Linear regression using closed-form solution and outer product for X^T X
    def linearRegressionClosedForm_OuterProduct(X: DataFrame, y: DataFrame, maxIndex: Int, spark: SparkSession): DenseVector[Double] = {

      X.createOrReplaceTempView("X")
      y.createOrReplaceTempView("y")

      // Aggregate values row-wise into vector (j, value) pairs
     
      val rowVectors = spark.sql("SELECT rowIndex, collect_list(array(colIndex, value)) AS rowVec FROM X GROUP BY rowIndex")
      
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
       xtx.show()
      
      xtx.createOrReplaceTempView("xtx")
 

      // Compute X^Ty
      val xty = spark.sql(
        """
          SELECT X.colIndex AS i, SUM(X.value * y.value) AS value
          FROM X JOIN y ON X.rowIndex = y.rowIndex
          GROUP BY X.colIndex
        """)

      xty.createOrReplaceTempView("XTy")


      val xtxArray = xtx.collect() //brings to driver memory

      
      val xtxMatrix = DenseMatrix.zeros[Double](maxIndex , maxIndex )

      xtxArray.foreach { row =>
        val i = row.getAs[Int]("i")
        val j = row.getAs[Int]("j")
        val value = row.getAs[Double]("value")
        xtxMatrix(i, j) = value
      }



      //computes the (XTX)-1
      val XTX_inv = pinv(xtxMatrix)

      
      val xtyArray = spark.sql("SELECT * FROM XTy").collect() //driver mem
      val xtyVector = DenseVector.zeros[Double](maxIndex)
      
      xtyArray.foreach { row =>
        val i = row.getAs[Long]("i").toInt
        val value = row.getAs[Double]("value")
        xtyVector(i) = value
      }

      val weights = XTX_inv * xtyVector

      weights
    }

    def test(X: DataFrame, spark: SparkSession) = {

      val br = 2
      val bc = 2

      // spark.conf.set("app.block.rows", br)
      // spark.conf.set("app.block.cols", bc)

      X.createOrReplaceTempView("X")

      val xb = spark.sql(s"""
          SELECT
            rowIndex  AS r,
            colIndex AS c,
            value    AS v,
          FLOOR(rowIndex / ${br})  AS rbid,  
          FLOOR(colIndex / ${bc}) AS cbid,   
          MOD(colIndex, ${bc})   AS coff     
          FROM X
        """)
        xb.show()

        xb.createOrReplaceTempView("Xb")

        val a = spark.sql("SELECT r,rbid, cbid AS bi, coff AS ci, v AS vi FROM Xb")
        a.show()
        a.createOrReplaceTempView("A")
        

        val b = spark.sql("SELECT r,rbid,  cbid AS bj, coff AS cj, v AS vj FROM Xb")
        b.show()
        b.createOrReplaceTempView("B")

        val temp = spark.sql("""
        SELECT
             *
          FROM A JOIN B
           ON A.rbid = B.rbid
            AND A.r = B.r
          
        """)
        temp.show()
        
        val xtx_blocks = spark.sql("""
          SELECT
             A.rbid,
            A.bi AS bi, B.bj AS bj,
            A.ci AS ci, B.cj AS cj,
            SUM(A.vi * B.vj) AS value
          FROM A JOIN B
           ON A.rbid = B.rbid
            AND A.r = B.r
          GROUP BY A.rbid, A.bi, A.ci, B.bj, B.cj
        """)
        xtx_blocks.show()
        xtx_blocks.createOrReplaceTempView("xtx_blocks")

       val xtx_block_sum = spark.sql("""
            SELECT
              bi, bj, ci, cj,
              SUM(value) AS value
            FROM xtx_blocks
            GROUP BY bi, bj, ci, cj
          """)

          xtx_block_sum.show()
          xtx_block_sum.createOrReplaceTempView("xtx_block_sum")


          val xtx = spark.sql(s"""
            SELECT CAST((bi*${bc} + ci) AS INT) AS i, CAST((bj*${bc} + cj) AS INT) AS j, value
            FROM xtx_block_sum
          """)
          xtx.show()
          xtx.createOrReplaceTempView("xtx")
      
    }

    def linearRegressionClosedFormBlock_OuterProduct(X: DataFrame, y: DataFrame, maxIndex: Int, spark: SparkSession): DenseVector[Double] = {
        val br = 2
        val bc = 2


        X.createOrReplaceTempView("X")
        y.createOrReplaceTempView("y")

        val xb = spark.sql(s"""
          SELECT
            rowIndex  AS r,
            colIndex AS c,
            value    AS v,
          FLOOR(rowIndex / ${br})  AS rbid,  
          FLOOR(colIndex / ${bc}) AS cbid,   
          MOD(colIndex, ${bc})   AS coff     
          FROM X
        """)
        xb.show()

        xb.createOrReplaceTempView("Xb")

        val a = spark.sql("SELECT r,rbid,  cbid AS bi, coff AS ci, v AS vi FROM Xb")
        
        a.createOrReplaceTempView("A")
        

        val b = spark.sql("SELECT r,rbid,  cbid AS bj, coff AS cj, v AS vj FROM Xb")
        
        b.createOrReplaceTempView("B")
        
        val xtx_blocks = spark.sql("""
          SELECT
             A.rbid,
            A.bi AS bi, B.bj AS bj,
            A.ci AS ci, B.cj AS cj,
            SUM(A.vi * B.vj) AS value
          FROM A JOIN B
           ON A.rbid = B.rbid
            AND A.r = B.r
          GROUP BY A.rbid, A.bi, A.ci, B.bj, B.cj
        """)
        
        xtx_blocks.createOrReplaceTempView("xtx_blocks")

       val xtx_block_sum = spark.sql("""
            SELECT
              bi, bj, ci, cj,
              SUM(value) AS value
            FROM xtx_blocks
            GROUP BY bi, bj, ci, cj
          """)

          
          xtx_block_sum.createOrReplaceTempView("xtx_block_sum")


          val xtx = spark.sql(s"""
            SELECT CAST((bi*${bc} + ci) AS INT) AS i, CAST((bj*${bc} + cj) AS INT) AS j, value
            FROM xtx_block_sum
          """)
        


        val xty = spark.sql(
        """
          SELECT X.colIndex AS i, SUM(X.value * y.value) AS value
          FROM X JOIN y ON X.rowIndex = y.rowIndex
          GROUP BY X.colIndex
        """)

      xty.createOrReplaceTempView("XTy")


      val xtxArray = xtx.collect() //brings to driver memory

      
      val xtxMatrix = DenseMatrix.zeros[Double](maxIndex , maxIndex )

      xtxArray.foreach { row =>
        val i = row.getAs[Int]("i")
        val j = row.getAs[Int]("j")
        val value = row.getAs[Double]("value")
        xtxMatrix(i, j) = value
      }



      //computes the (XTX)-1
      val XTX_inv = pinv(xtxMatrix)

      
      val xtyArray = spark.sql("SELECT * FROM XTy").collect() //driver mem
      val xtyVector = DenseVector.zeros[Double](maxIndex)
      
      xtyArray.foreach { row =>
        val i = row.getAs[Long]("i").toInt
        val value = row.getAs[Double]("value")
        xtyVector(i) = value
      }

      val weights = XTX_inv * xtyVector

      weights

  


    }

    def libsvmToCOO(training: DataFrame, spark: SparkSession): (DataFrame, DataFrame) = {
      import spark.implicits._


      val indexed = training.rdd.zipWithIndex()

      
      val combinedRDD = indexed.map { case (row: Row, rowIndex: Long) =>
        val label = row.getAs[Double]("label")
        val features = row.getAs[SparseVector]("features")

        val shiftedTriplets = features.indices.zip(features.values).map {
          case (colIndex, value) => (rowIndex, colIndex.toLong + 1L, value)
        }

        (rowIndex, label, shiftedTriplets)
      }
      
      val y = combinedRDD.map { case (rowIndex, label, _) =>
        (rowIndex, label)
      }.toDF("rowIndex", "value")

      
      val biasRDD = combinedRDD.map { case (rowIndex, _, _) =>
        (rowIndex, 0L, 1.0)
      }

      val featuresRDD = combinedRDD.flatMap { case (_, _, triplets) => triplets }

      val fullFeatureRDD = biasRDD.union(featuresRDD)

      
      val X = fullFeatureRDD.toDF("rowIndex", "colIndex", "value")

      (X, y)
}
    
    implicit val spark: SparkSession =  SparkSession.builder
      .appName("LocalLinearRegressionWithBreeze")
      //.master("local-cluster[3,1,1024]") 
      .master("local[*]")
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")
    import spark.implicits._

    var t1 = 0.0
    var t2 = 0.0
    
    //val X = List( (0L, 0L, 1.0), (1L, 0L, 1.0),(2L, 0L, 1.0),(3L, 0L, 1.0), (4L, 0L, 1.0), (0L, 1L, 5.0), (1L, 1L, 7.0),(2L, 1L, 12.0),(3L, 1L, 16.0), (4L, 1L, 20.0)).toDF("rowIndex", "colIndex", "value")
    // val y = List((0L, 40.0), (1L, 120.0), (2L, 180.0), (3L, 210.0), (4L, 240.0)).toDF("rowIndex", "value")

    
    // val X_partitioned = X.repartition($"rowIndex")
    
    // val data = List(
    // (40.0, Vectors.dense(5.0)),
    // (120.0, Vectors.dense(7.0)),
    // (180.0, Vectors.dense(12.0)),
    // (210.0, Vectors.dense(16.0)),
    // (240.0, Vectors.dense(20.0))
    // ).toDF("label", "features")

    val training = spark.read.format("libsvm").load("sample_linear_regression_data.txt")

    

    // val lr = new LinearRegression().setSolver("normal")

    // println("parameters of model:")
    // lr.extractParamMap().toSeq.foreach { paramPair =>
    //   println(s"${paramPair.param.name}: ${paramPair.value}")
    // } 
    // t1 = System.nanoTime
    // val model = lr.fit(training)
    // t2 =(System.nanoTime - t1) / 1e9d
    // val coefficients = model.coefficients
    // val intercept = model.intercept
    // val summary = model.summary

    // println("------------------------------------")
    // println("Final Calculated Weights using ml linear regression library:")
    // println("coefficient", coefficients)
    // println("intercept", intercept)
    // println("MSE", summary.meanSquaredError)
    // println("Time taken", t2)
    // println("------------------------------------")

    
    
    val (x, y) = libsvmToCOO(training, spark)
    
    //val  x = List( (0L, 0L, 1.0), (0L, 1L, 0.0),(0L, 2L, 2.0),(0L, 3L, 0.0), (1L, 0L, 0.0), (1L, 1L, 3.0),(1L, 2L, 0.0),(1L, 3L, 4.0), (2L, 0L, 5.0), (2L, 1L, 0.0),(2L, 2L, 6.0),(2L, 3L, 0.0),(3L, 0L, 0.0), (3L, 1L, 7.0),(3L, 2L, 0.0),(3L, 3L, 8.0)).toDF("rowIndex", "colIndex", "value")
    //test(x, spark)


    //do not include
    //val X_partitioned = x.repartition($"rowIndex")
    
    
    // t1 = System.nanoTime
    // val weights_innerProduct = linearRegressionClosedForm_InnerProduct(X, y)(spark)
    // t2 =(System.nanoTime - t1) / 1e9d
    
    // println("------------------------------------")
    // println("Final Calculated Weights using Inner Product without input partitioned:")
    // println(weights_innerProduct)
    // println("Time taken", t2)
    // println("------------------------------------")


    // t1 = System.nanoTime
    // val weights_innerProduct_partitioned = linearRegressionClosedForm_InnerProduct(x, y, 11, spark)
    // t2 =(System.nanoTime - t1) / 1e9d
    
    // println("------------------------------------")
    // println("Final Calculated Weights using inner product")
    // println(weights_innerProduct_partitioned)
    // println("Time taken", t2)
    
    // println("------------------------------------")

    // t1 = System.nanoTime
    // val weights_outerProduct = linearRegressionClosedForm_OuterProduct(X, y)(spark)
    // t2 =(System.nanoTime - t1) / 1e9d
    
    // println("------------------------------------")
    // println("Final Calculated Weights using Outer product without input partitioned:")
    // println(weights_outerProduct)
    // println("Time taken", t2)
    // println("------------------------------------")

    // t1 = System.nanoTime
    // val weights_outerProduct_partitioned = linearRegressionClosedForm_OuterProduct(x, y, 11, spark)
    // t2 =(System.nanoTime - t1) / 1e9d
    
    // println("------------------------------------")
    // println("Final Calculated Weights using Outer product")
    // println(weights_outerProduct_partitioned)
    // println("Time taken", t2)
    // println("------------------------------------")


    t1 = System.nanoTime
    val weights_outerBlockProduct = linearRegressionClosedFormBlock_OuterProduct(x, y, 11, spark)
    t2 =(System.nanoTime - t1) / 1e9d
    println("------------------------------------")
    println("Final Calculated Weights using Block Outer product")
    println(weights_outerBlockProduct)
    println("Time taken ", t2)
    println("------------------------------------")
   
    
    
    //input to keep program running to see output in sparkUI http://127.0.0.1:4040/
    spark.sparkContext.setLogLevel("INFO")
    println("Spark is running. Press ENTER to exit.")
    scala.io.StdIn.readLine()


    
    
    spark.stop()
  }
}