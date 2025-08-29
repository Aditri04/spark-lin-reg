
# File Structure
src/main/scala/com/example contain the main class
build.sbt - contains dependencies and defines the location and name of main class
project/plugins.sbt - optional; used to create one single fat jar file

# Local setup
Installations
1. openjdk17 
2. apache-spark
3. sbt (scala build tool)

## To Run locally, run the following commands

sbt clean compile
sbt assembly
spark-submit --class com.example.LinearRegressionApp target/scala-2.13/SparkLinearRegression-assembly-1.0.jar

--conf "spark.driver.extraJavaOptions=-Ddev.ludovic.netlib.blas.nativeLibPath=/opt/homebrew/opt/openblas/lib/libopenblas.dylib -Ddev.ludovic.netlib.lapack.nativeLibPath=/opt/homebrew/opt/lapack/lib/liblapack.dylib"  
--conf "spark.executor.extraJavaOptions=-Ddev.ludovic.netlib.blas.nativeLibPath=/opt/homebrew/opt/openblas/lib/libopenblas.dylib -Ddev.ludovic.netlib.lapack.nativeLibPath=/opt/homebrew/opt/lapack/lib/liblapack.dylib" \

> local_cluster_output.txt 2>&1

spark-submit --conf "spark.driver.extraJavaOptions=-Ddev.ludovic.netlib.blas.nativeLibPath=/opt/homebrew/opt/openblas/lib/libopenblas.dylib -Ddev.ludovic.netlib.lapack.nativeLibPath=/opt/homebrew/opt/lapack/lib/liblapack.dylib" --conf "spark.executor.extraJavaOptions=-Ddev.ludovic.netlib.blas.nativeLibPath=/opt/homebrew/opt/openblas/lib/libopenblas.dylib -Ddev.ludovic.netlib.lapack.nativeLibPath=/opt/homebrew/opt/lapack/lib/liblapack.dylib" --class com.example.LinearRegressionApp target/scala-2.13/SparkLinearRegression-assembly-1.0.jar > local_cluster_output.txt 2>&1