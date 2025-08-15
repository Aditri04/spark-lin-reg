// spark-linear-regression/build.sbt
name := "SparkLinearRegression"
version := "1.0"

// IMPORTANT: Match this Scala version to the one Spark was compiled with.
// Homebrew Spark 4.0.0 uses Scala 2.13.x. Older Spark 3.x might use Scala 2.12.x.
// You can check your Spark's Scala version by running `spark-shell` and looking at the startup logs.
scalaVersion := "2.13.16" // Or "2.12.18" if your Spark is compiled with Scala 2.12

// Spark Dependencies (use 'provided' scope as Spark runtime will provide them)
libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "4.0.0" % "provided", // Match your installed Spark version
  "org.apache.spark" %% "spark-sql" % "4.0.0" % "provided",   // Match your installed Spark version
  "org.apache.spark" %% "spark-mllib" % "3.5.1",

   
)

// Breeze Dependency (use 'compile' scope to bundle it into your fat JAR if needed, or 'provided' if you add it via --packages)
// The `%%` automatically appends the correct Scala binary version (_2.13 or _2.12)
libraryDependencies += "org.scalanlp" %% "breeze" % "2.1.0"

// Set up a main class for `spark-submit`
mainClass := Some("com.example.LinearRegressionApp") // Change com.example if you use a package

// Configuration for creating a single "fat JAR" with all dependencies (except 'provided' ones)
// This is often easier for local spark-submit
assembly / mainClass := Some("com.example.LinearRegressionApp")
assembly / assemblyMergeStrategy := {
  case PathList("META-INF", xs @ _*) => MergeStrategy.discard
  case x => MergeStrategy.first
}
