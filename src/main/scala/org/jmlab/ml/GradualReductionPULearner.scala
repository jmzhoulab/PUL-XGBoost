package org.jmlab.ml

import java.io.{File, PrintWriter}

import ml.dmlc.xgboost4j.LabeledPoint
import ml.dmlc.xgboost4j.scala.{Booster, DMatrix, XGBoost}
import org.apache.log4j.{LogManager, Logger}
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.SparkSession

/**
  * Created by jmzhou on 2018/9/18.
  */
class GradualReductionPULearner {

  val log: Logger = LogManager.getLogger(getClass)
  val prebiThreshold = 0.3f
  var spark: SparkSession = _
  val iterationNum = 20

  def loadData(): DMatrix ={
    val valid = MLUtils.loadLibSVMFile(spark.sparkContext, "data/data.libsvm")
      .map(point => {
        LabeledPoint(point.label.toFloat,
          point.features.toSparse.indices,
          point.features.toSparse.values.map(_.toFloat)
        )
      }).collect().toIterator
    new DMatrix(valid)
  }

  def weight(labeledPoints: Array[LabeledPoint]): (Booster, Array[LabeledPoint]) ={
    val posPoint = labeledPoints.filter(p => p.label == 1.0)
    val init = zeroStep(labeledPoints)

    var relNegPoint = init._1
    var negPoint = init._2
    var preNegPoint = negPoint
    var classifier: Booster = null
    var iterNum = 1

    val validDMat = loadData()

    var relNegNum = 0
    var stopFlag = false

    while (negPoint.length <= preNegPoint.length && posPoint.length < relNegPoint.length && !stopFlag){
      iterNum += 1
      println("iterNum: " + iterNum)
      val dmat = new DMatrix((posPoint++relNegPoint).toIterator)
      val posNum = posPoint.length
      val negNum = relNegPoint.length
      classifier = XGBoost.train(dmat, getParamMap(posNum, negNum), iterationNum)
//      evaluate(spark, classifier, validDMat)
      val predict = classifier.predict(new DMatrix(relNegPoint.toIterator)).flatten
        .map(p => if(p > prebiThreshold) 1.0f else 0.0f)
      preNegPoint = negPoint
      negPoint = relNegPoint.zip(predict).filter{case(p, l) => l == 0.0f}.map(_._1)
      relNegPoint = (relNegPoint ++ negPoint).distinct
      println("posNum: " + posNum)
      if (relNegNum != relNegPoint.length)
        relNegNum = relNegPoint.length
      else if (iterNum >= 2)
        stopFlag = true
      println("relNegPoint: " + relNegNum)
    }
    (classifier, posPoint++relNegPoint)
  }

  def zeroStep(labeledPoints: Array[LabeledPoint]): (Array[LabeledPoint], Array[LabeledPoint]) = {
    val posNum = labeledPoints.count(p => p.label == 1.0)
    val negNum = labeledPoints.count(p => p.label == 0.0)
    val unLabelPoint = labeledPoints.filter(p => p.label == 0.0)
    val dmat = new DMatrix(labeledPoints.toIterator)
    val classifier = XGBoost.train(dmat, getParamMap(posNum, negNum), iterationNum)
    val validDMat = loadData()
//    evaluate(spark, classifier, validDMat)
    val predict = classifier.predict(new DMatrix(unLabelPoint.toIterator))
      .flatten.map(p => if(p > prebiThreshold) 1.0f else 0.0f)
    val negPoint = unLabelPoint.zip(predict).filter{case(p, l) => l == 0.0f}.map(_._1)
    val relNegPoint = negPoint
    (relNegPoint, negPoint)
  }

  def getParamMap(posNum: Int, negNum: Int): Map[String, Any] = {
    List("eta" -> 0.1f,
      "scale_pos_weight" -> negNum/posNum.toDouble,
      "max_depth" -> 5,
      "silent" -> 0,
      "objective" -> "binary:logistic",
      "lambda" -> 2.5,
      "rate_drop" -> 0.5,
      "alpha" -> 1
    ).toMap
  }

  def evaluate(spark: SparkSession, model: Booster, test_dmat: DMatrix): Unit ={
    val labels = test_dmat.getLabel.map(_.toDouble)
    val predict_xgb = model.predict(test_dmat).flatten

    val scoreAndLabels = spark.sparkContext.makeRDD(predict_xgb.map(_.toDouble) zip labels)

    val xgbMetrics = new BinaryClassificationMetrics(scoreAndLabels)
    val auROC = xgbMetrics.areaUnderROC()

    println("xgboost: Area under ROC = " + auROC)

    val predicts = predict_xgb.map(p => if(p >= prebiThreshold) 1.0 else 0.0)
    val predictsAndLabels = spark.sparkContext.makeRDD(predicts zip labels)

    val roc = xgbMetrics.roc().map{case(fpr, recall) => s"$fpr,$recall"}.collect()

    val metrics = new MulticlassMetrics(predictsAndLabels)
    val confusionMatrix = metrics.confusionMatrix
    println("confusionMatrix: ")
    println(confusionMatrix)

    val TP = confusionMatrix.apply(1, 1)
    val FP = confusionMatrix.apply(0, 1)
    val FN = confusionMatrix.apply(1, 0)
    val P = TP/(TP+FP)
    val R = TP/(TP+FN)

    println("P: " + P)
    println("R: " + R)

    val f1 = 2*P*R/(P+R)

    println("accuracy: " + metrics.accuracy)
    println("f1 score: " + f1)
    println("class 1 recall: " + metrics.recall(1.0))
    println("class 0 recall: " + metrics.recall(0.0))

  }

}

object GradualReductionPULearner{

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder
      .master("local")
      .appName("GradualReductionPULearner")
      .getOrCreate()

    val pULearner = new GradualReductionPULearner
    pULearner.spark = spark


    val data = MLUtils.loadLibSVMFile(spark.sparkContext, "data/data.libsvm")

    val train = data.map(point => {
      LabeledPoint(point.label.toFloat,
        point.features.toSparse.indices,
        point.features.toSparse.values.map(_.toFloat)
      )
    }).collect()

    val result = pULearner.weight(train)

    val model = result._1

    model.saveModel("model/model.bin")

    val sample = result._2

    println("pu_sample_count: " + sample.length)

    val libsvm = sample.map(p => s"${p.label.toInt} " + (p.indices zip p.values).map { case (i, v) => "%d:%.2f".format(i+1, v) }.mkString(" "))

    val writer = new PrintWriter(new File("data/data_with_rn.libsvm"))

    libsvm.foreach(line => writer.write(line + "\n"))

    writer.close()

    println(sample.length)

    println(sample.count(p => p.label == 1.0))
    println(sample.count(p => p.label == 0.0))


  }

}


