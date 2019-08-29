package classifications

import smile.math.Math.mean
import smile.validation.ConfusionMatrix
import java.io.File
import kotlin.random.Random


data class Source(val label:String, val feature:Map<String,Double>)
data class Prediction(val label:String, val score:Double, val candidates:Map<String,Double>)


class ClassificatorPipeLine {
    data class TrainedClassificator(val name:String, val classificator:Classificator, val yTrainee:IntArray, val xTrainee:Array<DoubleArray>, val yValidation:IntArray, val xValidation:Array<DoubleArray>)

    private lateinit var yTrainee:IntArray
    private lateinit var xTrainee:Array<DoubleArray>

    private lateinit var yValidation:IntArray
    private lateinit var xValidation:Array<DoubleArray>

    private var validationRate:Double = 0.3
    private var randomSeed = 12344
//    private lateinit var random:Random

    private val classificators:MutableList<TrainedClassificator> = mutableListOf()

    var labels:List<String> = listOf()
        private set(value) { field=value }

    var labelToY:Map<String,Int> = mapOf()
        private set(value) { field=value }

    var columns:List<String> = listOf()
        private set(value) { field=value }

    constructor(y:List<Int>, x:List<DoubleArray>, columns:List<String>, validationRate:Double=0.3, randomSeed:Int=12344){
        this.columns = columns
        this.labels = y.map{it.toString()}
        labelToY = labels.mapIndexed{i,v-> v to i}.toMap()
        preprocess(x,y, validationRate, randomSeed)
    }

    constructor(src:List<Source>, validationRate:Double=0.3, randomSeed:Int=12344){
        columns = src.flatMap{it.feature.keys}.toSet().toList()
        labels = src.map{it.label}.toSet().toList()
        labelToY = labels.mapIndexed{i,v-> v to i}.toMap()

        val x = src.map{ oneHotEncoding(it.feature) }
        val y = src.map{ labelToY[it.label]!! }
        preprocess(x, y, validationRate, randomSeed)
    }

    private fun preprocess(x:List<DoubleArray>, y:List<Int>, validationRate:Double, randomSeed:Int){
        // 確定資料筆數正確
        require(y.size==x.size){"x & y are not symmetrical"}

        // 確定label有連續
        val sortedY = y.toSet().toList().sorted()
        require(sortedY.first()==0 && sortedY.last()==(sortedY.size-1) ){"labels of training data are not continue"}

        this.validationRate = validationRate
        this.randomSeed = randomSeed

        val pairs = y.zip(x).shuffled(Random(randomSeed))
        val numValidation = (pairs.size * validationRate).toInt()

        val trainee = pairs.takeLast(numValidation)
        yTrainee = trainee.map{it.first}.toIntArray()
        xTrainee = trainee.map{it.second}.toTypedArray()

        // 確定training data有平均分到各種樣本
        val yTraineeSize = yTrainee.toSet().size
        println("${sortedY}")
        println(" ${yTrainee.toSet().sorted()}")
        require(sortedY.size==yTraineeSize){"training data does not contain whole labels"}

        val validation = pairs.take(numValidation)
        yValidation = validation.map{it.first}.toIntArray()
        xValidation = validation.map{it.second}.toTypedArray()
    }

    private fun oneHotEncoding(d:Map<String,Double>):DoubleArray{
        return columns.map{ d[it]?:0.0 }.toDoubleArray()
    }

    fun addAndTrain(name:String, classificator:Classificator, doesFilter:(TrainedClassificator)->Boolean ):ClassificatorPipeLine {
        classificator.train(xTrainee,yTrainee)
        val trained = TrainedClassificator(name, classificator, yTrainee, xTrainee, yValidation, xValidation)
        if(doesFilter(trained)) {
            classificators.add(trained)
        }
        return this
    }

    fun addAndTrain(name:String, classificator:Classificator):ClassificatorPipeLine {
        return addAndTrain(name, classificator){ true }
    }

    fun saveReoprt(file:String):ClassificatorPipeLine{
        val w = File(file).bufferedWriter()
        classificators.forEach{trained->
            val predictions = xValidation.map{trained.classificator.predict(labels.size, it)}
            val yBar = predictions.map{ it.first }.toIntArray()
            w.write(trained.name+"\n")

            if(labels.size==2) {
                val acc = smile.validation.Accuracy().measure(yValidation, yBar)
                val f1 = smile.validation.FMeasure().measure(yValidation, yBar)
                val precision = smile.validation.Precision().measure(yValidation, yBar)
                val recall = smile.validation.Recall().measure(yValidation, yBar)
                w.write("Accuracy: $acc\n")
                w.write("F1 score: $f1\n")
                w.write("Precision: $precision\n")
                w.write("Recall: $recall\n")
            }

            val cm = ConfusionMatrix(yValidation, yBar)
            w.write(cm.toString())
            w.newLine()
            w.newLine()
        }
        w.close()
        return this
    }


    fun predictWithLabel(feature:DoubleArray, topN:Int=3):List<Prediction>{
        data class Buff(val label:String, val candidates:MutableMap<String,Double>)

        val predictions = labels.map{ Buff(it, mutableMapOf()) }
        classificators.forEach{model->
            val (pre, prob) = model.classificator.predict(labels.size, feature)
            predictions.forEachIndexed{i,p->
                p.candidates.put(model.name, prob[i])
            }
        }

        return predictions.map{
            val mean = mean(it.candidates.values.toDoubleArray())
            Prediction(it.label, mean, it.candidates)
        }.sortedByDescending{it.score}.take(topN)
    }

    fun predictWithLabel(feature:Map<String,Double>, topN:Int=3):List<Prediction>{
        val f = oneHotEncoding(feature)
        return predictWithLabel(f, topN)
    }

    fun _vote(feature:DoubleArray){
        val candidate = classificators.map {model->
            val (pre, prob) = model.classificator.predict(labels.size, feature)
            prob
        }
    }


    fun vote(file:String):ClassificatorPipeLine {
//        val newLabels = classificators.map{it.name}

        val newX = xValidation + xTrainee
        val newY = (yValidation + yValidation).toList()

        val expandedProb = newX.map { feature->
            classificators.flatMap {model->
                val (pre, prob) = model.classificator.predict(labels.size, feature)
                prob.toList()
            }.toDoubleArray()
        }


        ClassificatorPipeLine(newY, expandedProb, listOf(), validationRate, randomSeed+5)
                .addAndTrain("voteLr", LogisticRegression(maxIter = 10000))
                .saveReoprt(file)

        return this
    }
}