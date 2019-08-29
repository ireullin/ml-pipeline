import classifications.*
import libs.json.Node
import libs.json.toJson
import smile.validation.ConfusionMatrix

import java.io.File


data class Data(val item:String, val src:Source)

fun main(args:Array<String>) {
    val path = "/home/ireullin/data_disk/projects/java_projects/ml-pipeline/test_data"
    val lines = File("$path/10000_items.tsv").readLines()
    val data = lines.map{line->
        val c = line.split("\t")
        val item = c[1]
        val category = c[4]
        val tfidf =  Node.ofMap(c[5]).toMapNotNull{(k,v)-> k to v.toDouble() }
        Data(item, Source(category, tfidf))
    }

    val undropedCategory = data.groupingBy {it.src.label}.eachCount()
            .toList()
            .sortedBy{ it.second }
            .filter { it.second>=100 && it.second<=300 }
            .map{ it.first }
            .toSet()
//            .forEach{ println(it) }
//    return

    val filtered = data.filter{ it.src.label in undropedCategory }
    val src = filtered.map{it.src}
    val pipe = ClassificatorPipeLine(src)
            .addAndTrain("KNN", KNN(20)){ println("knn finished"); true }
            .addAndTrain("LogisticRegression", LogisticRegression()){ println("lr finished"); true }
//            .addAndTrain("NaiveBayes",NaiveBayes()){ println("NaiveBayes finished"); true }
//            .addAndTrain("lda", LDA())
//            .saveReoprt("$path/report.txt")
//            .vote("$path/report2.txt")

    filtered.take(10).forEach {
        val p = pipe.predictWithLabel(it.src.feature)
        println("${it.item} ${it.src.label} ${p.toJson()}"  )
    }

    val validation = filtered.map {
        val p = pipe.predictWithLabel(it.src.feature,1).first().label
        val yBar = pipe.labelToY[p]!!
        val y = pipe.labelToY[it.src.label]!!
        y to yBar
    }

    val y  = validation.map{it.first}.toIntArray()
    val yBar = validation.map{it.second}.toIntArray()
    val cm = ConfusionMatrix(y, yBar)
    File("ConfusionMatrix.txt").writeText(cm.toString())
}