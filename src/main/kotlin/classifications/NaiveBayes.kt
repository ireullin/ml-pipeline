package classifications

import smile.classification.NaiveBayes

class NaiveBayes(val m:NaiveBayes.Model=NaiveBayes.Model.MULTINOMIAL):Classificator {
    private lateinit var model:smile.classification.NaiveBayes

    override fun train(x:Array<DoubleArray>, y:IntArray):Classificator {
        val k = y.toSet().size
        val p = x[0].size
        model = smile.classification.NaiveBayes.Trainer(m, k, p).train(x,y)
        return this
    }

    override fun predict(numPrediction:Int, feature:DoubleArray):Pair<Int,DoubleArray> {
        val probArray = DoubleArray(numPrediction){0.0}
        val prediction = model.predict(feature, probArray)
        return Pair(prediction,probArray)
    }
}