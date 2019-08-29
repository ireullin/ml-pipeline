package classifications

class KNN(val k:Int=3) :Classificator{
    private lateinit var model:smile.classification.KNN<DoubleArray>

    override fun train(x:Array<DoubleArray>,y:IntArray):Classificator{
        model = smile.classification.KNN.learn(x,y,k)
        return this
    }

    override fun predict(numPrediction:Int, feature:DoubleArray):Pair<Int,DoubleArray> {
        val probArray = DoubleArray(numPrediction){0.0}
        val prediction = model.predict(feature, probArray)
        return Pair(prediction,probArray)
    }
}