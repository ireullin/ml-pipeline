package classifications

class LDA(val priori:DoubleArray?=null, val tol:Double=0.0001) :Classificator{
    private lateinit var model:smile.classification.LDA

    override fun train(x:Array<DoubleArray>,y:IntArray):Classificator{
        model = smile.classification.LDA
                .Trainer()
                .setPriori(priori)
                .setTolerance(tol)
                .train(x,y)

        return this
    }

    override fun predict(numPrediction:Int, feature:DoubleArray):Pair<Int,DoubleArray> {
        val probArray = DoubleArray(numPrediction){0.0}
        val prediction = model.predict(feature, probArray)
        return Pair(prediction,probArray)
    }
}