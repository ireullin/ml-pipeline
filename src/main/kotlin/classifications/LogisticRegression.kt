package classifications

class LogisticRegression(val lambda: Double = 0.0,val tol: Double = 1E-5,val maxIter: Int = 500):Classificator{
    private lateinit var model:smile.classification.LogisticRegression

    override fun train(x:Array<DoubleArray>, y:IntArray):Classificator {
        model = smile.classification.LogisticRegression
                .Trainer()
                .setMaxNumIteration(maxIter)
                .setRegularizationFactor(lambda)
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