package classifications

interface Classificator{
    fun train(x:Array<DoubleArray>,y:IntArray):Classificator
    fun predict(numPrediction:Int, feature:DoubleArray):Pair<Int,DoubleArray>

}