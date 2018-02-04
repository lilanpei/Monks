using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using AA1_MLP.CustomExtensionMethods;
using MathNet.Numerics.LinearAlgebra;
using AA1_MLP.Enums;
using AA1_MLP.Entities.TrainersParams;

namespace AA1_MLP.Entities.Trainers
{
    public class Adam : IOptimizer
    {
        private static readonly object thisLock = new object();

        public override List<double[]> Train(TrainerParams trainParams)
        {
            AdamParams passedParams = (AdamParams)trainParams;
            //int valSplitSize = 0;
            List<double[]> learningCurve = new List<double[]>();
            List<int> trainingSetIndices = Enumerable.Range(0, passedParams.trainingSet.Labels.RowCount).ToList();
            List<int> testSetIndices = null;
            DataSet test = new DataSet(null, null);
            if (passedParams.validationSet != null)
            {
                testSetIndices = Enumerable.Range(0, passedParams.validationSet.Labels.RowCount).ToList();
                /*  if (shuffle)
                  {
                      testSetIndices.Shuffle();
                  }
                  */
                test.Inputs = CreateMatrix.Dense(testSetIndices.Count, passedParams.validationSet.Inputs.ColumnCount, 0.0);
                test.Labels = CreateMatrix.Dense(testSetIndices.Count, passedParams.validationSet.Labels.ColumnCount, 0.0);
                for (int i = 0; i < testSetIndices.Count; i++)
                {
                    test.Inputs.SetRow(i, passedParams.validationSet.Inputs.Row(testSetIndices[i]));//, 1, 0, Dataset.Inputs.ColumnCount));
                    test.Labels.SetRow(i, passedParams.validationSet.Labels.Row(testSetIndices[i]));//.SubMatrix(trainingSetIndices[batchIndex], 1, 0, Dataset.Labels.ColumnCount));

                }
            }
            if (passedParams.shuffle)
            {
                trainingSetIndices.Shuffle();
            }



            Matrix<double> batchesIndices = null;//a 2d matrix of shape(nmberOfBatches,2), rows are batches, row[0] =barchstart, row[1] = batchEnd 
            Dictionary<int, Matrix<double>> previousWeightsUpdate = null;//for the momentum updates

            int adamUpdateStep = 1;
            for (int epoch = 1; epoch <= passedParams.numberOfEpochs; epoch++)
            {
                if (passedParams.batchSize != null)//will build a matrix "batchesIndices" describing the batches that in each row, contains the start and the end of a batch
                {
                    var numberOfBatches = (int)Math.Ceiling(((passedParams.trainingSet.Labels.RowCount / (double)(passedParams.batchSize))));
                    batchesIndices = CreateMatrix.Dense(numberOfBatches, 2, 0.0);
                    for (int j = 0; j < numberOfBatches; j++)
                    {
                        batchesIndices.SetRow(j, new double[] { j * (double)passedParams.batchSize, Math.Min(passedParams.trainingSet.Inputs.RowCount - 1, (j + 1) * (double)passedParams.batchSize - 1) });
                    }
                }
                else//put all of the dataset in one batch
                {
                    batchesIndices = CreateMatrix.Dense(1, 2, 0.0);
                    batchesIndices.SetRow(0, new double[] { 0, passedParams.trainingSet.Inputs.RowCount - 1 });
                }

                double iterationLoss = 0;//will hold the average of the batches average losses, each batch contributes to this with its loss average =  batchloss/batchsize
                Dictionary<int, Matrix<double>> firstMoment = new Dictionary<int, Matrix<double>>();
                Dictionary<int, Matrix<double>> secondMoment = new Dictionary<int, Matrix<double>>();
                Dictionary<int, Matrix<double>> mhat = new Dictionary<int, Matrix<double>>();
                Dictionary<int, Matrix<double>> vhat = new Dictionary<int, Matrix<double>>();

                Dictionary<int, Matrix<double>> prevFirstMoment = new Dictionary<int, Matrix<double>>();
                Dictionary<int, Matrix<double>> prevSecondMoment = new Dictionary<int, Matrix<double>>();
                for (int batchIndex = 0; batchIndex < batchesIndices.RowCount; batchIndex++)//for each batch
                    previousWeightsUpdate = passedParams.parallelize ? Parallel_PerformBatchComputations(passedParams, batchesIndices, adamUpdateStep, ref iterationLoss, firstMoment, secondMoment, mhat, vhat, prevFirstMoment, prevSecondMoment, batchIndex) : PerformBatchComputations(passedParams, batchesIndices, adamUpdateStep, ref iterationLoss, firstMoment, secondMoment, mhat, vhat, prevFirstMoment, prevSecondMoment, batchIndex);//for each batch

                adamUpdateStep++;


                iterationLoss /= batchesIndices.RowCount;



                // computing the test loss:
                double validationError = 0;
                validationError = passedParams.parallelize ? Parallel_ComputeValidationLoss(passedParams, testSetIndices, test) : ComputeValidationLoss(passedParams, testSetIndices, test);
                double trainingAccuracy = 0, validationSetAccuracy = 0;

                if (passedParams.trueThreshold != null)
                {
                    trainingAccuracy = Utilities.Tools.ComputeAccuracy(passedParams.network, passedParams.trainingSet, passedParams.trueThreshold);
                    validationSetAccuracy = Utilities.Tools.ComputeAccuracy(passedParams.network, passedParams.validationSet, passedParams.trueThreshold);
                }


                learningCurve.Add(new double[] { iterationLoss, passedParams.validationSet != null ? validationError : 0, passedParams.trueThreshold != null ? trainingAccuracy : 0, passedParams.trueThreshold != null ? validationSetAccuracy : 0 });
                if (passedParams.PrintLoss)
                {
                    Console.ForegroundColor = ConsoleColor.Green;
                    Console.WriteLine("Epoch:{0} train loss:{1} - validation loss:{2}", epoch, iterationLoss, validationError);

                }


                Console.ResetColor();


            }
            return learningCurve;
        }

        private static double ComputeValidationLoss(AdamParams passedParams, List<int> testSetIndices, DataSet test)
        {
            double validationError = 0;
            if (passedParams.validationSet != null)
            {
                for (int i = 0; i < testSetIndices.Count; i++)
                {
                    var nwOutput = passedParams.network.Predict(test.Inputs.Row(i));
                    var loss = ((test.Labels.Row(i) - nwOutput).PointwiseMultiply(test.Labels.Row(i) - nwOutput)).Sum();
                    validationError += passedParams.MEE ? Math.Sqrt(loss) : loss;

                }
                validationError /= testSetIndices.Count;


            }

            return validationError;
        }
        private static double Parallel_ComputeValidationLoss(AdamParams passedParams, List<int> testSetIndices, DataSet test)
        {
            // computing the test loss:
            double validationError = 0;
            double[] validationErrors = new double[testSetIndices.Count];
            if (passedParams.validationSet != null)
            {
                Parallel.For(0, testSetIndices.Count, threadidx =>
                {
                    int i = testSetIndices[threadidx];
                    var nwOutput = passedParams.network.Predict(test.Inputs.Row(i));
                    var loss = ((test.Labels.Row(i) - nwOutput).PointwiseMultiply(test.Labels.Row(i) - nwOutput)).Sum();
                    validationErrors[threadidx] = passedParams.MEE ? Math.Sqrt(loss) : loss;

                });
                validationError = validationErrors.Sum() / testSetIndices.Count;


            }

            return validationError;
        }

        private static Dictionary<int, Matrix<double>> PerformBatchComputations(AdamParams passedParams, Matrix<double> batchesIndices, int adamUpdateStep, ref double iterationLoss, Dictionary<int, Matrix<double>> firstMoment, Dictionary<int, Matrix<double>> secondMoment, Dictionary<int, Matrix<double>> mhat, Dictionary<int, Matrix<double>> vhat, Dictionary<int, Matrix<double>> prevFirstMoment, Dictionary<int, Matrix<double>> prevSecondMoment, int batchIndex)
        {
            Dictionary<int, Matrix<double>> previousWeightsUpdate;
            {


                double batchLoss = 0;
                Dictionary<int, Matrix<double>> weightsUpdates = new Dictionary<int, Matrix<double>>();


                int numberOfBatchExamples = (((int)batchesIndices.Row(batchIndex).At(1) - (int)batchesIndices.Row(batchIndex).At(0)) + 1);//not all batches have batchSize, unfortunately, the last one could be smaller
                var batchIndices = Enumerable.Range((int)batchesIndices.Row(batchIndex).At(0), (int)batchesIndices.Row(batchIndex).At(1) - (int)batchesIndices.Row(batchIndex).At(0) + 1).ToList();
                if (passedParams.shuffle)
                {
                    batchIndices.Shuffle();
                }

                foreach (int k in batchIndices)//for each elemnt in th batch
                    batchLoss = PerformExampleComputations(passedParams, firstMoment, secondMoment, mhat, vhat, prevFirstMoment, prevSecondMoment, batchIndex, batchLoss, weightsUpdates, k);//per example in the batch




                if (passedParams.debug)
                    Console.WriteLine("batch end");



                for (int y = 0; y < weightsUpdates.Keys.Count; y++)
                {





                    /*   passedParams.learningRate = passedParams.learningRate * Math.Sqrt(1 - Math.Pow(passedParams.beta2, adamUpdateStep)) / (1 - Math.Pow(passedParams.beta1, adamUpdateStep));

                       firstMoment[y] = passedParams.beta1 * prevFirstMoment[y] + (1 - passedParams.beta1) * (-1 * weightsUpdates[y]);
                       secondMoment[y] = passedParams.beta2 * prevSecondMoment[y] + (1 - passedParams.beta2) * weightsUpdates[y].PointwisePower(2);

                       passedParams.network.Weights[y] -= (passedParams.learningRate * firstMoment[y]).PointwiseDivide((secondMoment[y].PointwiseSqrt() + passedParams.epsilon));
                       */

                    //The simple implementation of adam at the beginning of the paper
                    weightsUpdates[y] = weightsUpdates[y] / numberOfBatchExamples;
                    firstMoment[y] = passedParams.beta1 * prevFirstMoment[y] + (1 - passedParams.beta1) * (-1 * weightsUpdates[y]);
                    secondMoment[y] = passedParams.beta2 * prevSecondMoment[y] + (1 - passedParams.beta2) * weightsUpdates[y].PointwisePower(2);
                    mhat[y] = firstMoment[y] / (1 - Math.Pow(passedParams.beta1, adamUpdateStep));
                    vhat[y] = secondMoment[y] / (1 - Math.Pow(passedParams.beta2, adamUpdateStep));
                    var finalUpdates = (passedParams.learningRate * mhat[y]).PointwiseDivide((vhat[y].PointwiseSqrt() + passedParams.epsilon));
                    passedParams.network.Weights[y] -= finalUpdates;

                    /*  for check
                    alpha_t = learningRate * np.sqrt(1 - beta2**t) / (1 - beta1**t)

                    m_t = beta1 * m + (1 - beta1) * g_t
                    v_t = beta2 * v + (1 - beta2) * g_t * g_t

                    param_t = param - alpha_t * m_t / (np.sqrt(v_t) + epsilon)

                   */

                    prevFirstMoment[y] = firstMoment[y].Clone();
                    prevSecondMoment[y] = secondMoment[y].Clone();
                    weightsUpdates[y] = finalUpdates.Clone();




                }
                previousWeightsUpdate = weightsUpdates;

                iterationLoss += batchLoss / ((int)batchesIndices.Row(batchIndex).At(1) - (int)batchesIndices.Row(batchIndex).At(0) + 1);
                if (passedParams.network.Debug)
                    Console.WriteLine("Batch: {0} Error: {1}", batchIndex, batchLoss);
            }

            return previousWeightsUpdate;
        }

        private static Dictionary<int, Matrix<double>> Parallel_PerformBatchComputations(AdamParams passedParams, Matrix<double> batchesIndices, int adamUpdateStep, ref double iterationLoss, Dictionary<int, Matrix<double>> firstMoment, Dictionary<int, Matrix<double>> secondMoment, Dictionary<int, Matrix<double>> mhat, Dictionary<int, Matrix<double>> vhat, Dictionary<int, Matrix<double>> prevFirstMoment, Dictionary<int, Matrix<double>> prevSecondMoment, int batchIndex)
        {
            Dictionary<int, Matrix<double>> previousWeightsUpdate;
            {


                double batchLoss = 0;
                Dictionary<int, Matrix<double>> weightsUpdates = new Dictionary<int, Matrix<double>>();


                int numberOfBatchExamples = (((int)batchesIndices.Row(batchIndex).At(1) - (int)batchesIndices.Row(batchIndex).At(0)) + 1);//not all batches have batchSize, unfortunately, the last one could be smaller
                var batchIndices = Enumerable.Range((int)batchesIndices.Row(batchIndex).At(0), (int)batchesIndices.Row(batchIndex).At(1) - (int)batchesIndices.Row(batchIndex).At(0) + 1).ToList();
                if (passedParams.shuffle)
                {
                    batchIndices.Shuffle();
                }

                for (int layerIndex = passedParams.network.Layers.Count - 1; layerIndex >= 1; layerIndex--)
                {

                    if (!weightsUpdates.ContainsKey(layerIndex - 1))
                    {
                        weightsUpdates.Add(layerIndex - 1, CreateMatrix.Dense(passedParams.network.Weights[layerIndex - 1].RowCount, passedParams.network.Weights[layerIndex - 1].ColumnCount, 0.0));

                    }


                    if (!firstMoment.ContainsKey(layerIndex - 1))
                    {
                        firstMoment.Add(layerIndex - 1, CreateMatrix.Dense(passedParams.network.Weights[layerIndex - 1].RowCount, passedParams.network.Weights[layerIndex - 1].ColumnCount, 0.0));
                        secondMoment.Add(layerIndex - 1, CreateMatrix.Dense(passedParams.network.Weights[layerIndex - 1].RowCount, passedParams.network.Weights[layerIndex - 1].ColumnCount, 0.0));
                        mhat.Add(layerIndex - 1, CreateMatrix.Dense(passedParams.network.Weights[layerIndex - 1].RowCount, passedParams.network.Weights[layerIndex - 1].ColumnCount, 0.0));
                        vhat.Add(layerIndex - 1, CreateMatrix.Dense(passedParams.network.Weights[layerIndex - 1].RowCount, passedParams.network.Weights[layerIndex - 1].ColumnCount, 0.0));

                        prevFirstMoment.Add(layerIndex - 1, CreateMatrix.Dense(passedParams.network.Weights[layerIndex - 1].RowCount, passedParams.network.Weights[layerIndex - 1].ColumnCount, 0.0));
                        prevSecondMoment.Add(layerIndex - 1, CreateMatrix.Dense(passedParams.network.Weights[layerIndex - 1].RowCount, passedParams.network.Weights[layerIndex - 1].ColumnCount, 0.0));


                    }
                }

                double[] examplesLosses = new double[batchIndices.Count];

                var parops = new ParallelOptions();
               // parops.MaxDegreeOfParallelism = 2;
                Parallel.For(0, batchIndices.Count, parops,
                        batchElementThreadIndx =>
                        {
                            Parallel_PerformExampleComputations(passedParams, batchIndex, weightsUpdates, batchIndices[ batchElementThreadIndx], examplesLosses, batchElementThreadIndx);

                        });//per example in the batch



                batchLoss = examplesLosses.Sum();

                if (passedParams.debug)
                    Console.WriteLine("batch end");



                for (int y = 0; y < weightsUpdates.Keys.Count; y++)
                {





                    /*   passedParams.learningRate = passedParams.learningRate * Math.Sqrt(1 - Math.Pow(passedParams.beta2, adamUpdateStep)) / (1 - Math.Pow(passedParams.beta1, adamUpdateStep));

                       firstMoment[y] = passedParams.beta1 * prevFirstMoment[y] + (1 - passedParams.beta1) * (-1 * weightsUpdates[y]);
                       secondMoment[y] = passedParams.beta2 * prevSecondMoment[y] + (1 - passedParams.beta2) * weightsUpdates[y].PointwisePower(2);

                       passedParams.network.Weights[y] -= (passedParams.learningRate * firstMoment[y]).PointwiseDivide((secondMoment[y].PointwiseSqrt() + passedParams.epsilon));
                       */

                    //The simple implementation of adam at the beginning of the paper
                    weightsUpdates[y] = weightsUpdates[y] / numberOfBatchExamples;
                    firstMoment[y] = passedParams.beta1 * prevFirstMoment[y] + (1 - passedParams.beta1) * (-1 * weightsUpdates[y]);
                    secondMoment[y] = passedParams.beta2 * prevSecondMoment[y] + (1 - passedParams.beta2) * weightsUpdates[y].PointwisePower(2);
                    mhat[y] = firstMoment[y] / (1 - Math.Pow(passedParams.beta1, adamUpdateStep));
                    vhat[y] = secondMoment[y] / (1 - Math.Pow(passedParams.beta2, adamUpdateStep));
                    var finalUpdates = (passedParams.learningRate * mhat[y]).PointwiseDivide((vhat[y].PointwiseSqrt() + passedParams.epsilon));
                    passedParams.network.Weights[y] -= finalUpdates;

                    /*  for check
                    alpha_t = learningRate * np.sqrt(1 - beta2**t) / (1 - beta1**t)

                    m_t = beta1 * m + (1 - beta1) * g_t
                    v_t = beta2 * v + (1 - beta2) * g_t * g_t

                    param_t = param - alpha_t * m_t / (np.sqrt(v_t) + epsilon)

                   */

                    prevFirstMoment[y] = firstMoment[y].Clone();
                    prevSecondMoment[y] = secondMoment[y].Clone();
                    weightsUpdates[y] = finalUpdates.Clone();




                }
                previousWeightsUpdate = weightsUpdates;

                iterationLoss += batchLoss / ((int)batchesIndices.Row(batchIndex).At(1) - (int)batchesIndices.Row(batchIndex).At(0) + 1);
                if (passedParams.network.Debug)
                    Console.WriteLine("Batch: {0} Error: {1}", batchIndex, batchLoss);
            }

            return previousWeightsUpdate;
        }

        private static double PerformExampleComputations(AdamParams passedParams, Dictionary<int, Matrix<double>> firstMoment, Dictionary<int, Matrix<double>> secondMoment, Dictionary<int, Matrix<double>> mhat, Dictionary<int, Matrix<double>> vhat, Dictionary<int, Matrix<double>> prevFirstMoment, Dictionary<int, Matrix<double>> prevSecondMoment, int batchIndex, double batchLoss, Dictionary<int, Matrix<double>> weightsUpdates, int k)
        {
            {
                var nwOutput = passedParams.network.Predict(passedParams.trainingSet.Inputs.Row(k));
                var label = passedParams.trainingSet.Labels.Row(k);


                //TODO: get the loss computation out as a parameter to the function, so that the user can specify it freely
                var loss = ((label - nwOutput).PointwiseMultiply(label - nwOutput)).Sum();
                batchLoss += passedParams.MEE ? Math.Sqrt(loss) : loss;

                var residual = label - nwOutput;

                if (passedParams.debug)
                {
                    Console.WriteLine("Target:{0}", label);
                    Console.WriteLine("Calculated:{0}", nwOutput);
                    Console.WriteLine("Target-calculated (residual):{0}", residual);
                }

                residual = residual.Map(r => double.IsNaN(r) ? 0 : r);

                for (int layerIndex = passedParams.network.Layers.Count - 1; layerIndex >= 1; layerIndex--)
                {
                    if (passedParams.debug)
                        Console.WriteLine("##### enting backpropagation layer index: {0} ######", layerIndex);
                    Vector<double> derivative = passedParams.network.Layers[layerIndex].Activation.CalculateDerivative(passedParams.network.Layers[layerIndex].LayerActivationsSumInputs);

                    if (passedParams.debug)
                    {
                        Console.WriteLine("output sum(the sum inputted to the activation(LayerActivationsSumInputs)): {0}", passedParams.network.Layers[layerIndex].LayerActivationsSumInputs);
                        Console.WriteLine("derivative: {0}", derivative);
                        Console.WriteLine("output sum margin of error(residual): {0}", residual);
                        Console.WriteLine("Delta output sum of Layer(residual*derivative): {0}", layerIndex);

                    }
                    if (layerIndex == passedParams.network.Layers.Count - 1)
                    {
                        passedParams.network.Layers[layerIndex].Delta = residual.PointwiseMultiply(derivative);

                    }
                    else
                    {
                        Matrix<double> wei = passedParams.network.Weights[layerIndex];
                        if (wei.RowCount > derivative.Count)//there is a bias
                        {
                            wei = wei.SubMatrix(0, wei.RowCount - 1, 0, wei.ColumnCount);
                        }

                        passedParams.network.Layers[layerIndex].Delta = (wei * passedParams.network.Layers[layerIndex + 1].Delta).PointwiseMultiply(derivative);
                    }

                    if (!weightsUpdates.ContainsKey(layerIndex - 1))
                    {
                        weightsUpdates.Add(layerIndex - 1, CreateMatrix.Dense(passedParams.network.Weights[layerIndex - 1].RowCount, passedParams.network.Weights[layerIndex - 1].ColumnCount, 0.0));

                    }


                    if (!firstMoment.ContainsKey(layerIndex - 1))
                    {
                        firstMoment.Add(layerIndex - 1, CreateMatrix.Dense(passedParams.network.Weights[layerIndex - 1].RowCount, passedParams.network.Weights[layerIndex - 1].ColumnCount, 0.0));
                        secondMoment.Add(layerIndex - 1, CreateMatrix.Dense(passedParams.network.Weights[layerIndex - 1].RowCount, passedParams.network.Weights[layerIndex - 1].ColumnCount, 0.0));
                        mhat.Add(layerIndex - 1, CreateMatrix.Dense(passedParams.network.Weights[layerIndex - 1].RowCount, passedParams.network.Weights[layerIndex - 1].ColumnCount, 0.0));
                        vhat.Add(layerIndex - 1, CreateMatrix.Dense(passedParams.network.Weights[layerIndex - 1].RowCount, passedParams.network.Weights[layerIndex - 1].ColumnCount, 0.0));

                        prevFirstMoment.Add(layerIndex - 1, CreateMatrix.Dense(passedParams.network.Weights[layerIndex - 1].RowCount, passedParams.network.Weights[layerIndex - 1].ColumnCount, 0.0));
                        prevSecondMoment.Add(layerIndex - 1, CreateMatrix.Dense(passedParams.network.Weights[layerIndex - 1].RowCount, passedParams.network.Weights[layerIndex - 1].ColumnCount, 0.0));


                    }



                    if (passedParams.network.Debug)
                    {
                        Console.ForegroundColor = ConsoleColor.Red;
                        Console.WriteLine("Weights layer:{0} {1}", layerIndex - 1, passedParams.network.Weights[layerIndex - 1]);
                        Console.ResetColor();
                    }
                    Matrix<double> weightsUpdate = null;
                    var acti = passedParams.network.Layers[layerIndex - 1].LayerActivations;
                    if (passedParams.network.Layers[layerIndex - 1].Bias && acti.Count - passedParams.network.Layers[layerIndex - 1].NumberOfNeurons < 1)//if the user asked a bias should be added, we need to add a dummy neuron of activation =1 as a bias at the end of the layer's activations
                    {
                        var l = acti.ToList();
                        l.Add(1);//adding the bias

                        acti = CreateVector.Dense(l.ToArray());
                    }
                    weightsUpdate = acti.OuterProduct(passedParams.network.Layers[layerIndex].Delta);



                    if (passedParams.regularization == Regularizations.L2)
                    {

                        weightsUpdates[layerIndex - 1] = weightsUpdates[layerIndex - 1].Add((weightsUpdate - 2 * passedParams.regularizationRate * passedParams.network.Weights[layerIndex - 1]));

                    }
                    else
                    { weightsUpdates[layerIndex - 1] = weightsUpdates[layerIndex - 1].Add(weightsUpdate); }


                    if (passedParams.debug)
                    {
                        Console.WriteLine("weights updates of weightsMatrix(learning rate* outerproduct(Layer{1} delta,layer{2} output from activations) ): {0} ", layerIndex - 1, layerIndex, layerIndex - 1);
                        Console.WriteLine("learning rate:{0}", passedParams.learningRate);
                        Console.WriteLine("Layer:{0} delta: {1}", layerIndex, passedParams.network.Layers[layerIndex].Delta);
                        Console.WriteLine("layer{0} output from activations:{1}", layerIndex - 1, passedParams.network.Layers[layerIndex - 1].LayerActivations);
                        Console.WriteLine(weightsUpdate);
                        Console.WriteLine("----------- Gradientdescent LayerIndex{0} ------------", layerIndex);

                    }

                }//back propagating per layer

                if (passedParams.debug)
                    Console.WriteLine("-------- Batch:{0} element:{1} end-----", batchIndex, k);
            }

            return batchLoss;
        }

        private static void Parallel_PerformExampleComputations(AdamParams passedParams, int batchIndex, Dictionary<int, Matrix<double>> weightsUpdates, int elementIdxInBatch, double[] lossesarray, int batchElementThreadIndx)
        {

            var nwOutput = passedParams.network.Predict(passedParams.trainingSet.Inputs.Row(elementIdxInBatch));
            var label = passedParams.trainingSet.Labels.Row(elementIdxInBatch);


            //TODO: get the loss computation out as a parameter to the function, so that the user can specify it freely
            var loss = ((label - nwOutput).PointwiseMultiply(label - nwOutput)).Sum();

            lossesarray[batchElementThreadIndx] = passedParams.MEE ? Math.Sqrt(loss) : loss;

            var residual = label - nwOutput;

            if (passedParams.debug)
            {
                Console.WriteLine("Target:{0}", label);
                Console.WriteLine("Calculated:{0}", nwOutput);
                Console.WriteLine("Target-calculated (residual):{0}", residual);
            }

            residual = residual.Map(r => double.IsNaN(r) ? 0 : r);

            var layers = new Layer[passedParams.network.Layers.Count];
            for (int lyrIdx = 0; lyrIdx < passedParams.network.Layers.Count; lyrIdx++)
            {
                layers[lyrIdx] = passedParams.network.Layers[lyrIdx].GetDeepClone();
            }


            for (int layerIndex = layers.Length - 1; layerIndex >= 1; layerIndex--)
            {
                if (passedParams.debug)
                    Console.WriteLine("##### enting backpropagation layer index: {0} ######", layerIndex);
                Vector<double> derivative = layers[layerIndex].Activation.CalculateDerivative(layers[layerIndex].LayerActivationsSumInputs);

                if (passedParams.debug)
                {
                    Console.WriteLine("output sum(the sum inputted to the activation(LayerActivationsSumInputs)): {0}", layers[layerIndex].LayerActivationsSumInputs);
                    Console.WriteLine("derivative: {0}", derivative);
                    Console.WriteLine("output sum margin of error(residual): {0}", residual);
                    Console.WriteLine("Delta output sum of Layer(residual*derivative): {0}", layerIndex);

                }
                if (layerIndex == layers.Length - 1)
                {
                    layers[layerIndex].Delta = residual.PointwiseMultiply(derivative);

                }
                else
                {
                    Matrix<double> wei = passedParams.network.Weights[layerIndex];
                    if (wei.RowCount > derivative.Count)//there is a bias
                    {
                        wei = wei.SubMatrix(0, wei.RowCount - 1, 0, wei.ColumnCount);
                    }

                    layers[layerIndex].Delta = (wei * layers[layerIndex + 1].Delta).PointwiseMultiply(derivative);
                }





                if (passedParams.network.Debug)
                {
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.WriteLine("Weights layer:{0} {1}", layerIndex - 1, passedParams.network.Weights[layerIndex - 1]);
                    Console.ResetColor();
                }
                Matrix<double> weightsUpdate = null;
                var acti = layers[layerIndex - 1].LayerActivations;
                if (layers[layerIndex - 1].Bias && acti.Count - layers[layerIndex - 1].NumberOfNeurons < 1)//if the user asked a bias should be added, we need to add a dummy neuron of activation =1 as a bias at the end of the layer's activations
                {
                    var l = acti.ToList();
                    l.Add(1);//adding the bias

                    acti = CreateVector.Dense(l.ToArray());
                }
                weightsUpdate = acti.OuterProduct(layers[layerIndex].Delta);


                lock (thisLock)
                {
                    if (passedParams.regularization == Regularizations.L2)
                    {

                        weightsUpdates[layerIndex - 1] = weightsUpdates[layerIndex - 1].Add((weightsUpdate - 2 * passedParams.regularizationRate * passedParams.network.Weights[layerIndex - 1]));

                    }
                    else
                    {

                        weightsUpdates[layerIndex - 1] = weightsUpdates[layerIndex - 1].Add(weightsUpdate);
                    }

                }
                if (passedParams.debug)
                {
                    Console.WriteLine("weights updates of weightsMatrix(learning rate* outerproduct(Layer{1} delta,layer{2} output from activations) ): {0} ", layerIndex - 1, layerIndex, layerIndex - 1);
                    Console.WriteLine("learning rate:{0}", passedParams.learningRate);
                    Console.WriteLine("Layer:{0} delta: {1}", layerIndex, layers[layerIndex].Delta);
                    Console.WriteLine("layer{0} output from activations:{1}", layerIndex - 1, layers[layerIndex - 1].LayerActivations);
                    Console.WriteLine(weightsUpdate);
                    Console.WriteLine("----------- Gradientdescent LayerIndex{0} ------------", layerIndex);

                }

            }//back propagating per layer

            if (passedParams.debug)
                Console.WriteLine("-------- Batch:{0} element:{1} end-----", batchIndex, elementIdxInBatch);


        }
    }
}
