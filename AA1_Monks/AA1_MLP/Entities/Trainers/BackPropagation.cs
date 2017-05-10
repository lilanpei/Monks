using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CustomExtensionMethods;
using MathNet.Numerics.LinearAlgebra;
using AA1_MLP.Enums;

namespace AA1_MLP.Entities.Trainers
{
    class BackPropagation : IOptimizer
    {
        public override List<double[]> Train(Network network, DataSet wholeData, double learningRate, int numberOfEpochs, bool shuffle = false, int? batchSize = null, float? validationSplit = null, IOptimizer.Historian historian = null, IOptimizer.CheckPointer checkPointer = null, bool debug = false, double regularizationRate = 0, Regularizations regularization = Regularizations.None, double momentum = 0, bool resilient = false, double resilientUpdateAccelerationRate = 1, double resilientUpdateSlowDownRate = 1)
        {
            List<double[]> learningCurve = new List<double[]>();
            List<int> indices = Enumerable.Range(0, wholeData.Labels.RowCount).ToList();
            if (shuffle)
            {
                indices.Shuffle();
            }

            DataSet validation = new DataSet(null, null);
            DataSet training = new DataSet(null, null);
            if (validationSplit != null)
            {
                int valSplitSize = (int)(indices.Count * validationSplit);

                validation.Inputs = CreateMatrix.Dense(valSplitSize, wholeData.Inputs.ColumnCount, 0.0);
                validation.Labels = CreateMatrix.Dense(valSplitSize, wholeData.Labels.ColumnCount, 0.0);

                for (int i = 0; i < valSplitSize; i++)
                {

                    validation.Inputs.SetRow(i, wholeData.Inputs.Row(indices[i]));// .SubMatrix(indices[i], 1, 0, wholeData.Inputs.ColumnCount));
                    validation.Labels.SetRow(i, wholeData.Labels.Row(indices[i]));//SubMatrix(indices[i], 1, 0, wholeData.Labels.ColumnCount));

                }




                training.Inputs = CreateMatrix.Dense(indices.Count - valSplitSize, wholeData.Inputs.ColumnCount, 0.0);
                training.Labels = CreateMatrix.Dense(indices.Count - valSplitSize, wholeData.Labels.ColumnCount, 0.0);


                for (int i = valSplitSize; i < indices.Count; i++)
                {
                    training.Inputs.SetRow(i - valSplitSize, wholeData.Inputs.Row(indices[i]));//, 1, 0, wholeData.Inputs.ColumnCount));
                    training.Labels.SetRow(i - valSplitSize, wholeData.Labels.Row(indices[i]));//.SubMatrix(indices[i], 1, 0, wholeData.Labels.ColumnCount));

                }






            }
            else
            {
                training.Inputs = wholeData.Inputs;
                training.Labels = wholeData.Labels;

            }

            Matrix<double> batchesIndices = null;//a 2d matrix of shape(nmberOfBatches,2), rows are batches, row[0] =barchstart, row[1] = batchEnd 
            Dictionary<int, Matrix<double>> previousWeightsUpdate = null;
            Dictionary<int, Matrix<double>> PreviousUpdateSigns = new Dictionary<int, Matrix<double>>();

            for (int epoch = 0; epoch < numberOfEpochs; epoch++)
            {


                if (batchSize != null)
                {
                    var numberOfBatches = (int)((training.Labels.RowCount / (double)(batchSize)));

                    batchesIndices = CreateMatrix.Dense(numberOfBatches, 2, 0.0);

                    for (int j = 0; j < numberOfBatches; j++)
                    {
                        batchesIndices.SetRow(j, new double[] { j * (double)batchSize, Math.Min(indices.Count, (j + 1) * (double)batchSize) });
                    }


                }
                else
                {
                    batchesIndices = CreateMatrix.Dense(1, 2, 0.0);
                    batchesIndices.SetRow(0, new double[] { 0, indices.Count - 1 });
                }

                double iterationLoss = 0;

                for (int i = 0; i < batchesIndices.RowCount; i++)//for each batch
                {

                    double batchLoss = 0;
                    Dictionary<int, Matrix<double>> weightsUpdates = new Dictionary<int, Matrix<double>>();
                    int numberOfBatchExamples = (((int)batchesIndices.Row(i).At(1) - (int)batchesIndices.Row(i).At(0)) + 1);
                    var batchIndices = Enumerable.Range((int)batchesIndices.Row(i).At(0), (int)batchesIndices.Row(i).At(1) + 1).ToList();
                    if (shuffle)
                    {
                        batchIndices.Shuffle();
                    }
                    foreach (int k in batchIndices)//for each elemnt in th batch
                    {
                        var nwOutput = network.ForwardPropagation(training.Inputs.Row(k));
                        // network.Layers[0].LayerActivationsSumInputs = training.Inputs.Row(k);
                        var label = training.Labels.Row(k);
                        //comute the loss 
                        //batchLoss += ((label - nwOutput.Map(s => s >= 0.5 ? 1.0 : 0.0)).PointwiseMultiply(label - nwOutput.Map(s => s > 0.5 ? 1.0 : 0.0))).Sum();
                        batchLoss += ((label - nwOutput).PointwiseMultiply(label - nwOutput)).Sum();
                        //batchLoss += -1 * label * (nwOutput.Map(f => Math.Log(f))) - (1 - label) * (1 - nwOutput.Map(f => Math.Log(f)));
                        var residual = label - nwOutput;
                        //var residual = -((label.PointwiseMultiply(nwOutput.Map(f => Math.Log(f)))) + (1 - label).PointwiseMultiply((nwOutput.Map(f => Math.Log(1 - f)))));

                        if (debug)
                        {
                            Console.WriteLine("Target:{0}", label);
                            Console.WriteLine("Calculated:{0}", nwOutput);
                            Console.WriteLine("Target-calculated (residual):{0}", residual);
                        }

                        residual = residual.Map(r => double.IsNaN(r) ? 0 : r);
                        //compute the error and backpropagate it 
                        //batchLoss += residual.Sum();
                        // network.Layers.Last().Delta = residual;
                        //compute the delta of previous layer
                        //Matrix<double> previousDeltaOptSum = null;
                        for (int layerIndex = network.Layers.Count - 1; layerIndex >= 1; layerIndex--)
                        {
                            if (debug)
                                Console.WriteLine("##### enting backpropagation layer index: {0} ######", layerIndex);
                            Vector<double> derivative = network.Layers[layerIndex].Activation.CalculateDerivative(network.Layers[layerIndex].LayerActivationsSumInputs);
                            /* if (network.Layers[layerIndex].Bias)
                             {
                                  derivative = network.Layers[layerIndex].Activation.CalculateDerivative(network.Layers[layerIndex].LayerActivationsSumInputs);

                             }
                             else
                             {
                                  derivative = network.Layers[layerIndex].Activation.CalculateDerivative(network.Layers[layerIndex].LayerActivationsSumInputs);

                             }*/
                            //  var residualTimesDerivative = residual.PointwiseMultiply(derivative);
                            if (debug)
                            {
                                Console.WriteLine("output sum(the sum inputted to the activation(LayerActivationsSumInputs)): {0}", network.Layers[layerIndex].LayerActivationsSumInputs);
                                Console.WriteLine("derivative: {0}", derivative);
                                Console.WriteLine("output sum margin of error(residual): {0}", residual);
                                Console.WriteLine("Delta output sum of Layer(residual*derivative): {0}", layerIndex);
                                //    Console.WriteLine(residualTimesDerivative);

                            }
                            if (layerIndex == network.Layers.Count - 1)
                            {
                                network.Layers[layerIndex].Delta = residual.PointwiseMultiply(derivative);

                            }
                            else
                            {
                                Matrix<double> wei = network.Weights[layerIndex];
                                if (wei.RowCount > derivative.Count)
                                {
                                    wei = wei.SubMatrix(0, wei.RowCount - 1, 0, wei.ColumnCount);
                                }

                                network.Layers[layerIndex].Delta = (wei * network.Layers[layerIndex + 1].Delta).PointwiseMultiply(derivative);
                            }
                            //if (layerIndex != 1)
                            //{
                            //    /*    Matrix<double> nwWeights = null;
                            //        if (network.Layers[layerIndex - 1].Bias)
                            //        {
                            //            nwWeights = network.Weights[layerIndex - 1].SubMatrix(1, network.Weights[layerIndex - 1].RowCount - 1, 0, network.Weights[layerIndex - 1].ColumnCount);
                            //        }
                            //        else
                            //        {
                            //            nwWeights = network.Weights[layerIndex - 1];

                            //        }*/
                            //    var der = (network.Layers[layerIndex].Activation.CalculateDerivative(network.Layers[layerIndex].LayerActivationsSumInputs));
                            //    network.Layers[layerIndex - 1].Delta = network.Layers[layerIndex].Delta.PointwiseMultiply( der);


                            //}

                            //   residual = network.Layers[layerIndex - 1].Delta;

                            // network.Weights[layerIndex - 1] -= LearningRate * network.Layers[layerIndex].Delta.OuterProduct( network.Layers[layerIndex - 1].LayerActivationsSumInputs);
                            if (!weightsUpdates.ContainsKey(layerIndex - 1))
                            {
                                weightsUpdates.Add(layerIndex - 1, CreateMatrix.Dense(network.Weights[layerIndex - 1].RowCount, network.Weights[layerIndex - 1].ColumnCount, 0.0));
                            }
                            Console.ForegroundColor = ConsoleColor.Red;
                            Console.WriteLine("Weights layer:{0} {1}", layerIndex - 1, network.Weights[layerIndex - 1]);
                            Console.ResetColor();
                            Matrix<double> weightsUpdate = null;
                            var acti = network.Layers[layerIndex - 1].LayerActivations;
                            if (network.Layers[layerIndex - 1].Bias && acti.Count - network.Layers[layerIndex - 1].NumberOfNeurons < 1)
                            {
                                var l = acti.ToList();
                                l.Add(1);

                                acti = CreateVector.Dense(l.ToArray());
                            }
                            weightsUpdate = acti.OuterProduct(network.Layers[layerIndex].Delta);
                            //  if (layerIndex == network.Layers.Count - 1)
                            {
                                //delta output sum * hidden layer results

                                // outrprod = network.Layers[layerIndex].Delta.Vec2Vecmultiply(network.Layers[layerIndex - 1].LayerActivations);
                            }
                            // if (layerIndex == network.Layers.Count - 1)
                            {
                                // previousDeltaOptSum = network.Layers[layerIndex].Activation.CalculateDerivative(network.Layers[layerIndex].LayerActivationsSumInputs).Vec2Vecmultiply(network.Layers[layerIndex].Delta);
                                //weightsUpdate = network.Layers[layerIndex - 1].LayerActivations.OuterProduct(network.Layers[layerIndex].Delta);
                                //(network.Layers[layerIndex].LayerActivations).Vec2Mtrxmultiply(previousDeltaOptSum).Transpose();

                            }
                            /*   else
                               {
                                   previousDeltaOptSum = previousDeltaOptSum * network.Weights[layerIndex].Transpose().Mtrx2Vecmultiply(network.Layers[layerIndex].Activation.CalculateDerivative(network.Layers[layerIndex].LayerActivationsSumInputs));
                                   weightsUpdate = network.Layers[layerIndex].LayerActivationsSumInputs.Vec2Mtrxmultiply(previousDeltaOptSum);
                               }*/


                            // outrprod = network.Layers[layerIndex].Delta.Vec2Mtrxmultiply(network.Weights[layerIndex].Mtrx2Vecmultiply((network.Layers[layerIndex].Activation.CalculateDerivative(network.Layers[layerIndex].LayerActivationsSumInputs)))).Mtrx2Vecmultiply(network.Layers[layerIndex - 1].LayerActivations);

                            // else if (layerIndex == 1) { break; }
                            /*  else
                              {
                                  //delta output sum * hidden-to-outer weights * S'(hidden sum)
                              }
                              */

                            weightsUpdates[layerIndex - 1] = weightsUpdates[layerIndex - 1].Add(weightsUpdate);
                            // weightsUpdates[layerIndex - 1] = weightsupdatematrix;
                            if (debug)
                            {
                                Console.WriteLine("weights updates of weightsMatrix(learning rate* outerproduct(Layer{1} delta,layer{2} output from activations) ): {0} ", layerIndex - 1, layerIndex, layerIndex - 1);
                                Console.WriteLine("learning rate:{0}", learningRate);
                                Console.WriteLine("Layer:{0} delta: {1}", layerIndex, network.Layers[layerIndex].Delta);
                                Console.WriteLine("layer{0} output from activations:{1}", layerIndex - 1, network.Layers[layerIndex - 1].LayerActivations);
                                Console.WriteLine(weightsUpdate);
                                Console.WriteLine("----------- BackPropagation LayerIndex{0} ------------", layerIndex);

                            }

                        }

                        if (debug)
                            Console.WriteLine("-------- Batch:{0} element:{1} end-----", i, k);
                    }
                    if (debug)
                        Console.WriteLine("batch end");

                    //EpochBatchesLosses.Add(new double[] { batchLoss / numberOfBatchExamples });
                    // batchLoss /= (((int)batchesIndices.Row(i).At(1) - (int)batchesIndices.Row(i).At(0)) + 1);

                    for (int y = 0; y < weightsUpdates.Keys.Count; y++)
                    {
                        var resilientLearningRates = CreateMatrix.Dense(network.Weights[y].RowCount, network.Weights[y].ColumnCount, learningRate);

                        if (resilient && PreviousUpdateSigns.ContainsKey(y))
                        {
                            var currentUpdateSigns = weightsUpdates[y].PointwiseSign();
                            resilientLearningRates = PreviousUpdateSigns[y].PointwiseMultiply(currentUpdateSigns).Map(s => s > 0 ? learningRate * resilientUpdateAccelerationRate : learningRate * resilientUpdateSlowDownRate);
                        }


                        var momentumUpdate = CreateMatrix.Dense(network.Weights[y].RowCount, network.Weights[y].ColumnCount, 0.0);
                        if (previousWeightsUpdate != null)
                        {
                            momentumUpdate += momentum * previousWeightsUpdate[y];
                        }
                        if (regularization != Regularizations.None)
                        {
                            network.Weights[y] = momentumUpdate + (((1 - resilientLearningRates * regularizationRate / (((int)batchesIndices.Row(i).At(1) - (int)batchesIndices.Row(i).At(0)) + 1))).PointwiseMultiply(network.Weights[y]) + resilientLearningRates.PointwiseMultiply(weightsUpdates[y]) / (((int)batchesIndices.Row(i).At(1) - (int)batchesIndices.Row(i).At(0)) + 1));

                        }
                        else
                            network.Weights[y] += resilientLearningRates.PointwiseMultiply(momentumUpdate + weightsUpdates[y]) / numberOfBatchExamples;

                        if (!PreviousUpdateSigns.ContainsKey(y))
                        {
                            PreviousUpdateSigns.Add(y, null);
                        }
                        PreviousUpdateSigns[y] = weightsUpdates[y].PointwiseSign();
                    }
                    previousWeightsUpdate = weightsUpdates;

                    iterationLoss += batchLoss/((int)batchesIndices.Row(i).At(1)-(int)batchesIndices.Row(i).At(0)+1);
                    Console.WriteLine("Batch: {0} Error: {1}", i, batchLoss);
                }


                iterationLoss /= batchesIndices.RowCount;
                learningCurve.Add(new double[] { iterationLoss });
                Console.ForegroundColor = ConsoleColor.Green;
                Console.WriteLine("Epoch:{0} loss:{1}", epoch, iterationLoss);
                Console.ResetColor();
            }
            return learningCurve;
        }



    }
}
