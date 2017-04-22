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
        public override void Train(Network network, DataSet wholeData, double learningRate, int numberOfEpochs, bool shuffle = false, int? batchSize = null, float? validationSplit = null, IOptimizer.Historian historian = null, IOptimizer.CheckPointer checkPointer = null, bool debug = false, double regularizationRate = 0, Regularizations regularization = Regularizations.None)
        {
            List<int> indices = Enumerable.Range(0, wholeData.Labels.RowCount ).ToList();
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
                    batchesIndices.SetRow(0, new double[] { 0, indices.Count -1});
                }

                double iterationLoss = 0;

                for (int i = 0; i < batchesIndices.RowCount; i++)//for each batch
                {

                    double batchLoss = 0;
                    Dictionary<int, Matrix<double>> weightsUpdates = new Dictionary<int, Matrix<double>>();

                    for (int k = (int)batchesIndices.Row(i).At(0); k <= (int)batchesIndices.Row(i).At(1); k++)//for each elemnt in th batch
                    {
                        var nwOutput = network.ForwardPropagation(training.Inputs.Row(k));
                        // network.Layers[0].LayerActivationsSumInputs = training.Inputs.Row(k);
                        var label = training.Labels.Row(k);
                        //comute the loss 
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
                        batchLoss += residual.Sum();
                        // network.Layers.Last().Delta = residual;
                        //compute the delta of previous layer
                        for (int layerIndex = network.Layers.Count - 1; layerIndex >= 1; layerIndex--)
                        {
                            if (debug)
                                Console.WriteLine("##### enting backpropagation layer index: {0} ######", layerIndex);

                            var derivative = network.Layers[layerIndex].Activation.CalculateDerivative(network.Layers[layerIndex].LayerActivationsSumInputs);
                          //  var residualTimesDerivative = residual.PointwiseMultiply(derivative);
                            if (debug)
                            {
                                Console.WriteLine("output sum(the sum inputted to the activation(LayerActivationsSumInputs)): {0}", network.Layers[layerIndex].LayerActivationsSumInputs);
                                Console.WriteLine("derivative: {0}", derivative);
                                Console.WriteLine("output sum margin of error(residual): {0}", residual);
                                Console.WriteLine("Delta output sum of Layer(residual*derivative): {0}", layerIndex);
                            //    Console.WriteLine(residualTimesDerivative);

                            }
                            network.Layers[layerIndex].Delta = residual;
                            if (layerIndex != 1)
                            {
                                network.Layers[layerIndex - 1].Delta = network.Layers[layerIndex].Delta * ((network.Layers[layerIndex - 1].Bias? network.Weights[layerIndex - 1].SubMatrix(1, network.Weights[layerIndex - 1].RowCount-1,0, network.Weights[layerIndex - 1].ColumnCount) : network.Weights[layerIndex - 1] )* (network.Layers[layerIndex].Activation.CalculateDerivative(network.Layers[layerIndex].LayerActivationsSumInputs))).ToRowMatrix();

                            }
    
                            residual = network.Layers[layerIndex - 1].Delta;

                            // network.Weights[layerIndex - 1] -= LearningRate * network.Layers[layerIndex].Delta.OuterProduct( network.Layers[layerIndex - 1].LayerActivationsSumInputs);
                            if (!weightsUpdates.ContainsKey(layerIndex - 1))
                            {
                                weightsUpdates.Add(layerIndex - 1, CreateMatrix.Dense(network.Weights[layerIndex - 1].RowCount, network.Weights[layerIndex - 1].ColumnCount, 0.0));
                            }
                            Console.ForegroundColor = ConsoleColor.Red;
                            Console.WriteLine("Weights layer:{0} {1}", layerIndex - 1, network.Weights[layerIndex - 1]);
                            Console.ResetColor();
                            Matrix<double> outrprod = null;
                            //  if (layerIndex == network.Layers.Count - 1)
                            {
                                //delta output sum * hidden layer results

                                outrprod = network.Layers[layerIndex].Delta.Vec2Vecmultiply(network.Layers[layerIndex - 1].LayerActivations);
                            }
                            // else if (layerIndex == 1) { break; }
                            /*  else
                              {
                                  //delta output sum * hidden-to-outer weights * S'(hidden sum)
                                  outrprod = network.Layers[layerIndex].Delta.Vec2Mtrxmultiply(network.Weights[layerIndex ].Mtrx2Vecmultiply((network.Layers[layerIndex].Activation.CalculateDerivative(network.Layers[layerIndex].LayerActivationsSumInputs)))).Mtrx2Vecmultiply( network.Layers[layerIndex-1].LayerActivations);
                              }
                              */
                              
                            weightsUpdates[layerIndex - 1] = weightsUpdates[layerIndex - 1].Add(outrprod);
                            // weightsUpdates[layerIndex - 1] = weightsupdatematrix;
                            if (debug)
                            {
                                Console.WriteLine("weights updates of weightsMatrix(learning rate* outerproduct(Layer{1} delta,layer{2} output from activations) ): {0} ", layerIndex - 1, layerIndex, layerIndex - 1);
                                Console.WriteLine("learning rate:{0}", learningRate);
                                Console.WriteLine("Layer:{0} delta: {1}", layerIndex, network.Layers[layerIndex].Delta);
                                Console.WriteLine("layer{0} output from activations:{1}", layerIndex - 1, network.Layers[layerIndex - 1].LayerActivations);
                                Console.WriteLine(outrprod);
                                Console.WriteLine("----------- BackPropagation LayerIndex{0} ------------", layerIndex);

                            }
                           
                        }

                        if (debug)
                            Console.WriteLine("-------- Batch:{0} element:{1} end-----", i, k);
                    }
                    if (debug)
                        Console.WriteLine("batch end");

                    batchLoss /= (((int)batchesIndices.Row(i).At(1) - (int)batchesIndices.Row(i).At(0)) + 1);

                    for (int y = 0; y < weightsUpdates.Keys.Count; y++)
                    {
                        if (regularization != Regularizations.None)
                        {
                            network.Weights[y] = ((-1 + learningRate * regularizationRate/ (((int)batchesIndices.Row(i).At(1) - (int)batchesIndices.Row(i).At(0)) + 1)) * network.Weights[y] + learningRate * weightsUpdates[y]/ (((int)batchesIndices.Row(i).At(1) - (int)batchesIndices.Row(i).At(0)) + 1));

                        }
                        else
                            network.Weights[y] += learningRate* weightsUpdates[y] /( ((int)batchesIndices.Row(i).At(1) - (int)batchesIndices.Row(i).At(0)) + 1);

                    }
                    iterationLoss += batchLoss;///((int)batchesIndices.Row(i).At(1)-(int)batchesIndices.Row(i).At(0));
                    Console.WriteLine("Batch: {0} Error: {1}", i, batchLoss);
                }


                iterationLoss /= batchesIndices.RowCount;
                Console.ForegroundColor = ConsoleColor.Green;
                Console.WriteLine("Epoch:{0} loss:{1}", epoch, iterationLoss);
                Console.ResetColor();
            }
        }



    }
}
