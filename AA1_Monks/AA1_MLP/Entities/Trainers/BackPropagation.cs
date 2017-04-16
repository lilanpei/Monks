using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CustomExtensionMethods;
using MathNet.Numerics.LinearAlgebra;
namespace AA1_MLP.Entities.Trainers
{
    class BackPropagation : IOptimizer
    {
        public override void Train(Network network, DataSet wholeData, double learningRate, int numberOfEpochs, bool shuffle = false, int? batchSize = null, float? validationSplit = null, IOptimizer.Historian historian = null, IOptimizer.CheckPointer checkPointer = null)
        {
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

            Matrix<double> batchesIndices = null;

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

                    for (int k = (int)batchesIndices.Row(i).At(0); k < (int)batchesIndices.Row(i).At(1); k++)//for each elemnt in th batch
                    {
                        var nwOutput = network.ForwardPropagation(training.Inputs.Row(k));
                        // network.Layers[0].LayerActivations = training.Inputs.Row(k);
                        var label = training.Labels.Row(k);
                        //comute the loss 
                        //batchLoss += -1 * label * (nwOutput.Map(f => Math.Log(f))) - (1 - label) * (1 - nwOutput.Map(f => Math.Log(f)));
                        // var residual = nwOutput - label;

                        var residual = -((label.PointwiseMultiply(nwOutput.Map(f => Math.Log(f)))) + (1 - label).PointwiseMultiply((nwOutput.Map(f => Math.Log(1 - f)))));
                        residual = residual.Map(r => double.IsNaN(r) ? 0 : r);
                        //compute the error and backpropagate it 
                        batchLoss += residual.Sum();
                        network.Layers.Last().Delta = residual;
                        //compute the delta of previous layer
                        for (int layerIndex = network.Layers.Count - 1; layerIndex >= 1; layerIndex--)
                        {

                            network.Layers[layerIndex - 1].Delta = (residual * network.Layers[layerIndex].Activation.CalculateDerivative(network.Layers[layerIndex].LayerActivations) * network.Weights[layerIndex - 1].Transpose());
                            residual = network.Layers[layerIndex - 1].Delta;

                            // network.Weights[layerIndex - 1] -= LearningRate * network.Layers[layerIndex].Delta.OuterProduct( network.Layers[layerIndex - 1].LayerActivations);
                            if (!weightsUpdates.ContainsKey(layerIndex - 1))
                            {
                                weightsUpdates.Add(layerIndex - 1, CreateMatrix.Dense(network.Weights[layerIndex - 1].RowCount, network.Weights[layerIndex - 1].ColumnCount, 0.0));
                            }
                            var outrprod = network.Layers[layerIndex].Delta.OuterProduct(network.Layers[layerIndex - 1].LayerActivations);
                            weightsUpdates[layerIndex - 1] = weightsUpdates[layerIndex - 1].Add(learningRate * outrprod.Transpose());
                        }



                    }

                    batchLoss /= ((int)batchesIndices.Row(i).At(1) - (int)batchesIndices.Row(i).At(0)) + 1;

                    for (int y = 0; y < weightsUpdates.Keys.Count; y++)
                    {
                        network.Weights[y] += weightsUpdates[y];
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
