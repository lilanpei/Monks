using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CustomExtensionMethods;
using MathNet.Numerics.LinearAlgebra;
namespace AA1_MLP.Entities
{
    class BackPropagation : IOptimizer
    {
        public override void Train(Network network, DataSet wholeData, int numberOfEpochs, bool shuffle = false, int? batchSize = null, float? validationSplit = null, IOptimizer.Historian historian = null, IOptimizer.CheckPointer checkPointer = null)
        {
            List<int> indices = Enumerable.Range(0, wholeData.Labels.RowCount - 1).ToList();
            if (shuffle)
            {
                indices.Shuffle();
            }

            DataSet validation = new DataSet();
            if (validationSplit != null)
            {
                for (int i = 0; i < indices.Count * validationSplit; i++)
                {
                    validation.Inputs.Append(wholeData.Inputs.SubMatrix(indices[i], 1, 0, wholeData.Inputs.ColumnCount - 1));
                    validation.Labels.Append(wholeData.Labels.SubMatrix(indices[i], 1, 0, wholeData.Labels.ColumnCount - 1));

                }
                for (int i = (int)(indices.Count * validationSplit); i < indices.Count; i++)
                {
                    wholeData.Inputs.Append(wholeData.Inputs.SubMatrix(indices[i], 1, 0, wholeData.Inputs.ColumnCount - 1));
                    wholeData.Labels.Append(wholeData.Labels.SubMatrix(indices[i], 1, 0, wholeData.Labels.ColumnCount - 1));

                }

            }

            Matrix<double> batchesIndices = null;

            for (int epoch = 0; epoch < numberOfEpochs; epoch++)
            {
                if (batchSize != null)
                {
                    var numberOfBatches = (int)(Math.Ceiling(wholeData.Labels.RowCount / (double)(batchSize)));

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

                    for (int k = (int)batchesIndices.Row(i).At(0); k < (int)batchesIndices.Row(i).At(1); k++)//for each elemnt in th batch
                    {
                        var nwOutput = network.ForwardPropagation(wholeData.Inputs.Row(k));
                        var label = wholeData.Labels.Row(k);
                        //comute the loss 
                        batchLoss += -1 * label * (nwOutput.Map(f => Math.Log(f))) - (1 - label) * (1 - nwOutput.Map(f => Math.Log(f)));


                        //compute the error and backpropagate it 

                        var residual =   nwOutput -label;

                        //compute the delta of previous layer
                        for (int layerIndex = network.Layers.Count - 1; layerIndex >= 0; layerIndex--)
                        {
                            network.Layers[layerIndex - 1].Delta = (
                            residual * network.Layers[layerIndex].Activation.CalculateDerivative(
                                network.Layers.[layerIndex].LayerActivations
                                 )*network.Weights[layerIndex-1].Transpose()
    );
                            residual = network.Layers[layerIndex - 1].Delta;

                            network.Weights[layerIndex-1]  -= LearningRate* network.Layers[layerIndex ].Delta*network.Layers[layerIndex - 1].Activation;
                        }

                    }
                        iterationLoss += batchLoss;///((int)batchesIndices.Row(i).At(1)-(int)batchesIndices.Row(i).At(0));

                }
                iterationLoss/= batchesIndices.RowCount;

            }
        }



    }
}
