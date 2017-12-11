using AA1_MLP.Entities.TrainersParams;
using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AA1_MLP.Entities.Regression
{
    /// <summary>
    /// our basic linear least squares model
    /// </summary>
    public class LLSGradientDescent : AA1_MLP.Entities.Trainers.IOptimizer
    {


        public override List<double[]> Train(TrainersParams.ITrainerParams trainParams)
        {
            LinearLeastSquaresParams passedParams = (LinearLeastSquaresParams)trainParams;
            //should make the bias a passed param?

            int numberOfExamples = trainParams.trainingSet.Labels.RowCount;
            //adding the bias column o fones to the training trainingdataWithBias
            int numberOfColumns = 1 + trainParams.trainingSet.Inputs.ColumnCount;//+1 for the bias
            Matrix<double> trainingdataWithBias = CreateMatrix.Dense(numberOfExamples, numberOfColumns, 0.0);

            for (int i = 0; i < trainParams.trainingSet.Inputs.RowCount; i++)
            {
                double[] row = new double[numberOfColumns];
                row[0] = 1;
                for (int j = 1; j < numberOfColumns; j++)
                {
                    row[j] = trainParams.trainingSet.Inputs[i, j - 1];//j starts from one because we set the first element on its own, but the training set requires it to count from 0, so the -1 in the indexer

                }
                trainingdataWithBias.SetRow(i, row);

            }


            Matrix<double> weights = CreateMatrix.Random<double>(numberOfColumns, 1, new ContinuousUniform(-0.7, 0.7));


            List<double[]> lossHistory = new List<double[]>();

            for (int i = 0; i < passedParams.numOfIterations; i++)
            {
                var hypothesis = trainingdataWithBias.Multiply(weights);
                var loss = hypothesis - trainParams.trainingSet.Labels;
                var gradient = trainingdataWithBias.Transpose().Multiply(loss) / numberOfExamples;

                //updating the weights

                weights -= passedParams.learningRate * gradient;
                var cost = CostFunction(trainingdataWithBias, trainParams.trainingSet.Labels, weights);
                Console.WriteLine("iteration:{0}:{1}", i, cost);
                lossHistory.Append(new double[] { cost });
            }


            return lossHistory;
        }

        double CostFunction(Matrix<double> data, Matrix<double> targets, Matrix<double> weights)
        {

            return (data.Multiply(weights) - targets).PointwisePower(2).RowSums().Sum() / (2 * targets.RowCount);

        }

    }
}
