using AA1_MLP.Entities.TrainersParams;
using AA1_MLP.Enums;
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


        public override List<double[]> Train(TrainersParams.TrainerParams trainParams)
        {
            LinearLeastSquaresParams passedParams = (LinearLeastSquaresParams)trainParams;
            //should make the bias a passed param?

            int trainingNumberOfExamples = trainParams.trainingSet.Labels.RowCount;
            int numberOfDataColumns = passedParams.degree > 1 ? 1 + passedParams.degree * trainParams.trainingSet.Inputs.ColumnCount : 1 + trainParams.trainingSet.Inputs.ColumnCount;//+1 for the bias


            //adding the bias column o fones to the training trainingdataWithBias

            Matrix<double> trainingdataWithBias = CreateMatrix.Dense(trainingNumberOfExamples, numberOfDataColumns, 0.0);
            Matrix<double> validationdataWithBias = CreateMatrix.Dense(passedParams.validationSet.Labels.RowCount, numberOfDataColumns, 0.0);

            for (int i = 0; i < trainParams.trainingSet.Inputs.RowCount; i++)
            {
                double[] row = new double[numberOfDataColumns];
                row[0] = 1;
                for (int j = 1; j <= trainParams.trainingSet.Inputs.ColumnCount; j++)
                {
                    row[j] = trainParams.trainingSet.Inputs[i, j - 1];//j starts from one because we set the first element on its own, but the training set requires it to count from 0, so the -1 in the indexer
                    for (int k = 1; k < passedParams.degree; k++)
                    {
                        row[k * trainParams.trainingSet.Inputs.ColumnCount + j] = Math.Pow(row[j], k + 1);
                    }
                }

                trainingdataWithBias.SetRow(i, row);

            }

            for (int i = 0; i < trainParams.validationSet.Inputs.RowCount; i++)
            {
                double[] row = new double[numberOfDataColumns];
                row[0] = 1;
                for (int j = 1; j <= trainParams.validationSet.Inputs.ColumnCount; j++)
                {
                    row[j] = trainParams.validationSet.Inputs[i, j - 1];//j starts from one because we set the first element on its own, but the training set requires it to count from 0, so the -1 in the indexer
                    for (int k = 1; k < passedParams.degree; k++)
                    {
                        row[k * trainParams.validationSet.Inputs.ColumnCount + j] = Math.Pow(row[j], k+1);
                    }
                }
                validationdataWithBias.SetRow(i, row);

            }


            Matrix<double> weights = CreateMatrix.Random<double>(numberOfDataColumns, 1, new Normal(0, 1));


            List<double[]> lossHistory = new List<double[]>();

            for (int i = 0; i < passedParams.numOfIterations; i++)
            {
                var hypothesis = trainingdataWithBias.Multiply(weights);//Ax  -> where A is our data matrix and x is our parameters vector
                var loss = hypothesis - trainParams.trainingSet.Labels; //(Ax-b)
                var gradient = trainingdataWithBias.Transpose().Multiply(loss) / trainingNumberOfExamples; //A'*(Ax-b)/n  -> the gradient of (||Ax-b||^2)/(2n) what we are minimizing

                //updating the weights

                if (passedParams.regularizationType == Regularizations.L2)
                {
                    weights -= passedParams.learningRate * gradient + passedParams.regularizationRate * weights;

                }
                else
                {
                    weights -= passedParams.learningRate * gradient;

                }
                var cost = CostFunction(trainingdataWithBias, trainParams.trainingSet.Labels, weights);
                var valCost = CostFunction(validationdataWithBias, trainParams.validationSet.Labels, weights);

                Console.WriteLine("iteration:{0},{1},{2}", i, cost, valCost);
                lossHistory.Add(new double[] { cost, valCost });
            }


            return lossHistory;
        }

        double CostFunction(Matrix<double> data, Matrix<double> targets, Matrix<double> weights)
        {

            return (data.Multiply(weights) - targets).PointwisePower(2).RowSums().Sum() / (2 * targets.RowCount);//(||Ax-b||^2)/(2n)

        }

    }
}
