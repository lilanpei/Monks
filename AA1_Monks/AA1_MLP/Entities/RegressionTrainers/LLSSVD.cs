using AA1_MLP.Entities.Trainers;
using AA1_MLP.Entities.TrainersParams;
using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AA1_MLP.Entities.RegressionTrainers
{
    class LLSSVD : IOptimizer
    {


        public override List<double[]> Train(TrainersParams.TrainerParams trainParams)
        {
            LinearLeastSquaresParams passedParams = (LinearLeastSquaresParams)trainParams;

            var svd = trainParams.trainingSet.Inputs.Svd();
            int r = 1;

            while (r < trainParams.trainingSet.Inputs.ColumnCount && svd.S.At(r + 1) >= Math.Max(trainParams.trainingSet.Inputs.RowCount, trainParams.trainingSet.Inputs.ColumnCount) * passedParams.eps * svd.S[0])
                r++;

            var d = svd.U.Transpose().Multiply(trainParams.trainingSet.Labels);
            //var temp = CreateMatrix.Dense<double>();
            //passedParams.model.Weights = svd.VT.Multiply(d.SubMatrix(0,r-1,0,1).Column(0).PointwiseDivide(svd.S));

            //need to check if A has linearly independent rows or columns to properly use the pseudo inverse otherwise we might have a problem!

            //Ax=b  ->   A'Ax=A'b -> x = (inv(A'A))A'b ->pinv(A)b  where A is our data, b is our vector of targets 

            //passedParams.model.Weights = trainParams.trainingSet.Inputs.PseudoInverse().Multiply(trainParams.trainingSet.Labels);

            var cost = CostFunction(trainParams.trainingSet.Inputs, trainParams.trainingSet.Labels, passedParams.model.Weights);
            var valCost = CostFunction(trainParams.validationSet.Inputs, trainParams.validationSet.Labels, passedParams.model.Weights);
            Console.WriteLine("trainCost:{0},ValCost:{1}", cost, valCost);
            return new List<double[]> { { new double[] { cost, valCost } } };
        }

        double CostFunction(Matrix<double> data, Matrix<double> targets, Matrix<double> weights)
        {

            return (data.Multiply(weights) - targets).PointwisePower(2).RowSums().Sum() / (2 * targets.RowCount);//(||Ax-b||^2)/(2n)

        }
    }
}
