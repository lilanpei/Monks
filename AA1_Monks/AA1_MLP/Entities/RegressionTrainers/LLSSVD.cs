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
    public class LLSSVD : IOptimizer
    {


        public override List<double[]> Train(TrainersParams.TrainerParams trainParams)
        {
            LinearLeastSquaresParams passedParams = (LinearLeastSquaresParams)trainParams;

            var svd = trainParams.trainingSet.Inputs.Svd();
            int r = 1;//trainParams.trainingSet.Inputs.Rank();


            while (r < trainParams.trainingSet.Inputs.ColumnCount && svd.S.At(r ) >= Math.Max(trainParams.trainingSet.Inputs.RowCount, trainParams.trainingSet.Inputs.ColumnCount) * passedParams.eps * svd.S[0])
                r++;


            Console.WriteLine("Rank r is:{0}", r);
            var d = svd.U.Transpose().Multiply(trainParams.trainingSet.Labels);
            var temp = CreateMatrix.Dense<double>(svd.VT.ColumnCount, 1);
            temp.SetSubMatrix(0, 0, d.SubMatrix(0, r, 0, 1).Column(0).PointwiseDivide(svd.S.SubVector(0, r)).ToRowMatrix().Transpose());
            passedParams.model.Weights = svd.VT.Transpose().Multiply(temp);




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
