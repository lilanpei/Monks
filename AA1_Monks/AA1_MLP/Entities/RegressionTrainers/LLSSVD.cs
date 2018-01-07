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
            var watch = System.Diagnostics.Stopwatch.StartNew();
            var svd = trainParams.trainingSet.Inputs.Svd();
            int r = trainParams.trainingSet.Inputs.Rank();


/*            while (r < trainParams.trainingSet.Inputs.ColumnCount && svd.S.At(r) >= Math.Max(trainParams.trainingSet.Inputs.RowCount, trainParams.trainingSet.Inputs.ColumnCount) * passedParams.eps * svd.S[0])
                r++;*/


         //   Console.WriteLine("Rank r is:{0}", r);
            var d = svd.U.Transpose().Multiply(trainParams.trainingSet.Labels);
         //   var temp = CreateMatrix.Dense<double>(svd.VT.ColumnCount, trainParams.trainingSet.Labels.ColumnCount);
            //temp.SetSubMatrix(0, 0, d.SubMatrix(0, r, 0, trainParams.trainingSet.Labels.ColumnCount).Column(0).PointwiseDivide(svd.S.SubVector(0, r)).ToRowMatrix().Transpose());
            var sbmtrx = (d.SubMatrix(0, r, 0, trainParams.trainingSet.Labels.ColumnCount));
            sbmtrx.SetColumn(0, sbmtrx.Column(0).PointwiseDivide(svd.S.SubVector(0, r)));
            sbmtrx.SetColumn(1, sbmtrx.Column(1).PointwiseDivide(svd.S.SubVector(0, r)));

            passedParams.model.Weights = svd.VT.Transpose().Multiply(sbmtrx);
            watch.Stop();
            var elapsedMs = watch.ElapsedMilliseconds;
            Console.WriteLine("elapsed Time:{0} ms", elapsedMs);

            /*
              var sbmtrx = (d.SubMatrix(0, r, 0, trainParams.trainingSet.Labels.ColumnCount));
                sbmtrx.SetColumn(0,sbmtrx.Column(0).PointwiseDivide(svd.S.SubVector(0, r)));
                sbmtrx.SetColumn(1, sbmtrx.Column(1).PointwiseDivide(svd.S.SubVector(0, r)));

           // temp.SetSubMatrix(0, 0, sbmtrx.Transpose());
             
             */


            var cost = CostFunction(trainParams.trainingSet.Inputs, trainParams.trainingSet.Labels, passedParams.model.Weights);
            var valCost = CostFunction(trainParams.validationSet.Inputs, trainParams.validationSet.Labels, passedParams.model.Weights);
            Console.WriteLine("trainCost:{0},ValCost:{1}", cost, valCost);
            Console.WriteLine(Score(trainParams.validationSet.Inputs, trainParams.validationSet.Labels, passedParams.model.Weights));
            return new List<double[]> { { new double[] { cost, valCost } } };
        }

        double Score(Matrix<double> data, Matrix<double> targets, Matrix<double> weights)
        {
            Vector<double> u , v ;
            u = (targets - data.Multiply(weights)).PointwisePower(2).ColumnSums();
            var firstmean = targets.Column(0).Sum() / targets.Column(0).Count;
            var secondmean = targets.Column(1).Sum() / targets.Column(1).Count;

            v =CreateVector.Dense<double>( new double[]{ 
                (targets.Column(0) - firstmean).PointwisePower(2).Sum() , (targets.Column(1) - secondmean).PointwisePower(2).Sum()
            });
            return (1 - u.PointwiseDivide( v)).Average();
        }
        double CostFunction(Matrix<double> data, Matrix<double> targets, Matrix<double> weights)
        {

            return ((data.Multiply(weights) - targets).PointwisePower(2).RowSums()/  (2 * targets.RowCount)).Average();//(||Ax-b||^2)/(2n)

        }
    }
}
