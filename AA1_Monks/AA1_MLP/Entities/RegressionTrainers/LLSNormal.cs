using AA1_MLP.Entities.TrainersParams;
using MathNet.Numerics.LinearAlgebra;
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
    public class LLSNormal : AA1_MLP.Entities.Trainers.IOptimizer
    {


        public override List<double[]> Train(TrainersParams.ITrainerParams trainParams)
        {
            LinearLeastSquaresParams passedParams = (LinearLeastSquaresParams)trainParams;






            throw new NotImplementedException();
        }


    }
}
