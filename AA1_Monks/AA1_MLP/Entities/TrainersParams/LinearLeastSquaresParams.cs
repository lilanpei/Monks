using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using AA1_MLP.Enums;

namespace AA1_MLP.Entities.TrainersParams
{
   public class LinearLeastSquaresParams:ITrainerParams
    {
        public double learningRate=1.0,tol=0.001;
        public bool fit_intercept=true,normalize=true,copy_X=true;
        public int? numOfIterations = null;

    }
}
