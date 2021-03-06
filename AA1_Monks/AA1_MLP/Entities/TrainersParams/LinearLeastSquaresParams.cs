﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using AA1_MLP.Enums;
using AA1_MLP.Entities.Linear;

namespace AA1_MLP.Entities.TrainersParams
{
   public class LinearLeastSquaresParams:TrainerParams
    {
        public double learningRate=1.0,tol=0.001;
        public bool fit_intercept=true,normalize=true,copy_X=true;
        public int? numOfIterations = null;
        public double regularizationRate = 0.001;
        public Regularizations regularizationType = Regularizations.L2;
        public LinearModel model;
        public double eps= 2.2204e-16;


        public int degree = 1;
    }
}
