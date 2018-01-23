using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AA1_MLP.Entities.TrainersParams
{
    public class AdamParams : INeuralTrainerParams
    {

        public double beta1 = 0.9;
        public double beta2 = 0.999;
        public double epsilon = 1e-08;

    }
}
