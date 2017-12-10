using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AA1_MLP.Entities.TrainersParams
{
    public class ITrainerParams
    {
        public DataSet trainingSet;
        public bool shuffle = false;
        public bool debug = false;
        public DataSet validationSet = null;
        public double? trueThreshold = 0.5;
        public bool MEE = false;

    }
}
