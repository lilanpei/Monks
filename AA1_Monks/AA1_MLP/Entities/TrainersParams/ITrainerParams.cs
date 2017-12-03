using AA1_MLP.Enums;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace AA1_MLP.Entities.TrainersParams
{
    public class ITrainerParams
    {


        public Network network;
        public DataSet trainingSet;
        public double learningRate;
        public int numberOfEpochs;
        public bool shuffle = false;
        public int? batchSize = null;
        public bool debug = false;
        public double regularizationRate = 0;
        public Regularizations regularization = Regularizations.None;
        public DataSet validationSet = null;
        public double? trueThreshold = 0.5;
        public bool MEE = false;





    }
}
