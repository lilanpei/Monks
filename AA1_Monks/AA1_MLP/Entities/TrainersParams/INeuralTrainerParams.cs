using AA1_MLP.Enums;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace AA1_MLP.Entities.TrainersParams
{
    public class INeuralTrainerParams:TrainerParams
    {


        public Network network;
        public double learningRate;
        public int numberOfEpochs;
        public int? batchSize = null;
        public double regularizationRate = 0;
        public Regularizations regularization = Regularizations.None;
        public int NumberOfHiddenUnits = 0;




    }
}
