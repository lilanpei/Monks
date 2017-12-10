using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
namespace AA1_MLP.Entities.TrainersParams
{
    public class GradientDescentParams : INeuralTrainerParams
    {
        public double momentum = 0;
        public bool resilient = false;
        public double resilientUpdateAccelerationRate = 1;
        public double resilientUpdateSlowDownRate = 1;
        public bool reduceLearningRate = false;
        public double learningRateReduction = 0.5;
        public int learningRateReductionAfterEpochs = 1000;
        public int numberOfReductions = 2;
        public bool nestrov = false;
    }
}
