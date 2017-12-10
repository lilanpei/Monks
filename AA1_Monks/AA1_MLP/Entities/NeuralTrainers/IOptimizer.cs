using AA1_MLP.Entities.TrainersParams;
using AA1_MLP.Enums;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AA1_MLP.Entities.Trainers
{
    public abstract class IOptimizer
    {
        /// <summary>
        /// Given a network architecture, and the desired Dataset parameters and dataset for training and testing
        /// this function runs a forward propagation iteration,calculates the errors and backpropagates it to update the network weights
        /// for the accuracy computations, we assume a single output classification problem, otherwise we will need to implement cross entropy instead of MSE (perhaps later :) )
        /// </summary>
        /// <param name="network">a network architecture</param>
        /// <param name="Dataset"> the training dataset</param>
        /// <param name="learningRate"> the initial learning rate</param>
        /// <param name="numberOfEpochs">number of Dataset epochs</param>
        /// <param name="shuffle">set to true, will shuffle the data</param>
        /// <param name="batchSize">the Dataset data batch size</param>
        /// <param name="debug">set to true, will print verbose messages to the screen</param>
        /// <param name="regularizationRate">the L2 regularization rate used</param>
        /// <param name="regularization">Regularization method used, only L2 is implemented for now</param>
        /// <param name="momentum">Momentum rate</param>
        /// <param name="resilient">set to true, will enable the resilient property where the learning rate is multiplied by resilientUpdateAccelerationRate in case previous update was same sign as current update and resilientUpdateSlowDownRate otherwise</param>
        /// <param name="resilientUpdateAccelerationRate"> if resilient is set to true, the learning rate will be multiplied by this value in case the sign of the previous weights updates was the same as the current new one</param>
        /// <param name="resilientUpdateSlowDownRate">if resilient is set to true, the learning rate will be multiplied by this value in case the sign of the previous weights updates was NOT the same as the current new one</param>
        /// <param name="validationSet"> the validation dataset</param>
        /// <param name="trueThreshold"> between 0 to 1, if present accuracy of the Dataset and validation data will be computed at each epoch and reported in the returned learning curve list of doubles </param>
        /// <param name="MEE">set to true, will report the Mean Euclidean Error instead of Mean Square Error</param>
        /// <param name="reduceLearningRate">set to true, will enable reducing the learning rate during training</param>
        /// <param name="learningRateReduction">will be multiplied by the learning rate during training</param>
        /// <param name="learningRateReductionAfterEpochs">After how many epoch shall the learningRateReduction be multiplied by the learning rate</param>
        /// <param name="numberOfReductions">number of time a reduction shall happen</param>
        /// <returns> a list of double arrays each element is a 4 elements double array "a"  a[0] = iteration Dataset loss(MSE), a[1] = validation error(MSE), a[2] =Dataset set accuracy, a[3] = validation set accuracy  </returns>

        //Network network, DataSet trainingSet, double learningRate, int numberOfEpochs, bool shuffle = false, int? batchSize = null, bool debug = false, double regularizationRate = 0, Regularizations regularization = Regularizations.None, double momentum = 0, bool resilient = false, double resilientUpdateAccelerationRate = 1, double resilientUpdateSlowDownRate = 1, DataSet validationSet = null, double? trueThreshold = 0.5, bool MEE = false, bool reduceLearningRate = false, double learningRateReduction = 0.5, int learningRateReductionAfterEpochs = 1000, int numberOfReductions = 2, bool nestrov = false
        public abstract List<double[]> Train(ITrainerParams trainParams);




    }
}
