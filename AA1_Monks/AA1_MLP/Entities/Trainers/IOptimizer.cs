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
        /// Given a network architecture, and the desired training parameters and dataset for training and testing
        /// this function runs a forward propagation iteration,calculates the errors and backpropagate it to update the network weights
        /// </summary>
        /// <param name="network">a network architecture</param>
        /// <param name="wholeData"> the training dataset, can be split into training and validation with the validationsplit</param>
        /// <param name="learningRate"> the initial learning rate</param>
        /// <param name="numberOfEpochs">number of training epochs</param>
        /// <param name="shuffle">set to true, will shuffle the data</param>
        /// <param name="batchSize">the training data batch size</param>
        /// <param name="validationSplit">fraction of training data to be used for validation</param>
        /// <param name="debug">set to true, will print verbose messages to the screen</param>
        /// <param name="regularizationRate">the L2 regularization rate used</param>
        /// <param name="regularization">Regularization method used, only L@ is implemented for now</param>
        /// <param name="momentum">Momentum rate</param>
        /// <param name="resilient">set to true, will enable the resilient property where the learning rate is multiplied by resilientUpdateAccelerationRate in case previous update was same sign as current update and resilientUpdateSlowDownRate otherwise</param>
        /// <param name="resilientUpdateAccelerationRate"> if resilient is set to true, the learning rate will be multiplied by this value in case the sign of the previous weights updates was the same as the current new one</param>
        /// <param name="resilientUpdateSlowDownRate">if resilient is set to true, the learning rate will be multiplied by this value in case the sign of the previous weights updates was NOT the same as the current new one</param>
        /// <param name="testData"> the testing dataset</param>
        /// <returns></returns>
        public abstract List<double[]> Train(Network network, DataSet wholeData, double learningRate, int numberOfEpochs, bool shuffle = false, int? batchSize = null, float? validationSplit = null, bool debug = false, double regularizationRate = 0, Regularizations regularization = Regularizations.None, double momentum = 0, bool resilient = false, double resilientUpdateAccelerationRate = 1, double resilientUpdateSlowDownRate = 1, DataSet testData = null);


    }
}
