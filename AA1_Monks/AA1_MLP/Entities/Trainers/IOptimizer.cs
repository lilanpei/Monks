using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AA1_MLP.Entities.Trainers
{
    abstract class IOptimizer
    {


        public string Name { get; set; }
        public delegate void Historian(string historyLogFileLocation);//logs history :D
        public delegate void CheckPointer(string checkpointLocation);//saves a checkpoint of the file

        /// <summary>
        /// Given a network architecture, 
        /// this function runs a forward propagation iteration,calculates the errors and backpropagate it to update the network weights
        /// </summary>
        /// <param name="network">Is a network architecture</param>
        public abstract void Train(Network network, DataSet wholeData, double learningRate, int numberOfEpochs, bool shuffle = false, int? batchSize = null, float? validationSplit = null, Historian historian = null, CheckPointer checkPointer = null, bool debug = false);


    }
}
