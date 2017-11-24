using AA1_MLP.Entities;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AA1_MLP.Utilities
{
    /// <summary>
    /// THis calss provides a funtion for evaluating the trained network accuracy on the training set
    /// </summary>
  public  class Tools
    {
      /// <summary>
      /// Given a traied network, a dataset and the threshold to consider a network output as true, this function returns the accuracy of the network on the given training set
      /// </summary>
      /// <param name="network">a MLP archietecture</param>
      /// <param name="Dataset">a set to test the netowrk accuracy on</param>
      /// <param name="trueThreshold">the threshold that larger than or equal to it, we consider the network output is 1 </param>
      /// <returns></returns>
      public static double ComputeAccuracy(Network network, DataSet Dataset, double? trueThreshold=0.5)
      {
          double trainingAccuracy = 0;
          for (int i = 0; i < Dataset.Inputs.RowCount; i++)
          {
              var o = network.ForwardPropagation(Dataset.Inputs.Row(i));
              trainingAccuracy += ((o[0] >= trueThreshold ? 1 : 0) == Dataset.Labels.Row(i)[0]) ? 1 : 0;

          }
          trainingAccuracy /= Dataset.Inputs.RowCount;
          return trainingAccuracy;
      }


    }
}
