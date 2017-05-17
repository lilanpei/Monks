using AA1_MLP.Entities;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AA1_MLP.Utilities
{
  public  class Tools
    {
      public static double ComputeAccuracy(Network network, DataSet trainingSet, double? trueThreshold)
      {
          double trainingAccuracy = 0;
          for (int i = 0; i < trainingSet.Inputs.RowCount; i++)
          {
              var o = network.ForwardPropagation(trainingSet.Inputs.Row(i));
              trainingAccuracy += ((o[0] >= trueThreshold ? 1 : 0) == trainingSet.Labels.Row(i)[0]) ? 1 : 0;

          }
          trainingAccuracy /= trainingSet.Inputs.RowCount;
          return trainingAccuracy;
      }


    }
}
