using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AA1_MLP.Entities
{
    [Serializable]
   public abstract class IModel
    {

      public abstract  Vector<double> Predict( Vector<double> input);
    }
}
