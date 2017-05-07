using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AA1_MLP.Activations
{
    [Serializable]
     public class ActivationIdentity:IActivation
    {
        public MathNet.Numerics.LinearAlgebra.Vector<double> CalculateActivation(MathNet.Numerics.LinearAlgebra.Vector<double> x)
        {
            return x;
        }

        public MathNet.Numerics.LinearAlgebra.Vector<double> CalculateDerivative(MathNet.Numerics.LinearAlgebra.Vector<double> x)
        {
            throw new NotImplementedException();
        }
    }
}
