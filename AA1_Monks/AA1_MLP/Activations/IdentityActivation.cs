using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AA1_MLP.Activations
{
    /// <summary>
    /// -- marked for deletion --
    /// This special activation function is for the input layer neurons, it just returns the value of whatever vecor passed to it
    /// </summary>
    [Serializable]
    public class ActivationIdentity : IActivation
    {
        public MathNet.Numerics.LinearAlgebra.Vector<double> CalculateActivation(MathNet.Numerics.LinearAlgebra.Vector<double> x)
        {
            return x;
        }

        public MathNet.Numerics.LinearAlgebra.Vector<double> CalculateDerivative(MathNet.Numerics.LinearAlgebra.Vector<double> x)
        {
            return CreateVector.Dense<double>(x.Count, 1.0);
        }
    }
}
