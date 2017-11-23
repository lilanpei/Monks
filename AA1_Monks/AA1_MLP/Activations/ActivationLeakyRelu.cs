using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AA1_MLP.Activations
{
    /// <summary>
    /// the leaky relu of x = max(x,0.01x)
    /// </summary>
    [Serializable]
   public class ActivationLeakyRelu : IActivation
    {
        /// <summary>
        /// Given an input vector x, this function computes and returns max(x,0.01x) elementwise
        /// </summary>
        /// <param name="x">the inout vector x</param>
        /// <returns> max(x,0.01x) </returns>
        public MathNet.Numerics.LinearAlgebra.Vector<double> CalculateActivation(MathNet.Numerics.LinearAlgebra.Vector<double> x)
        {
            return x.Map(s => s > 0 ? s : 0.01 * s);
        }
        /// <summary>
        /// Given a vector x, returns the derivative of x, elementwise
        /// </summary>
        /// <param name="x">the input vector x</param>
        /// <returns> elementwise application of the derivative of the relu function =  x > 0 ? 1 : 0.01</returns>
        public MathNet.Numerics.LinearAlgebra.Vector<double> CalculateDerivative(MathNet.Numerics.LinearAlgebra.Vector<double> x)
        {
            return x.Map(s => s > 0 ? 1 : 0.01 );

        }
    }
}
