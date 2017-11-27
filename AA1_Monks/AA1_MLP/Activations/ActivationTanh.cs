using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AA1_MLP.Activations
{
    /// <summary>
    /// The hyperbolic tangent activation function
    /// </summary>
    [Serializable]
    public class ActivationTanh : IActivation
    {
        /// <summary>
        /// given a vector x, returns tanh(x)
        /// </summary>
        /// <param name="x">input vector to apply the tanh on its elemts</param>
        /// <returns>result from applying the tanh</returns>
        public Vector<double> CalculateActivation(Vector<double> x)
        {
            return x.PointwiseTanh();
        }
        /// <summary>
        /// Given an inout vector x, this function returns the result from the derivative of tanh,  1-tanh^2(x)
        /// </summary>
        /// <param name="x">input vector</param>
        /// <returns>the result from applying the derivative on the input vector</returns>
        public Vector<double> CalculateDerivative(Vector<double> x)
        {
            x = CalculateActivation(x);
            return 1 - x.PointwiseMultiply(x);
            //return 1 - x * x;
        }
    }
}
