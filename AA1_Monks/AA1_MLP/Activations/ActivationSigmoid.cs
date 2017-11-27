using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AA1_MLP.Activations
{
    /// <summary>
    /// The sigmoid activation function
    /// </summary>
    [Serializable]
    public class ActivationSigmoid : IActivation
    {
        /// <summary>
        /// Given an input vector x, this function computes and returns the result from applying sigmoid(x) =  1 / (1 + Exp(-x))
        /// </summary>
        /// <param name="x">the input vector</param>
        /// <returns>the result of applying the sigmoid on the input vector</returns>
        public Vector<double> CalculateActivation(Vector<double> x)
        {
            return 1 / (x.Multiply(-1).PointwiseExp()+1);
        }

        /// <summary>
        /// Given an input vector x, this function computes and returns the derivative of the sigmoid applied to x, = sigmoid(x) * (1 - sigmoid(x))
        /// </summary>
        /// <param name="x">the input vector</param>
        /// <returns>the result of applying the sigmoid on the input vector</returns>
        public Vector<double> CalculateDerivative(Vector<double> x)
        {
            x = CalculateActivation(x);
            return x.PointwiseMultiply(1-x);
        }
    }
}
