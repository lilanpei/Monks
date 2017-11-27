using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AA1_MLP.Activations
{
    /// <summary>
    /// Any new activation function should implement this interface to be usable by the library
    /// </summary>
    public interface IActivation
    {/// <summary>
    /// Given a vector x, this function computes and returns the value of the activation applied on x
    /// </summary>
    /// <param name="x"> the input to the activation function</param>
    /// <returns>a vector of the result of applying the activation function of x</returns>
        Vector<double> CalculateActivation(Vector<double> x);
        /// <summary>
        /// Given a vector x, this function computes and returns the derivative of the activation applied on x
        /// </summary>
        /// <param name="x"> in put vecot x</param>
        /// <returns> the result of the application of the derivative applied on x</returns>
        Vector<double>  CalculateDerivative(Vector<double> x);

    }
}
