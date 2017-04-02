using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AA1_MLP.Activations
{
    class ActivationSigmoid : IActivation
    {
        public Vector<double> CalculateActivation(Vector<double> x)
        {
            return 1 / (x.Multiply(-1).PointwiseExp()+1);
            //return 1 / (1 + Math.Exp(-x));
        }

        public double CalculateDerivative(Vector<double> x)
        {
            x = CalculateActivation(x);

            return x.DotProduct(1-x);
            //  return x * (1 - x);
        }
    }
}
