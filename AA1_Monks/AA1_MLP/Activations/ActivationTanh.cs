using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AA1_MLP.Activations
{
    class ActivationTanh : IActivation
    {

        public Vector<double> CalculateActivation(Vector<double> x)
        {
            return x.PointwiseTanh();
        }

        public double CalculateDerivative(Vector<double> x)
        {
            x = CalculateActivation(x);
            return 1-x.DotProduct(x);
            //return 1 - x * x;
        }
    }
}
