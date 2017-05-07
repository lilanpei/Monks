using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AA1_MLP.Activations
{
    [Serializable]
    public class ActivationTanh : IActivation
    {

        public Vector<double> CalculateActivation(Vector<double> x)
        {
            return x.PointwiseTanh();
        }

        public Vector<double> CalculateDerivative(Vector<double> x)
        {
            x = CalculateActivation(x);
            return 1 - x.PointwiseMultiply(x);
            //return 1 - x * x;
        }
    }
}
