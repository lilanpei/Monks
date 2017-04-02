using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AA1_MLP.Activations
{
    interface IActivation
    {
        Vector<double> CalculateActivation(Vector<double> x);
        double CalculateDerivative(Vector<double> x);

    }
}
