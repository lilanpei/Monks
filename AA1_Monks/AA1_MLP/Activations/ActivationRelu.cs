using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AA1_MLP.Activations
{
    [Serializable]
    class ActivationLeakyRelu : IActivation
    {
        public MathNet.Numerics.LinearAlgebra.Vector<double> CalculateActivation(MathNet.Numerics.LinearAlgebra.Vector<double> x)
        {
            return x.Map(s => s > 0 ? s : 0.01 * s);
        }

        public MathNet.Numerics.LinearAlgebra.Vector<double> CalculateDerivative(MathNet.Numerics.LinearAlgebra.Vector<double> x)
        {
            return x.Map(s => s > 0 ? 1 : 0.01 );

        }
    }
}
