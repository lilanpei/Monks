using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AA1_MLP.Activations
{/// <summary>
/// our enum of available activation functions, to add a new one, add its name here and make it implement the IActivation interface 
/// </summary>
    enum Activations
    {
        Sigmoid,
        Tanh,
        LeakyRelu
    }
}
