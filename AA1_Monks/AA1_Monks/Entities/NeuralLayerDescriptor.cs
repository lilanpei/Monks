using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Encog.Engine.Network.Activation;

namespace AA1_Monks.Entities
{
    /// <summary>
    /// 
    /// </summary>
    class NeuralLayerDescriptor
    {
        public IActivationFunction Activation { get; set; }

        public bool Bias { get; set; }

        public int  NumberOfNeurons { get; set; }

    }
}
