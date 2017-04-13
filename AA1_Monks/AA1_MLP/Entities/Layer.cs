using AA1_MLP.Activations;
using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AA1_MLP.Entities
{
    class Layer
    {
        public int NumberOfNeurons { get; set; }
        public IActivation Activation { get; set; }
        public bool Bias { get; set; }
        public Vector<double> LayerActivations { get; set; }

        public Vector<double> Delta { get; set; }//layer local error

        public Vector<double> ForwardPropagation(Vector<double> inputOfPrevLayer, Matrix<double> weights)
        {
            if (Bias)
            {
                var d = inputOfPrevLayer.ToList<double>();
                d.Insert(0, 1);
                inputOfPrevLayer = CreateVector.Dense(d.ToArray());
            }

            return LayerActivations = Activation.CalculateActivation(inputOfPrevLayer * weights);


        }
    }
}
