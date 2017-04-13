using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AA1_MLP.Entities
{
    class Network
    {
        public List<Layer> Layers { get; set; }
        public List<Matrix<double>> Weights { get; set; }

        public Network(List<Layer> _Layers)
        {
            Weights = new List<Matrix<double>>();
            Layers = _Layers;

            for (int i = 0; i < Layers.Count - 1; i++)
            {
                int addBias = 0;
                if (Layers[i + 1].Bias)
                {
                    addBias = 1;
                }
                var d = 1 / Math.Sqrt(Layers[i].NumberOfNeurons + 1);
                Weights.Add(CreateMatrix.Random<double>(Layers[i].NumberOfNeurons + addBias, Layers[i + 1].NumberOfNeurons, new ContinuousUniform(-1 * d, d)));

            }

        }

        public Vector<double> ForwardPropagation(Vector<double> input)
        {

            for (int i = 1; i < Layers.Count; i++)
            {
                input = Layers[i].ForwardPropagation(input, Weights[i - 1]);

            }
            return input;
        }
    }
}
