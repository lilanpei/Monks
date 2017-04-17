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
        public bool Debug { get; set; }
        public Network(List<Layer> _Layers, bool debug = false)
        {
            Debug = debug;
            Weights = new List<Matrix<double>>();
            Layers = _Layers;

            for (int i = 0; i < Layers.Count - 1; i++)
            {
                int addBias = 0;
                if (Layers[i + 1].Bias)
                {
                    addBias = 1;
                }
                if (debug)
                {
                    if (i == 0)
                    {
                        var weightsMatrix = CreateMatrix.Dense(2, 3, new double[] { 0.8, 0.2, 0.4, 0.9, 0.3, 0.5 });
                        Weights.Add(weightsMatrix);
                    }
                    else
                    {
                        var weightsMatrix = CreateMatrix.Dense(3, 1, new double[] { 0.3, 0.5, 0.9 });
                        Weights.Add(weightsMatrix);
                    }

                }
                else
                {
                   // var d = 1 / Math.Sqrt(Layers[i].NumberOfNeurons + 1);
                    Weights.Add(CreateMatrix.Random<double>(Layers[i].NumberOfNeurons + addBias, Layers[i + 1].NumberOfNeurons, new ContinuousUniform(0, 1)));
                }
            }

        }

        public Vector<double> ForwardPropagation(Vector<double> input)
        {

            Layers[0].LayerActivationsSumInputs = input;
            Layers[0].LayerActivations = input;


            for (int i = 1; i < Layers.Count; i++)
            {
                if (Debug)
                {
                    Console.WriteLine("Input to Layer:{0}",i);
                    Console.WriteLine(input);
                    Console.WriteLine("WeightsMatrix:");
                    Console.WriteLine(Weights[i - 1]);
                    Console.WriteLine("Layer ActivationsSigmoid:{0}", i);

                }
                input = Layers[i].ForwardPropagation(input, Weights[i - 1], Debug);

                Console.WriteLine("Output Of Layer:{0}",i);
                Console.WriteLine(input);


            }
            return input;
        }
    }
}
