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

                    var d = 1 / Math.Sqrt(Layers[i].NumberOfNeurons + 1);
                    Weights.Add(CreateMatrix.Random<double>(Layers[i].NumberOfNeurons + (Layers[i].Bias?1:0), Layers[i + 1].NumberOfNeurons, new MathNet.Numerics.Distributions.Normal(0, 1 / d)));
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.WriteLine("Initial Weights layer:{0} {1}", i, Weights[i ]);
                    Console.ResetColor();
                }
            }

        }

        public Vector<double> ForwardPropagation(Vector<double> input)
        {
            Layers[0].LayerActivationsSumInputs = CreateVector.DenseOfVector(input);
            
            Layers[0].LayerActivations = Layers[0].LayerActivationsSumInputs;

            if (Layers[0].Bias)
            {
                var d = Layers[0].LayerActivationsSumInputs.ToList<double>();
                d.Insert(0, 1);
                Layers[0].LayerActivationsSumInputs = CreateVector.Dense(d.ToArray());
                Layers[0].LayerActivations = Layers[0].LayerActivationsSumInputs;

            }




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

                if (i>0&& Math.Abs( Layers[i - 1].NumberOfNeurons- input.Count)<1 && Layers[i-1].Bias)
                {
                    var d = input.ToList<double>();
                    d.Insert(0, 1);
                    input = CreateVector.Dense(d.ToArray());
                }
                input = Layers[i].ForwardPropagation(input, Weights[i - 1] ,Debug);

                Console.WriteLine("Output Of Layer:{0}",i);
                Console.WriteLine(input);


            }
            return input;
        }
    }
}
