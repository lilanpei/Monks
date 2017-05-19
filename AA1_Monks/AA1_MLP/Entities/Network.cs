using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AA1_MLP.Entities
{
    [Serializable]
    public class Network
    {
        public List<Layer> Layers { get; set; }
        public List<Matrix<double>> Weights { get; set; }
        public bool Debug { get; set; }
        public Network(List<Layer> _Layers, IContinuousDistribution distribution, bool timesFanIn = false, bool debug = false)
        {
            Debug = debug;
            Weights = new List<Matrix<double>>();
            Layers = _Layers;

            for (int i = 0; i < Layers.Count - 2; i++)
            {

                if (debug)
                {
                    if (i == 0)
                    {
                        var weightsMatrix = CreateMatrix.Dense(3, 2, new double[] { 0.15, 0.25, 0.35, 0.20, 0.30, 0.35 });
                        Weights.Add(weightsMatrix);
                    }
                    else
                    {
                        var weightsMatrix = CreateMatrix.Dense(3, 2, new double[] { 0.4, 0.5, 0.6, 0.45, 0.55, 0.6 });
                        Weights.Add(weightsMatrix);
                    }

                }
                else
                {

                    //var d = 1 / Math.Sqrt(Layers[i].NumberOfNeurons + 1);
                    //Weights.Add(CreateMatrix.Random<double>(Layers[i].NumberOfNeurons + (Layers[i].Bias ? 1 : 0), Layers[i + 1].NumberOfNeurons, new MathNet.Numerics.Distributions.Normal(0,1))/d);
                    // = new MathNet.Numerics.Distributions.ContinuousUniform(-0.7, 0.7);
                    var d = timesFanIn ? 2f / (Layers[i].NumberOfNeurons) : 1;
                    Weights.Add(d * CreateMatrix.Random<double>(Layers[i].NumberOfNeurons + (Layers[i].Bias ? 1 : 0), Layers[i + 1].NumberOfNeurons, distribution));



                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.WriteLine("Initial Weights layer:{0} {1}", i, Weights[i]);
                    Console.ResetColor();
                }

            }

            //last weight layer
            Weights.Add(CreateMatrix.Random<double>(Layers[Layers.Count - 2].NumberOfNeurons + (Layers[Layers.Count - 2].Bias ? 1 : 0), Layers[Layers.Count - 1].NumberOfNeurons, distribution));

        }

        public Vector<double> ForwardPropagation(Vector<double> input)
        {
            //Layers[0].LayerActivationsSumInputs = CreateVector.DenseOfVector(input);

            Layers[0].LayerActivations = CreateVector.DenseOfVector(input);// Layers[0].LayerActivationsSumInputs;

            /*   if (Layers[0].Bias)
               {
                   var d = Layers[0].LayerActivations.ToList<double>();
                   d.Add(1);
                   Layers[0].LayerActivations = CreateVector.Dense(d.ToArray());
                   //Layers[0].LayerActivations = Layers[0].LayerActivationsSumInputs;

               }*/




            for (int i = 1; i < Layers.Count; i++)
            {
                if (Debug)
                {
                    Console.WriteLine("Input to Layer:{0}", i);
                    Console.WriteLine(input);
                    Console.WriteLine("WeightsMatrix:");
                    Console.WriteLine(Weights[i - 1]);
                    Console.WriteLine("Layer ActivationsSigmoid:{0}", i);

                }

                if (Math.Abs(Layers[i - 1].NumberOfNeurons - input.Count) < 1 && Layers[i - 1].Bias)
                {

                    var d = input.ToList<double>();
                    d.Add(1);
                    input = CreateVector.Dense(d.ToArray());
                }
                input = Layers[i].ForwardPropagation(input, Weights[i - 1], Debug);

                Console.WriteLine("Output Of Layer:{0}", i);
                Console.WriteLine(input);


            }
            return input;
        }
    }
}
