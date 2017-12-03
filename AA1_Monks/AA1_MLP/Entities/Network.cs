using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AA1_MLP.Entities
{
    /// <summary>
    ///  A class representing a multilayer perceptron object, here one can compose feedforward neural networks with different number of layers , neurons and different activation functions
    /// </summary>
    [Serializable]
    public class Network
    {
        public List<Layer> Layers { get; set; }
        public List<Matrix<double>> Weights { get; set; }
        public bool Debug { get; set; }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="_Layers">a list of layers each specifying its number of neurons, bias and activation function for the neurons</param>
        /// <param name="debug">set to true, will print debug messages and will test a small network archietecture</param>
        /// <param name="weightsInitMethod">How the weights should be instantiated, Xavier by default or Uniform </param>
        public Network(List<Layer> _Layers,  bool debug = false, Enums.WeightsInitialization weightsInitMethod = Enums.WeightsInitialization.Xavier)
        {
            Debug = debug;
            Weights = new List<Matrix<double>>();
            Layers = _Layers;

            for (int i = 0; i < Layers.Count - (weightsInitMethod == Enums.WeightsInitialization.Uniform ? 2 : 1); i++)
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
                    var d =  2f / (Layers[i].NumberOfNeurons) ;
                    if (weightsInitMethod == Enums.WeightsInitialization.Uniform)
                    {
                        Weights.Add(d * CreateMatrix.Random<double>(Layers[i].NumberOfNeurons + (Layers[i].Bias ? 1 : 0), Layers[i + 1].NumberOfNeurons, new ContinuousUniform(-0.7, 0.7)));
                    }
                    else
                    {
                        Weights.Add(CreateMatrix.Random<double>(Layers[i].NumberOfNeurons + (Layers[i].Bias ? 1 : 0), Layers[i + 1].NumberOfNeurons, new Normal(0, d,new Random(1))));

                    }

                    if (Debug)
                    {

                        Console.ForegroundColor = ConsoleColor.Red;
                        Console.WriteLine("Initial Weights layer:{0} {1}", i, Weights[i]);
                        Console.ResetColor();
                    }
                }

            }
            //TODO: fix the problem with the idea that uniform should have a different weights scale for the last layer, if this is not true, then there is no need for the condition in the for loop!
            if (weightsInitMethod == Enums.WeightsInitialization.Uniform)
            {
                Weights.Add(CreateMatrix.Random<double>(Layers[Layers.Count - 2].NumberOfNeurons + (Layers[Layers.Count - 2].Bias ? 1 : 0), Layers[Layers.Count - 1].NumberOfNeurons, new ContinuousUniform(-0.7, 0.7)));

            }

        }
        /// <summary>
        /// Iteratively calls the forward ropagation method of each layer on the output of the previous layer
        /// </summary>
        /// <param name="input">The inputs to the neural network for the current problem</param>
        /// <returns></returns>
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
                if (Debug)
                {
                    Console.WriteLine("Output Of Layer:{0}", i);
                    Console.WriteLine(input);

                }

            }
            return input;
        }
    }
}
