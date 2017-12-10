using AA1_MLP.Activations;
using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AA1_MLP.Entities
{
    /// <summary>
    /// Rrepresents a fully connected neural network layer
    /// </summary>
    [Serializable]
   public class Layer
    {
        public int NumberOfNeurons { get; set; }
        public IActivation Activation { get; set; }
        public bool Bias { get; set; }
        public Vector<double> LayerActivationsSumInputs { get; set; }
        public Vector<double> LayerActivations { get; set; }
        public Vector<double> Delta { get; set; }//layer local error


        /// <summary>
        /// The default constructor, specifying the activation function of the neurons in the layer, a boolean to add a bias or not and the number of neurons [not including the bias]
        /// </summary>
        /// <param name="_activation">the activaion function for all of the neurons in the layer</param>
        /// <param name="_bias">set to true, will add a bias unit</param>
        /// <param name="_numberOfNeurons">the number of the neurons in the layer</param>
        public Layer(IActivation _activation, bool _bias, int _numberOfNeurons)
        {
            Activation = _activation;
            Bias = _bias;
            NumberOfNeurons = _numberOfNeurons;
          /*  int addBias = 0;
            if (Bias)
            {
                addBias = 1;
            }
            NumberOfNeurons += addBias;*/
        }
        /// <summary>
        /// Multiplies the input from the previous layer by the weights between this layer and the previous layers and fires the activation function of this layer's neuron on the result.
        /// </summary>
        /// <param name="inputOfPrevLayer">the output of the prebious layer, before multiplying the weights between the two layers with it</param>
        /// <param name="weights">the weights between the current layer and the previous one</param>
        /// <param name="debug">set to true, will print debugging messages</param>
        /// <returns>returns the result of applying the neurons activation function on the previous layer output * weights matrix</returns>
        public Vector<double> ForwardPropagation(Vector<double> inputOfPrevLayer, Matrix<double> weights, bool debug = false)
        {
            if (debug)
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine("Weights layer forward prop:{0} ", weights);
                Console.ResetColor();
            }


            /* if (Bias)
             {
                 var d = inputOfPrevLayer.ToList<double>();
                 d.Insert(0, 1);
                 inputOfPrevLayer = CreateVector.Dense(d.ToArray());
             }*/
        
            var activationInput = inputOfPrevLayer * weights;
            if (debug)
            {
                Console.WriteLine("Activation Input(the sum):");
                Console.WriteLine(activationInput);
            }
            LayerActivationsSumInputs = activationInput;
             LayerActivations = Activation.CalculateActivation(activationInput);
           /* if (Bias)
            {
                var d = LayerActivations.ToList<double>();
                d.Add( 1);
                LayerActivations = CreateVector.Dense(d.ToArray());
            }*/
            return LayerActivations;


        }
    }
}
