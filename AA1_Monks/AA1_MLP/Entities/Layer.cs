using AA1_MLP.Activations;
using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AA1_MLP.Entities
{
    [Serializable]
   public class Layer
    {
        public int NumberOfNeurons { get; set; }
        public IActivation Activation { get; set; }
        public bool Bias { get; set; }
        public Vector<double> LayerActivationsSumInputs { get; set; }
        public Vector<double> LayerActivations { get; set; }
        public Vector<double> Delta { get; set; }//layer local error



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

        public Vector<double> ForwardPropagation(Vector<double> inputOfPrevLayer, Matrix<double> weights, bool debug = false)
        {

            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine("Weights layer forward prop:{0} ", weights);
            Console.ResetColor();


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
