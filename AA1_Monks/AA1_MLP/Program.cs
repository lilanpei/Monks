using AA1_MLP.Activations;
using AA1_MLP.Entities;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AA1_MLP
{
    class Program
    {
        static void Main(string[] args)
        {
            Network n = new Network(new List<Layer>() { 
            
            new Layer(){Activation=new ActivationIdentity(),Bias=true,NumberOfNeurons=17},
            new Layer(){Activation=new ActivationTanh(),Bias=true,NumberOfNeurons=17*2},
            new Layer(){Activation=new ActivationSigmoid(),Bias=true,NumberOfNeurons=1},
            
            });

            DataSet ds = DataManager.DataManager.LoadMonksData(Properties.Settings.Default.TrainingSetLocation);

            for (int i = 0; i < ds.Input.RowCount; i++)
            {
               

            var result =     n.ForwardPropagation(ds.Input.Row(i));
            
            
            }


        }
    }
}
