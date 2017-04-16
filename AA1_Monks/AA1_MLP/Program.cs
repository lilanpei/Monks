using AA1_MLP.Activations;
using AA1_MLP.Entities;
using AA1_MLP.Entities.Trainers;
using MathNet.Numerics.LinearAlgebra;
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

            new Layer(){Activation=new ActivationIdentity(),Bias=false,NumberOfNeurons=2},
            new Layer(){Activation=new ActivationTanh(),Bias=false,NumberOfNeurons=3},
            new Layer(){Activation=new ActivationSigmoid(),Bias=false,NumberOfNeurons=1},
            });




            //DataSet ds = DataManager.DataManager.LoadMonksData(Properties.Settings.Default.TrainingSetLocation, 17);
            DataSet ds = new DataSet(
                CreateMatrix.Dense(4, 2, new double[] { 1, 1, 0, 1, 1, 0, 0, 0 }),

                CreateMatrix.Dense(4, 1, new double[] { 1, 0, 1, 1 }));
            for (int i = 0; i < ds.Inputs.RowCount; i++)
            {
                var result = n.ForwardPropagation(ds.Inputs.Row(i));
            }

            BackPropagation br = new BackPropagation();
            br.Train(n, ds, 0.1,100);
        }
    }
}
