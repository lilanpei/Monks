﻿using AA1_MLP.Activations;
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

            DataSet ds = DataManager.DataManager.LoadMonksData(Properties.Settings.Default.TrainingSetLocation, 17);

            for (int i = 0; i < ds.Inputs.RowCount; i++)
            {
                var result = n.ForwardPropagation(ds.Inputs.Row(i));
            }

            BackPropagation br = new BackPropagation();
            br.Train(n, ds, 10, true, 3, 0.2f);
        }
    }
}
