using AA1_MLP.Activations;
using AA1_MLP.Entities;
using AA1_MLP.Entities.Trainers;
using MathNet.Numerics.LinearAlgebra;
using System.Collections.Generic;

namespace AA1_MLP
{
    class Program
    {
        static void Main(string[] args)
        {
            Network n = new Network(new List<Layer>() {

            new Layer(new ActivationIdentity(),true,17),
            new Layer(new ActivationSigmoid(),true,17*2),
            new Layer(new ActivationSigmoid(),false,1),
            }, false);




            DataSet ds = DataManager.DataManager.LoadMonksData(Properties.Settings.Default.TrainingSetLocation, 17);
            /*DataSet ds = new DataSet(
                CreateMatrix.Dense(4, 2, new double[] { 1, 0, 1, 0, 1, 1, 0, 0 }),
                CreateMatrix.Dense(4, 1, new double[] { 0, 1, 1, 0 }));*/
            for (int i = 0; i < ds.Inputs.RowCount; i++)
            {
                System.Console.WriteLine("Input");
                System.Console.WriteLine(ds.Inputs.Row(i));


                var result = n.ForwardPropagation(ds.Inputs.Row(i));
                System.Console.WriteLine("Final output");
                System.Console.WriteLine(result);
            }

            BackPropagation br = new BackPropagation();
            br.Train(n, ds, learningRate: 0.1, numberOfEpochs: 10000, debug: n.Debug);


            System.Console.WriteLine("~~~~~~~~~~Printing Results:~~~~~~~~~~");

            for (int i = 0; i < ds.Inputs.RowCount; i++)
            {
                System.Console.WriteLine("Input");
                System.Console.WriteLine(ds.Inputs.Row(i));
                System.Console.WriteLine("Target");
                System.Console.WriteLine(ds.Labels.Row(i));
                checked
                {
                    var result = n.ForwardPropagation(ds.Inputs.Row(i));

                    System.Console.WriteLine("Final output");
                    System.Console.WriteLine(result);
                }
                System.Console.WriteLine("Target");
                System.Console.WriteLine(ds.Labels.Row(i));
            }


        }
    }
}
