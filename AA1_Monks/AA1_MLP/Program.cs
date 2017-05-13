using AA1_MLP.Activations;
using AA1_MLP.Entities;
using AA1_MLP.Entities.Trainers;
using MathNet.Numerics.LinearAlgebra;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.Serialization.Formatters.Binary;
namespace AA1_MLP
{
    class Program
    {
        static void Main(string[] args)
        {
            Network n = new Network(new List<Layer>() {

            new Layer(new ActivationIdentity(),true,17),
            new Layer(new ActivationSigmoid(),true,4),
            //new Layer(new ActivationTanh(),true,2*17),

            new Layer(new ActivationSigmoid(),false,1),
            }, false);




            DataSet ds = DataManager.DataManager.LoadMonksData(Properties.Settings.Default.TrainingSetLocation, 17);
            /*   DataSet ds = new DataSet(
                   CreateMatrix.Dense(4, 2, new double[] { 1, 0, 1, 0, 1, 1, 0, 0 }),
                   CreateMatrix.Dense(4, 1, new double[] { 0, 1, 1, 0 }));*/
            /*
                        DataSet ds = new DataSet(
                            CreateMatrix.Dense(1, 2, new double[] { 0.05, 0.1 }),
                            CreateMatrix.Dense(1, 2, new double[] { 0.01, 0.99 })

                            );*/
            for (int i = 0; i < ds.Inputs.RowCount; i++)
            {
                System.Console.WriteLine("Input");
                System.Console.WriteLine(ds.Inputs.Row(i));
                var result = n.ForwardPropagation(ds.Inputs.Row(i));
                System.Console.WriteLine("Final output");
                System.Console.WriteLine(result);
            }
            string path2SaveModel = @"nw.AA1";
            // n = Utilities.ModelManager.LoadNetwork(path2SaveModel);

            BackPropagation br = new BackPropagation();
            var learningCurve = br.Train(n, ds, learningRate: 0.1, numberOfEpochs: 500,shuffle:false, debug: n.Debug, momentum: 0.5,resilient:false,resilientUpdateAccelerationRate:1.2,resilientUpdateSlowDownRate:0.5,regularization:Enums.Regularizations.L2,regularizationRate:0.01);

            File.WriteAllText("learningcurve.txt", string.Join("\n", learningCurve.Select(s => (s.Length == 2) ? (s[0] + "," + s[1]) : s[0] + "")));



            Utilities.ModelManager.SaveNetowrk(n, path2SaveModel);


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
           Utilities.ModelManager.Tester(ds, n);


        }


       
    }
}
