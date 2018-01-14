using System;
using System.Configuration;
using AA1_Monks.ModelManager;
using System.Linq;
using Encog.Neural.Networks.Training.Propagation.SGD.Update;
using AA1_MLP.Entities;
using Encog.Neural.Networks;
using Encog.Neural.Networks.Layers;
using Encog.Engine.Network.Activation;
using Encog.Neural.Data.Basic;
using System.IO;
using System.Runtime.Serialization.Formatters.Binary;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;

namespace AA1_Monks
{
    class Program
    {
        static void Main(string[] args)
        {


            //Properties.Settings.Default.trainingDataLocation;
            //  string testSetLocation = Properties.Settings.Default.testDataLocation;



            AA1_MLP.DataManagers.CupDataManager dm = new AA1_MLP.DataManagers.CupDataManager();
            var trainDS = LoadData("D:\\dropbox\\Dropbox\\Master Course\\SEM-3\\ML\\CM_CUP_Datasets\\Standardized60percenttrain.txt");
            var testDS = dm.LoadData("D:\\dropbox\\Dropbox\\Master Course\\SEM-3\\ML\\CM_CUP_Datasets\\Standardized60percenttest.txt", 10, 2);


            //  StandardizeData(trainDS);
            // StandardizeData(testDS);





            BasicNetwork network = new BasicNetwork();
            network.AddLayer(new BasicLayer(10));


            network.AddLayer(new BasicLayer(new ActivationTANH(), true, 100));
            network.AddLayer(new BasicLayer(new ActivationLinear(), true, 2));


            network.Structure.FinalizeStructure();
            network.Reset();


            var train = new Encog.Neural.Networks.Training.Propagation.SGD.StochasticGradientDescent(network, trainDS);
            var update = new Encog.Neural.Networks.Training.Propagation.SGD.Update.NesterovUpdate();
            update.Init(train);
            train.LearningRate = 0.001;
            train.Momentum = 0.5;
            train.L2 = 0.001;
            train.BatchSize = 10;
            var watch = System.Diagnostics.Stopwatch.StartNew();

            int epoch = 0;
            using (System.IO.StreamWriter file =
                new System.IO.StreamWriter(@"C:\Users\ahmad\Documents\monks\Monks\AA1_Monks\AA1_CUP\bin\Release\encog.txt"))
            {
                do
                {
                    train.Iteration();
                    


                    double valMEE = 0.0, valMSE = 0.0;

                    TesterCUPRegression(testDS, network, out valMEE, out valMSE);

                    Console.WriteLine("Epoch #" + epoch + " Error:" + train.Error+" valMSE:"+valMSE);


                    file.WriteLine("{0},{1}",train.Error,valMSE);
                    epoch++;
                } while ((epoch < 5000) && (train.Error > 0));

            }

            watch.Stop();
            var elapsedMs = watch.ElapsedMilliseconds;
            Console.WriteLine("elapsed Time:{0} ms", elapsedMs);
            var serializer = new BinaryFormatter();
            using (var s = new FileStream(@"C:\Users\ahmad\Documents\monks\Monks\AA1_Monks\AA1_CUP\bin\Release\encog.n", FileMode.Create))
            {
                serializer.Serialize(s, network);
            }

            double MEE = 0.0, MSE = 0.0;

            TesterCUPRegression(testDS, network, out MEE, out MSE);
            Console.WriteLine("MEE{0},MSE{1}", MEE, MSE);

        }


        public static List<double[]> TesterCUPRegression(DataSet testSet, BasicNetwork n, out double MEE, out double MSE)
        {
            List<double[]> predictionsVSActuals = new List<double[]>();
            MEE = 0;
            MSE = 0;
            for (int i = 0; i < testSet.Inputs.RowCount; i++)
            {
                double[] opt = new double[2];
                n.Compute(testSet.Inputs.Row(i).ToArray(), opt);

                predictionsVSActuals.Add(new double[] { opt[0], opt[1], testSet.Labels.Row(i)[0], testSet.Labels.Row(i)[1] });

                var o = CreateVector.Dense<double>(opt);
                var loss = ((testSet.Labels.Row(i) - o).PointwiseMultiply(testSet.Labels.Row(i) - o)).Sum();
                MEE += Math.Sqrt(loss);
                MSE += loss;
            }
            MSE /= testSet.Labels.RowCount;
            MEE /= testSet.Labels.RowCount;
            return predictionsVSActuals;


        }

        public static BasicNeuralDataSet LoadData(string datasetLocation)
        {
            string[] lines = File.ReadAllLines(datasetLocation).Where(s => !string.IsNullOrEmpty(s) && !s.StartsWith("#")).ToArray();
            double[][] input = new double[lines.Length][];
            double[][] output = new double[lines.Length][];
            for (int i = 0; i < lines.Length; i++)
            {
                var line = lines[i].Trim().Split(',');
                input[i] = new double[10];

                for (int x = 0; x < 10; x++)
                {
                    input[i][x] = double.Parse(line[x]);

                }

                output[i] = new double[] { double.Parse((line[10])), double.Parse(line[11]) };
            }


            BasicNeuralDataSet trainingSet = new BasicNeuralDataSet(input, output);
            return trainingSet;
        }

    }
}
