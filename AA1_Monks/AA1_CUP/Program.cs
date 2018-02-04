using AA1_MLP.Activations;
using AA1_MLP.Entities;
using AA1_MLP.Entities.Trainers;
using AA1_MLP.Entities.TrainersParams;
using AA1_MLP.Enums;
using AA1_MLP.Utilities;
using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AA1_CUP
{
    /// <summary>
    /// Performing an automated grid search for hyperparameters for the model for the Cup problem
    /// </summary>
    public class Program
    {
        static void Main(string[] args)
        {
            //Loading and parsing cup dataset
            /* CupDataManager dm = new CupDataManager();
             DataSet wholeSet = dm.LoadData(Properties.Settings.Default.TrainingSetLocation, 10, 2, permute: true, seed: 1);
             List<double> momentums = new List<double> { 0, 0.5 };
             List<double> learningRates = new List<double> { 0.005, 0.01 };
             List<double> regularizationRates = new List<double> { 0, 0.001 };
             List<int> humberOfHiddenNeurons = new List<int> { 80 };
            //screening SGD+Momentum experiments
             GradientDescentParams passedParams = new GradientDescentParams();
             passedParams.nestrov = false;
             passedParams.resilient = false;
             passedParams.resilientUpdateAccelerationRate = 0.3;
             passedParams.resilientUpdateSlowDownRate = 0.1;
             new KFoldValidation().ScreenGD(wholeSet, 5, momentums, learningRates, regularizationRates, humberOfHiddenNeurons, passedParams,5000);*/
            //screening Adam
            //new KFoldValidation().ScreenAdam(wholeSet, 5, learningRates, regularizationRates, humberOfHiddenNeurons, 5000);


            //ReportHowCloseWeightsAcquiredFromDifferentSeedsAre();


             AA1_MLP.DataManagers.CupDataManager dm = new AA1_MLP.DataManagers.CupDataManager();
             DataSet trainDS = dm.LoadData(@"C:\Users\Ronin\Documents\monks\Monks\UsedFiles\TrainValSplits\60percenttrain.txt", 10, 2, standardize: true);
             DataSet testDS = dm.LoadData(@"C:\Users\Ronin\Documents\monks\Monks\UsedFiles\TrainValSplits\60percenttest.txt", 10, 2, standardize: true);

            /*  Console.WriteLine("Training Adam");
              AdamParams adampassedParams = new AdamParams();
              IOptimizer adamtrainer = new Adam();

              adampassedParams.numberOfEpochs = 100;
              adampassedParams.batchSize = 10;
              adampassedParams.trainingSet = trainDS;
              adampassedParams.validationSet = testDS;
              adampassedParams.learningRate = 0.001;
              adampassedParams.regularization = Regularizations.L2;
              adampassedParams.regularizationRate = 0.001;
              adampassedParams.NumberOfHiddenUnits = 100;
              adampassedParams.parallelize = true;
              LastTrain(testDS, adampassedParams, adamtrainer, "100epoadam_profiling_parlock", 1);*/

            Console.WriteLine("training SGD");
            GradientDescentParams passedParams = new GradientDescentParams();
            Gradientdescent trainer = new Gradientdescent();
            passedParams.numberOfEpochs = 100;
            passedParams.batchSize = 10;
            passedParams.trainingSet = trainDS;
            passedParams.validationSet = testDS;
            passedParams.learningRate = 0.001;
            passedParams.regularization = Regularizations.L2;
            passedParams.regularizationRate = 0.001;
            passedParams.nestrov = true;
            passedParams.resilient = false;
            passedParams.resilientUpdateAccelerationRate = 2;
            passedParams.resilientUpdateSlowDownRate = 0.5;
            passedParams.momentum = 0.5;
            passedParams.NumberOfHiddenUnits = 100;
            passedParams.parallelize = true;
            LastTrain(testDS, passedParams, trainer, "5kepochsprofiling_seq", 1);

            Console.WriteLine();


            /*
             List<int> seeds = new List<int>() { 1,15,40,4,73,2};

             foreach (var seed in seeds)
             {
                 Console.WriteLine("Seed:{0}",seed);

                 /*AdamParams passedParams = new AdamParams();
                 IOptimizer trainer = new Adam();
                 Console.WriteLine("training SGD");
                 GradientDescentParams passedParams = new GradientDescentParams();
                 Gradientdescent trainer = new Gradientdescent();
                 passedParams.numberOfEpochs = 20000;
                 passedParams.batchSize = 10;
                 passedParams.trainingSet = trainDS;
                 passedParams.validationSet = testDS;
                 passedParams.learningRate = 0.001;
                 passedParams.regularization = Regularizations.L2;
                 passedParams.regularizationRate = 0.001;
                 passedParams.nestrov = true;
                 passedParams.resilient = false;
                 passedParams.resilientUpdateAccelerationRate = 2;
                 passedParams.resilientUpdateSlowDownRate = 0.5;

                 passedParams.momentum = 0.5;
                 passedParams.NumberOfHiddenUnits = 100;

                 LastTrain(testDS, passedParams, trainer, "20kseed_"+seed+"_", seed);
             }*/
            /* Console.WriteLine("Training Adam");
             AdamParams adampassedParams = new AdamParams();
             IOptimizer adamtrainer = new Adam();

             adampassedParams.numberOfEpochs = 30000;
             adampassedParams.batchSize = 50;
             adampassedParams.trainingSet = trainDS;
             adampassedParams.validationSet = testDS;
             adampassedParams.learningRate = 0.001;
             adampassedParams.regularization = Regularizations.L2;
             adampassedParams.regularizationRate = 0.001;
             adampassedParams.NumberOfHiddenUnits = 100;

             LastTrain(testDS, adampassedParams, adamtrainer, "BS50_30kepochs_100_final_adam_hdn", 1);
             */


            //Loading and parsing cup dataset

            //  CupDataManager dm = new CupDataManager();
            //Loading the test dataset
            //DataSet TestSet = dm.LoadData(Properties.Settings.Default.TestSetLocation, 10, reportOsutput: false);
            //Loading the trained model
            //var n = AA1_MLP.Utilities.ModelManager.LoadNetwork("Final_hidn18_reg0.01_mo0.5_lr9E-06_model.AA1");

            //double MEE = 0;
            //applying the model on the test data
            //var predictions = ModelManager.GeneratorCUP(TestSet, n);
            //writing the results
            // File.WriteAllText("OMG_LOC-OSM2-TS.txt", string.Join("\n", predictions.Select(s => string.Join(",", s))));



        }

        private static void ReportHowCloseWeightsAcquiredFromDifferentSeedsAre()
        {
            string baseAddress = @"C:\\Users\\Ronin\\Desktop\\Release\\";


            List<string> modelsNames = new List<string>() { "20kseed_1_100_lr0.001_reg0.001.n", "20kseed_73_100_lr0.001_reg0.001.n", "20kseed_40_100_lr0.001_reg0.001.n", "20kseed_4_100_lr0.001_reg0.001.n", "20kseed_2_100_lr0.001_reg0.001.n", "20kseed_15_100_lr0.001_reg0.001.n" };

            DumpTrainedModelWeights(baseAddress, modelsNames);

            List<string> modelWeigtsFiles = new List<string>() { "20kseed_1_100_lr0.001_reg0.001.w", "20kseed_73_100_lr0.001_reg0.001.w", "20kseed_40_100_lr0.001_reg0.001.w", "20kseed_4_100_lr0.001_reg0.001.w", "20kseed_2_100_lr0.001_reg0.001.w", "20kseed_15_100_lr0.001_reg0.001.w" };
            List<Vector<double>> weights = new List<Vector<double>>();
            ReportWeightsDifference(baseAddress, modelWeigtsFiles, weights);
        }

        private static void DumpTrainedModelWeights(string baseAddress, List<string> modelsNames)
        {
            foreach (var modelName in modelsNames)
            {
                var n = ModelManager.LoadNetwork(Path.Combine(baseAddress, modelName));
                StringBuilder sb = new StringBuilder();
                for (int weightLayerIndex = 0; weightLayerIndex < n.Weights.Count; weightLayerIndex++)
                {
                    sb.Append(string.Join(",", n.Weights[weightLayerIndex].ToRowMajorArray())).Append(",");

                }
                sb.Remove(sb.Length - 1, 1);
                File.WriteAllText(Path.Combine(baseAddress, modelName.Replace(".n", ".w")), sb.ToString());
            }
        }

        private static void ReportWeightsDifference(string baseAddress, List<string> modelWeigtsFiles, List<Vector<double>> weights)
        {
            foreach (var mwf in modelWeigtsFiles)
            {
                Console.WriteLine(mwf.Replace("100_lr0.001_reg0.001.w", ""));
                weights.Add(CreateVector.Dense<Double>(File.ReadAllText(Path.Combine(baseAddress, mwf)).Split(',').Select(s => double.Parse(s)).ToArray()));
                Console.WriteLine("Euclidean Norm: {0} ", Math.Round(weights.Last().L2Norm(), 3));
            }
            int c = 0;
            foreach (var w in weights)
            {

                Console.WriteLine("weight:{0}", modelWeigtsFiles[c]);
                int i = 0;
                foreach (var w2 in weights)
                {
                    Console.WriteLine("EuclideanDistance({0},{1})", modelWeigtsFiles[c].Replace("100_lr0.001_reg0.001.w", ""), modelWeigtsFiles[i].Replace("100_lr0.001_reg0.001.w", ""));
                    Console.WriteLine(Math.Round(Math.Sqrt((w - w2).PointwisePower(2).Sum()), 3));
                    i++;
                }
                c++;

            }
        }

        private static void StandardizeData(DataSet trainDS)
        {
            for (int idxdataFold = 0; idxdataFold < trainDS.Inputs.ColumnCount; idxdataFold++)
            {
                double mean = trainDS.Inputs.Column(idxdataFold).Average();
                double std = Math.Sqrt((trainDS.Inputs.Column(idxdataFold) - mean).PointwisePower(2).Sum() / trainDS.Inputs.Column(idxdataFold).Count);
                trainDS.Inputs.SetColumn(idxdataFold, (trainDS.Inputs.Column(idxdataFold) - mean) / std);


            }
        }

        private static void LastTrain(DataSet testDS, INeuralTrainerParams passedParams, IOptimizer trainer, string prefix, int seed)
        {

            string path = prefix + passedParams.NumberOfHiddenUnits + "_lr" + passedParams.learningRate + "_reg" + passedParams.regularizationRate;
            //building the architecture
            Network n = new Network(new List<Layer>() {
                     new Layer(new ActivationIdentity(),true,10),
                     new Layer(new ActivationTanh(),true,passedParams.NumberOfHiddenUnits),
                  //   new Layer(new ActivationLeakyRelu(),true,40),


                     new Layer(new ActivationIdentity(),false,2),
                     }, false, AA1_MLP.Enums.WeightsInitialization.Xavier, seed);
            passedParams.network = n;
            var watch = System.Diagnostics.Stopwatch.StartNew();
            List<double[]> learningCurve = trainer.Train(passedParams);
            watch.Stop();
            var elapsedMs = watch.ElapsedMilliseconds;
            Console.WriteLine("elapsed Time:{0} ms", elapsedMs);
            double MEE = 0;
            double MSE = 0;

            var log = ModelManager.TesterCUPRegression(testDS, n, out MEE, out  MSE);

            File.WriteAllText(path + ".txt", string.Join("\n", learningCurve.Select(s => string.Join(",", s))));
            File.AppendAllText(path + ".txt", "\nMEE:" + MEE + "MSE:" + MSE);
            File.WriteAllText(path + "predVsActual.txt", string.Join("\n", log.Select(s => string.Join(",", s))));



            ModelManager.SaveNetowrk(n, path + ".n");

        }
    }
}
