using AA1_MLP.Activations;
using AA1_MLP.Entities;
using AA1_MLP.Entities.Trainers;
using AA1_MLP.Entities.TrainersParams;
using AA1_MLP.Enums;
using AA1_MLP.Utilities;
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


            AA1_MLP.DataManagers.CupDataManager dm = new AA1_MLP.DataManagers.CupDataManager();
            DataSet trainDS = dm.LoadData(@"C:\Users\Ronin\Documents\monks\Monks\UsedFiles\TrainValSplits\60percenttrain.txt", 10, 2, standardize: true);
            DataSet testDS = dm.LoadData(@"C:\Users\Ronin\Documents\monks\Monks\UsedFiles\TrainValSplits\60percenttest.txt", 10, 2, standardize: true);




            /*AdamParams passedParams = new AdamParams();
            IOptimizer trainer = new Adam();*/
          /*  Console.WriteLine("training SGD");
            GradientDescentParams passedParams = new GradientDescentParams();
            Gradientdescent trainer = new Gradientdescent();
            passedParams.numberOfEpochs = 10;
            passedParams.batchSize = 50;
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

            LastTrain(testDS, passedParams, trainer, "BS50_10epochs_mo0.5_100_final_sgdnestrov_hdn", 1);*/
            Console.WriteLine("Training Adam");
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
