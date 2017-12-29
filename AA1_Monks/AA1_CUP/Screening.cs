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
    public class Screening
    {
        public static void Screen()
        {


            //Loading and parsing cup dataset
            CupDataManager dm = new CupDataManager();
            DataSet wholeSet = dm.LoadData(Properties.Settings.Default.TrainingSetLocation, 10, 2);
            //wholeSet.Inputs = wholeSet.Inputs.NormalizeColumns(2) ;

           // wholeSet.Labels = wholeSet.Labels/10;


            //standardiing data

            //x=(x-mean)/std

             /* for (int i = 0; i < wholeSet.Inputs.ColumnCount; i++)
              {
                  double mean = wholeSet.Inputs.Column(i).Average();
                  double std = Math.Sqrt((wholeSet.Inputs.Column(i) - mean).PointwisePower(2).Sum() / wholeSet.Inputs.Column(i).Count);
                  wholeSet.Inputs.SetColumn(i, (wholeSet.Inputs.Column(i) - mean) / std);


              }*/


           /* for (int i = 0; i < wholeSet.Inputs.ColumnCount; i++)
            {
                double min = wholeSet.Inputs.Column(i).Min();
                double max = wholeSet.Inputs.Column(i).Max();
                double max_min = max - min;
                wholeSet.Inputs.SetColumn(i, (wholeSet.Inputs.Column(i) - min) / max_min);

            }*/

            //the training set split
            int trainSplit = (int)(0.6 * wholeSet.Inputs.RowCount);
            DataSet TrainDataset = new DataSet(
               inputs: wholeSet.Inputs.SubMatrix(0, trainSplit, 0, wholeSet.Inputs.ColumnCount),
          labels: wholeSet.Labels.SubMatrix(0, trainSplit, 0, wholeSet.Labels.ColumnCount));
            //the validation set
            DataSet ValidationSplit = new DataSet(
              inputs: wholeSet.Inputs.SubMatrix(trainSplit, (wholeSet.Inputs.RowCount - trainSplit) / 2, 0, wholeSet.Inputs.ColumnCount),
         labels: wholeSet.Labels.SubMatrix(trainSplit, (wholeSet.Inputs.RowCount - trainSplit) / 2, 0, wholeSet.Labels.ColumnCount));
            //the hold out set for reporting the MEE of the model on the data
            DataSet TestDatasetSplit = new DataSet(
       inputs: wholeSet.Inputs.SubMatrix(trainSplit + (wholeSet.Inputs.RowCount - trainSplit) / 2, (int)Math.Ceiling((double)(wholeSet.Inputs.RowCount - trainSplit) / 2), 0, wholeSet.Inputs.ColumnCount),
       labels: wholeSet.Labels.SubMatrix(trainSplit + (wholeSet.Inputs.RowCount - trainSplit) / 2, (int)Math.Ceiling((double)(wholeSet.Inputs.RowCount - trainSplit) / 2), 0, wholeSet.Labels.ColumnCount));

            Gradientdescent bp = new Gradientdescent();

            //will hold a number of possible values for the hidden units to try
            List<int> PossibleHiddenUnits = new List<int>();
            for (int numberOfUnits = 30; numberOfUnits <= 100; numberOfUnits += 10)
            {
                PossibleHiddenUnits.Add(numberOfUnits);
            }
            //holds different values for the Regularization to try
            List<double> RegularizationRates = new List<double>() { 0.001, 0.0001 };
            //holds different values for the momentum to try for training
            List<double> Momentums = new List<double>() { 0 };
            //holds different values for the learning rate to try for training
            List<double> learningRate = new List<double>() { 0.0007, 0.001, 0.0001 };

            //these directories will hold the experiments results
            Directory.CreateDirectory("learningCurves");
            Directory.CreateDirectory("scatters");
            Directory.CreateDirectory("models");
            // a simple resume mechanism, in case the search was interrupted
            if (!File.Exists("passed.txt"))
            {
                File.WriteAllText("passed.txt", "0,0,0,0\n");

            }
            var init = File.ReadLines("passed.txt").Last().Split(',');

            int vs = int.Parse(init[0]), bs = int.Parse(init[1]), ms = int.Parse(init[2]), us = int.Parse(init[3]);
            //  File.Delete("passed.txt");
            for (int v = vs; v < PossibleHiddenUnits.Count; v++)
            {
                var hidn = PossibleHiddenUnits[v];
                for (int b = bs; b < RegularizationRates.Count; b++)
                {
                    var reg = RegularizationRates[b];
                    for (int m = ms; m < Momentums.Count; m++)
                    {
                        var mo = Momentums[m];
                        for (int u = us; u < learningRate.Count; u++)
                        {


                            var lr = learningRate[u];
                            string pre = string.Format("hidn{0}_reg{1}_mo{2}_lr{3}", hidn, reg, mo, lr);

                            //building the architecture
                            Network n = new Network(new List<Layer>() {
                     new Layer(new ActivationIdentity(),true,10),
                     new Layer(new ActivationSigmoid(),true,hidn),
                  //   new Layer(new ActivationLeakyRelu(),true,40),


                     new Layer(new ActivationIdentity(),false,2),
                     }, false, AA1_MLP.Enums.WeightsInitialization.Grot);

                            /*try
                            {*/




                            Gradientdescent br = new Gradientdescent();

                            //Calling the Train method of the trainer with the desired parameters
                            //n, ds, learningRate: .3, numberOfEpochs: 200, shuffle: false, debug: n.Debug, nestrov:false, momentum:0.9, resilient: false, resilientUpdateAccelerationRate: 0.3,
                            //resilientUpdateSlowDownRate: 0.1, regularization: AA1_MLP.Enums.RegularizationRates.L2, regularizationRate: 0.001, validationSet: dt, batchSize: 7
                            GradientDescentParams passedParams = new GradientDescentParams();
                            passedParams.network = n;
                            passedParams.trainingSet = TrainDataset;
                            passedParams.learningRate = lr;
                            passedParams.numberOfEpochs = 100000;
                            passedParams.shuffle = false;
                            passedParams.debug = n.Debug;
                            passedParams.nestrov = false;
                            passedParams.momentum = mo;
                            passedParams.resilient = false;
                            passedParams.resilientUpdateAccelerationRate = 0.3;
                            passedParams.resilientUpdateSlowDownRate = 0.1;
                            passedParams.regularization = Regularizations.L2;
                            passedParams.regularizationRate = reg;
                            passedParams.validationSet = TestDatasetSplit;
                            passedParams.batchSize = 10;
                            passedParams.MEE = true;



                            var learningCurve = br.Train(passedParams);


                            /* Adam br = new Adam();

                             //Calling the Train method of the trainer with the desired parameters
                             //n, ds, learningRate: .3, numberOfEpochs: 200, shuffle: false, debug: n.Debug, nestrov:false, momentum:0.9, resilient: false, resilientUpdateAccelerationRate: 0.3,
                             //resilientUpdateSlowDownRate: 0.1, regularization: AA1_MLP.Enums.RegularizationRates.L2, regularizationRate: 0.001, validationSet: dt, batchSize: 7
                             AdamParams passedParams = new AdamParams();
                             passedParams.network = n;
                             passedParams.trainingSet = TrainDataset;
                             passedParams.learningRate = lr;
                             passedParams.numberOfEpochs = 10000;
                             passedParams.shuffle = true;
                             passedParams.debug = n.Debug;
                             passedParams.regularization = Regularizations.L2;
                             passedParams.regularizationRate = reg;
                             passedParams.validationSet = TestDatasetSplit;
                             passedParams.batchSize = 7;




                             var learningCurve = br.Train(passedParams);

                             */


                            ////the training loop
                            // var learningCurve = bp.Train(n,
                            //          TrainDataset,
                            //          lr,
                            //          100000,
                            //          true,
                            //          regularizationRate: reg,
                            //          regularization: AA1_MLP.Enums.RegularizationRates.L2,
                            //          momentum: mo,
                            //          validationSet: TestDatasetSplit,

                            //          MEE: true
                            //     /*  resilient: true, resilientUpdateAccelerationRate: 10, resilientUpdateSlowDownRate: 1,
                            //       reduceLearningRate: true,
                            //       learningRateReduction: 0.8,
                            //       numberOfReductions: 3,
                            //       learningRateReductionAfterEpochs: 7500*/
                            //          );

                            //writing the learning curve data to desk (ugly for memory, but simple)
                            File.WriteAllText("learningCurves/" + pre + "learningCurve.txt", string.Join("\n", learningCurve.Select(s => string.Join(",", s))));

                            //saving the trained model
                            AA1_MLP.Utilities.ModelManager.SaveNetowrk(n, "models/" + pre + "_model.AA1");
                            // var n = AA1_MLP.Utilities.ModelManager.LoadNetwork("model.AA1");

                            double MEE = 0;
                            var log = ModelManager.TesterCUPRegression(TestDatasetSplit, n, out MEE);

                            //reporting the scatter plot of the output against the actual predictions on the held out dataset split
                            File.WriteAllText("scatters/" + pre + "scatter.txt", string.Join("\n", log.Select(s => string.Join(",", s))));

                            File.AppendAllText("MEEs.txt", pre + ":" + MEE + "\n");

                            Console.WriteLine(MEE);

                            /*  }
                              catch (Exception e)
                              {
                                  Console.WriteLine(pre + " Failed!");
                                  File.AppendAllText("fails.txt", pre + "\n");

                                  Console.WriteLine(e.Message);
                              }*/
                            us = (u + 1) % learningRate.Count == 0 ? 0 : us;
                            File.AppendAllText("passed.txt", string.Format("{0},{1},{2},{3}\n", (((u + 1) % learningRate.Count == 0) && ((m + 1) % Momentums.Count == 0) && ((b + 1) % RegularizationRates.Count == 0)) ? v + 1 : v, ((((u + 1) % learningRate.Count == 0) && (m + 1) % Momentums.Count == 0) ? b + 1 : b) % RegularizationRates.Count, ((u + 1) % learningRate.Count == 0 ? m + 1 : m) % Momentums.Count, (u + 1) % learningRate.Count));

                        }
                        ms = (m + 1) % Momentums.Count == 0 ? 0 : ms;
                    }
                    bs = (b + 1) % RegularizationRates.Count == 0 ? 0 : bs;
                }

            }



        }

        private static List<double[]> TrainWithSGD(Network n, DataSet ds, DataSet dt)
        {
            Gradientdescent br = new Gradientdescent();

            //Calling the Train method of the trainer with the desired parameters
            //n, ds, learningRate: .3, numberOfEpochs: 200, shuffle: false, debug: n.Debug, nestrov:false, momentum:0.9, resilient: false, resilientUpdateAccelerationRate: 0.3,
            //resilientUpdateSlowDownRate: 0.1, regularization: AA1_MLP.Enums.RegularizationRates.L2, regularizationRate: 0.001, validationSet: dt, batchSize: 7
            GradientDescentParams passedParams = new GradientDescentParams();
            passedParams.network = n;
            passedParams.trainingSet = ds;
            passedParams.learningRate = 1;
            passedParams.numberOfEpochs = 200;
            passedParams.shuffle = false;
            passedParams.debug = n.Debug;
            passedParams.nestrov = true;
            passedParams.momentum = 0.9;
            passedParams.resilient = false;
            passedParams.resilientUpdateAccelerationRate = 0.3;
            passedParams.resilientUpdateSlowDownRate = 0.1;
            passedParams.regularization = Regularizations.L2;
            passedParams.regularizationRate = 0.001;
            passedParams.validationSet = dt;
            passedParams.batchSize = 7;



            var learningCurve = br.Train(passedParams);
            return learningCurve;
        }

        private static List<double[]> TrainWithAdam(Network n, DataSet ds, DataSet dt)
        {
            Adam br = new Adam();

            //Calling the Train method of the trainer with the desired parameters
            //n, ds, learningRate: .3, numberOfEpochs: 200, shuffle: false, debug: n.Debug, nestrov:false, momentum:0.9, resilient: false, resilientUpdateAccelerationRate: 0.3,
            //resilientUpdateSlowDownRate: 0.1, regularization: AA1_MLP.Enums.RegularizationRates.L2, regularizationRate: 0.001, validationSet: dt, batchSize: 7
            AdamParams passedParams = new AdamParams();
            passedParams.network = n;
            passedParams.trainingSet = ds;
            passedParams.learningRate = 0.09;
            passedParams.numberOfEpochs = 200;
            passedParams.shuffle = true;
            passedParams.debug = n.Debug;
            passedParams.regularization = Regularizations.None;
            passedParams.regularizationRate = 0.0001;
            passedParams.validationSet = dt;
            passedParams.batchSize = 7;




            var learningCurve = br.Train(passedParams);
            return learningCurve;
        }



    }
}