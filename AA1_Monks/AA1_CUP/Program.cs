using AA1_MLP.Activations;
using AA1_MLP.Entities;
using AA1_MLP.Entities.Trainers;
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
            CupDataManager dm = new CupDataManager();
            DataSet wholeSet = dm.LoadData(Properties.Settings.Default.TrainingSetLocation, 10, 2);
            //the training set split
            int trainSplit = (int)(0.6 * wholeSet.Inputs.RowCount);
            DataSet trainingSplit = new DataSet(
               inputs: wholeSet.Inputs.SubMatrix(0, trainSplit, 0, wholeSet.Inputs.ColumnCount),
          labels: wholeSet.Labels.SubMatrix(0, trainSplit, 0, wholeSet.Labels.ColumnCount));
            //the validation set
         //   DataSet ValidationSplit = new DataSet(
         //     inputs: wholeSet.Inputs.SubMatrix(trainSplit, (wholeSet.Inputs.RowCount - trainSplit) / 2, 0, wholeSet.Inputs.ColumnCount),
         //labels: wholeSet.Labels.SubMatrix(trainSplit, (wholeSet.Inputs.RowCount - trainSplit) / 2, 0, wholeSet.Labels.ColumnCount));
            //the hold out set for reporting the MEE of the model on the data
            DataSet TestSplit = new DataSet(
       inputs: wholeSet.Inputs.SubMatrix(trainSplit , (int)Math.Ceiling((double)(wholeSet.Inputs.RowCount - trainSplit)), 0, wholeSet.Inputs.ColumnCount),
       labels: wholeSet.Labels.SubMatrix(trainSplit, (int)Math.Ceiling((double)(wholeSet.Inputs.RowCount - trainSplit)), 0, wholeSet.Labels.ColumnCount));

            BackPropagation bp = new BackPropagation();

            //will hold a number of possible values for the hidden units to try
            List<int> PossibleHiddenUnits = new List<int>();
            for (int numberOfUnits = 18; numberOfUnits < 21; numberOfUnits += 1)
            {
                PossibleHiddenUnits.Add(numberOfUnits);
            }
            //holds different values for the Regularization to try
            List<double> Regularizations = new List<double>() { 0.01,  0.0001 };
            //holds different values for the momentum to try for training
            List<double> Momentums = new List<double>() { 0.7, 0.9 };
            //holds different values for the learning rate to try for training
            List<double> learningRate = new List<double>() { 0.000009, 0.00001 };

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
                for (int b = bs; b < Regularizations.Count; b++)
                {
                    var reg = Regularizations[b];
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
                     new Layer(new ActivationLeakyRelu(),true,hidn),
                  //   new Layer(new ActivationLeakyRelu(),true,40),


                     new Layer(new ActivationLeakyRelu(),false,2),
                     }, false, AA1_MLP.Enums.WeightsInitialization.Xavier);

                            try
                            {

                                //the training loop
                                var learningCurve = bp.Train(n,
                                         trainingSplit,
                                         lr,
                                         100000,
                                         true,
                                         regularizationRate: reg,
                                         regularization: AA1_MLP.Enums.Regularizations.L2,
                                         momentum: mo,
                                         validationSet: TestSplit,

                                         MEE: true
                                    /*  resilient: true, resilientUpdateAccelerationRate: 10, resilientUpdateSlowDownRate: 1,
                                      reduceLearningRate: true,
                                      learningRateReduction: 0.8,
                                      numberOfReductions: 3,
                                      learningRateReductionAfterEpochs: 7500*/
                                         );

                                //writing the learning curve data to desk (ugly for memory, but simple)
                                File.WriteAllText("learningCurves/" + pre + "learningCurve.txt", string.Join("\n", learningCurve.Select(s => string.Join(",", s))));

                                //saving the trained model
                                AA1_MLP.Utilities.ModelManager.SaveNetowrk(n, "models/" + pre + "_model.AA1");
                                // var n = AA1_MLP.Utilities.ModelManager.LoadNetwork("model.AA1");

                                double MEE = 0;
                                var log = ModelManager.TesterCUPRegression(TestSplit, n, out MEE);

                                //reporting the scatter plot of the output against the actual predictions on the held out dataset split
                                File.WriteAllText("scatters/" + pre + "scatter.txt", string.Join("\n", log.Select(s => string.Join(",", s))));

                                File.AppendAllText("MEEs.txt", pre + ":" + MEE + "\n");

                                Console.WriteLine(MEE);

                            }
                            catch (Exception)
                            {
                                Console.WriteLine(pre + " Failed!");
                                File.AppendAllText("fails.txt", pre + "\n");


                            }
                            us = (u + 1) % learningRate.Count == 0 ? 0 : us;
                            File.AppendAllText("passed.txt", string.Format("{0},{1},{2},{3}\n", (((u + 1) % learningRate.Count == 0) && ((m + 1) % Momentums.Count == 0) && ((b + 1) % Regularizations.Count == 0)) ? v + 1 : v, ((((u + 1) % learningRate.Count == 0) && (m + 1) % Momentums.Count == 0) ? b + 1 : b) % Regularizations.Count, ((u + 1) % learningRate.Count == 0 ? m + 1 : m) % Momentums.Count, (u + 1) % learningRate.Count));

                        }
                        ms = (m + 1) % Momentums.Count == 0 ? 0 : ms;
                    }
                    bs = (b + 1) % Regularizations.Count == 0 ? 0 : bs;
                }

            }



        }
    }
}
