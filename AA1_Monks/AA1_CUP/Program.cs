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
    public class Program
    {
        static void Main(string[] args)
        {
            // Building a simple network


            CupDataManager dm = new CupDataManager();
            DataSet wholeSet = dm.LoadData(Properties.Settings.Default.TrainingSetLocation, 10, 2);
            int trainSplit = (int)(0.6 * wholeSet.Inputs.RowCount);


            DataSet trainingSplit = new DataSet(
               inputs: wholeSet.Inputs.SubMatrix(0, trainSplit, 0, wholeSet.Inputs.ColumnCount),
          labels: wholeSet.Labels.SubMatrix(0, trainSplit, 0, wholeSet.Labels.ColumnCount));

            DataSet ValidationSplit = new DataSet(
              inputs: wholeSet.Inputs.SubMatrix(trainSplit, (wholeSet.Inputs.RowCount - trainSplit) / 2, 0, wholeSet.Inputs.ColumnCount),
         labels: wholeSet.Labels.SubMatrix(trainSplit, (wholeSet.Inputs.RowCount - trainSplit) / 2, 0, wholeSet.Labels.ColumnCount));

            DataSet TestSplit = new DataSet(
       inputs: wholeSet.Inputs.SubMatrix(trainSplit + (wholeSet.Inputs.RowCount - trainSplit) / 2, (int)Math.Ceiling((double)(wholeSet.Inputs.RowCount - trainSplit) / 2), 0, wholeSet.Inputs.ColumnCount),
       labels: wholeSet.Labels.SubMatrix(trainSplit + (wholeSet.Inputs.RowCount - trainSplit) / 2, (int)Math.Ceiling((double)(wholeSet.Inputs.RowCount - trainSplit) / 2), 0, wholeSet.Labels.ColumnCount));

            BackPropagation bp = new BackPropagation();


            List<int> PossibleHiddenUnits = new List<int>();
            for (int numberOfUnits = 2; numberOfUnits < 100; numberOfUnits += 8)
            {
                PossibleHiddenUnits.Add(numberOfUnits);
            }
            List<double> Regularizations = new List<double>() { 1, 0.1, 0.01, 0.001, 0.0001, 0.00001 };

            List<double> Momentums = new List<double>() { 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 };
            List<double> learningRate = new List<double>() { 1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.000001 };

            Directory.CreateDirectory("learningCurves");
            Directory.CreateDirectory("scatters");
            Directory.CreateDirectory("models");

            foreach (var hidn in PossibleHiddenUnits)
            {
                foreach (var reg in Regularizations)
                {
                    foreach (var mo in Momentums)
                    {
                        foreach (var lr in learningRate)
                        {

                            string pre = string.Format("hidn{0}_reg{1}_mo{2}_lr{3}", hidn, reg, mo, lr);

                            Network n = new Network(new List<Layer>() {
                     new Layer(new ActivationIdentity(),true,10),
                     new Layer(new ActivationLeakyRelu(),true,hidn),
                  //   new Layer(new ActivationLeakyRelu(),true,40),


                     new Layer(new ActivationLeakyRelu(),false,2),
                     }, new MathNet.Numerics.Distributions.Normal(0, 1), true, false);

                            try
                            {


                                var learningCurve = bp.Train(n,
                                         trainingSplit,
                                         lr,
                                         100000,
                                         true,
                                         regularizationRate: reg,
                                         regularization: AA1_MLP.Enums.Regularizations.L2,
                                         momentum: mo,
                                         validationSet: ValidationSplit,

                                         MEE: true
                                    /*  resilient: true, resilientUpdateAccelerationRate: 10, resilientUpdateSlowDownRate: 1,
                                      reduceLearningRate: true,
                                      learningRateReduction: 0.8,
                                      numberOfReductions: 3,
                                      learningRateReductionAfterEpochs: 7500*/
                                         );

                                //writing the learning curve data to desk (ugly for memory, but simple)
                                File.WriteAllText("learningCurves/" + pre + "learningCurve.txt", string.Join("\n", learningCurve.Select(s => string.Join(",", s))));

                                    AA1_MLP.Utilities.ModelManager.SaveNetowrk(n, "models/"+pre+"_model.AA1");
                                // var n = AA1_MLP.Utilities.ModelManager.LoadNetwork("model.AA1");

                                double MEE = 0;
                                var log = ModelManager.TesterRegression(TestSplit, n, out MEE);


                                File.WriteAllText("scatters/" + pre + "scatter.txt", string.Join("\n", log.Select(s => string.Join(",", s))));

                                File.AppendAllText("MEEs.txt", pre + ":" + MEE + "\n");

                                Console.WriteLine(MEE);

                            }
                            catch (Exception)
                            {
                                Console.WriteLine(pre + " Failed!");
                                File.AppendAllText("fails.txt", pre + "\n");


                            }
                        }
                    }

                }

            }



        }
    }
}
