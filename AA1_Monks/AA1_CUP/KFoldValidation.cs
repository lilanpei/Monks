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
    /// For screening with a k fold validation
    /// </summary>
    /// 
    /*
      After conducting a few experiments with k-fold and no regularization, 
     * we found that the runs are diverging at very different number of epochs which made it difficult to set a unified criterion for early
     * stopping or decide the last number of epochs for the training on the whole dataset
     * we decided to utilize regularization + a standard stopping criterion [no improvement in the loss anymore, could be examined easily from the learning curve of the runs] to decide the final number of epochs
     *
      
     
     
     */
    public class KFoldValidation : IScreening
    {

        DataSet TrainDataset = null;
        DataSet ValidationSplit = null;



        public void Screen(AA1_MLP.Entities.DataSet wholeSet, int k = 0)
        {

            //Calling the Train method of the trainer with the desired parameters
            //n, ds, learningRate: .3, numberOfEpochs: 200, shuffle: false, debug: n.Debug, nestrov:false, momentum:0.9, resilient: false, resilientUpdateAccelerationRate: 0.3,
            //resilientUpdateSlowDownRate: 0.1, regularization: AA1_MLP.Enums.RegularizationRates.L2, regularizationRate: 0.001, validationSet: dt, batchSize: 7

            string reportsDirectory = "SGDKFoldsReports";
            if (Directory.Exists(reportsDirectory))
            {
                Directory.Delete(reportsDirectory, true);
            }
            Directory.CreateDirectory(reportsDirectory);

            List<double> momentums = new List<double> { 0, 0.5 };
            List<double> learningRates = new List<double> { 0.005, 0.01 };
            List<double> regularizationRates = new List<double> { 0, 0.001 };

            GradientDescentParams passedParams = new GradientDescentParams();
            IOptimizer trainer = new Gradientdescent();
            //AdamParams passedParams = new AdamParams();
            //IOptimizer trainer = new Adam();
            passedParams.numberOfEpochs = 5000;
            passedParams.batchSize = 10;
            for (int nh = 100; nh >= 10; nh -= 20)
                for (int idxmo = 0; idxmo < momentums.Count; idxmo++)
                    for (int idxLR = 0; idxLR < learningRates.Count; idxLR++)
                        for (int idxReg = 0; idxReg < regularizationRates.Count; idxReg++)
                        {

                            passedParams.learningRate = learningRates[idxLR];
                            passedParams.regularization = Regularizations.L2;
                            passedParams.regularizationRate = regularizationRates[idxReg];

                            passedParams.nestrov = true;
                            passedParams.momentum = momentums[idxmo];
                            passedParams.resilient = false;
                            passedParams.resilientUpdateAccelerationRate = 0.3;
                            passedParams.resilientUpdateSlowDownRate = 0.1;
                            passedParams.NumberOfHiddenUnits = nh;

                            RunKFoldWithSetOfParams(wholeSet, k, passedParams, trainer, reportsDirectory);


                        }

        }

        private void RunKFoldWithSetOfParams(AA1_MLP.Entities.DataSet wholeSet, int k, INeuralTrainerParams passedParams, IOptimizer trainer, string reportsPath)
        {


            string kRunFolderName = string.Format("hdn{0}_k{1}_lr{2}_reg{3}", passedParams.NumberOfHiddenUnits, k, passedParams.learningRate, passedParams.regularizationRate);
            string KRunfolderPath = Path.Combine(reportsPath, kRunFolderName);

            if (passedParams is GradientDescentParams)
            {
                KRunfolderPath = string.Format("{0}_mo{1}", KRunfolderPath, ((GradientDescentParams)passedParams).momentum);
            }

            if (Directory.Exists(KRunfolderPath))
            {
                Directory.Delete(KRunfolderPath);
            }

            Directory.CreateDirectory(KRunfolderPath);






            double avgMSE = 0;


            double MEE = 0, MSE = 0;








            int sizeOfDataFold = wholeSet.Labels.RowCount / k;






            //the training set split
            TrainDataset = new DataSet(
               inputs: wholeSet.Inputs.SubMatrix(sizeOfDataFold,
                 wholeSet.Inputs.RowCount - sizeOfDataFold, 0, wholeSet.Inputs.ColumnCount),
          labels: wholeSet.Labels.SubMatrix(sizeOfDataFold,
                 wholeSet.Labels.RowCount - sizeOfDataFold, 0, wholeSet.Labels.ColumnCount));
            //the validation set
            ValidationSplit = new DataSet(
              inputs: wholeSet.Inputs.SubMatrix(0, sizeOfDataFold, 0, wholeSet.Inputs.ColumnCount),
         labels: wholeSet.Labels.SubMatrix(0, sizeOfDataFold, 0, wholeSet.Labels.ColumnCount));


            Console.WriteLine("Run number:{0}", 0);
            passedParams.trainingSet = TrainDataset;
            passedParams.validationSet = ValidationSplit;




            var lc = RunExperiment(trainer, passedParams, out  MEE, out  MSE);
            File.WriteAllText(Path.Combine(KRunfolderPath, "0_learningCurve.txt"), string.Join("\n", lc.Select(s => string.Join(",", s))));

            avgMSE += MSE;


            for (int idxdataFold = 1; idxdataFold < k - 1; idxdataFold++)
            {
                Console.WriteLine("Run number:{0}", idxdataFold);

                //the training set split
                TrainDataset = new DataSet(
                   inputs: wholeSet.Inputs.SubMatrix(0, idxdataFold * sizeOfDataFold, 0, wholeSet.Inputs.ColumnCount
                   ).Stack(wholeSet.Inputs.SubMatrix(idxdataFold * sizeOfDataFold + sizeOfDataFold,
                     wholeSet.Inputs.RowCount - idxdataFold * sizeOfDataFold - sizeOfDataFold, 0, wholeSet.Inputs.ColumnCount)),
              labels: wholeSet.Labels.SubMatrix(0, idxdataFold * sizeOfDataFold, 0, wholeSet.Labels.ColumnCount
                   ).Stack(wholeSet.Labels.SubMatrix(idxdataFold * sizeOfDataFold + sizeOfDataFold,
                     wholeSet.Labels.RowCount - idxdataFold * sizeOfDataFold - sizeOfDataFold, 0, wholeSet.Labels.ColumnCount)));
                //the validation set
                ValidationSplit = new DataSet(
                  inputs: wholeSet.Inputs.SubMatrix(idxdataFold * sizeOfDataFold, sizeOfDataFold, 0, wholeSet.Inputs.ColumnCount),
             labels: wholeSet.Labels.SubMatrix(idxdataFold * sizeOfDataFold, sizeOfDataFold, 0, wholeSet.Labels.ColumnCount));
                passedParams.trainingSet = TrainDataset;
                passedParams.validationSet = ValidationSplit;
                lc = RunExperiment(trainer, passedParams, out  MEE, out  MSE);
                File.WriteAllText(Path.Combine(KRunfolderPath, idxdataFold + "_learningCurve.txt"), string.Join("\n", lc.Select(s => string.Join(",", s))));

                avgMSE += MSE;
            }

            //the training set split
            TrainDataset = new DataSet(
               inputs: wholeSet.Inputs.SubMatrix(0, (k - 1) * sizeOfDataFold, 0, wholeSet.Inputs.ColumnCount
               ).Stack(wholeSet.Inputs.SubMatrix((k - 1) * sizeOfDataFold + sizeOfDataFold,
                 wholeSet.Inputs.RowCount - (k - 1) * sizeOfDataFold - sizeOfDataFold, 0, wholeSet.Inputs.ColumnCount)),
          labels: wholeSet.Labels.SubMatrix(0, (k - 1) * sizeOfDataFold, 0, wholeSet.Labels.ColumnCount
               ).Stack(wholeSet.Labels.SubMatrix((k - 1) * sizeOfDataFold + sizeOfDataFold,
                 wholeSet.Labels.RowCount - (k - 1) * sizeOfDataFold - sizeOfDataFold, 0, wholeSet.Labels.ColumnCount)));
            //the validation set
            ValidationSplit = new DataSet(
              inputs: wholeSet.Inputs.SubMatrix((k - 1) * sizeOfDataFold, sizeOfDataFold, 0, wholeSet.Inputs.ColumnCount),
         labels: wholeSet.Labels.SubMatrix((k - 1) * sizeOfDataFold, sizeOfDataFold, 0, wholeSet.Labels.ColumnCount));

            Console.WriteLine("Run number:{0}", k - 1);

            passedParams.trainingSet = TrainDataset;
            passedParams.validationSet = ValidationSplit;
            lc = RunExperiment(trainer, passedParams, out  MEE, out  MSE);
            File.WriteAllText(Path.Combine(KRunfolderPath, (k - 1) + "_learningCurve.txt"), string.Join("\n", lc.Select(s => string.Join(",", s))));

            avgMSE += MSE;


            avgMSE /= k;

            Console.WriteLine("Average MSE:{0}", avgMSE);
            File.AppendAllLines(Path.Combine(reportsPath, "avgMSEs"), new string[] { string.Format("{0},{1}", kRunFolderName, avgMSE) });
        }

        public List<double[]> RunExperiment(IOptimizer optimizer, INeuralTrainerParams passedParams, out double MEE, out double MSE)
        {

            //building the architecture
            Network n = new Network(new List<Layer>() {
                     new Layer(new ActivationIdentity(),true,10),
                     new Layer(new ActivationTanh(),true,passedParams.NumberOfHiddenUnits),
                  //   new Layer(new ActivationLeakyRelu(),true,40),


                     new Layer(new ActivationIdentity(),false,2),
                     }, false, AA1_MLP.Enums.WeightsInitialization.Xavier);
            passedParams.network = n;
            List<double[]> learningCurve = optimizer.Train(passedParams);
            MEE = 0;
            MSE = 0;
            var log = ModelManager.TesterCUPRegression(passedParams.validationSet, n, out MEE, out  MSE);

            return learningCurve;
        }



    }
}
