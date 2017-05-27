using AA1_MLP.Activations;
using AA1_MLP.DataManager;
using AA1_MLP.Entities;
using AA1_MLP.Entities.Trainers;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLPTestDemo
{
    /// <summary>
    /// a simple demo on how to create a network, load, train and save it
    /// </summary>
    class Program
    {

        static void Main(string[] args)
        {
            //path where the trained model shall be saved in or loaded from
            string path2SaveModel = Properties.Settings.Default.path2SaveModel;


            // Building a simple network
            Network n = new Network(new List<Layer>() {
                new Layer(new ActivationIdentity(),true,17),
                new Layer(new ActivationSigmoid(),true,3),
                new Layer(new ActivationSigmoid(),false,1),
                }, false, AA1_MLP.Enums.WeightsInitialization.Uniform);


            //loading monks trainingSet and testing datasets
            DataManager dm = new DataManager();
            DataSet ds = dm.LoadData(Properties.Settings.Default.TrainingSetLocation, 17);
            DataSet dt = dm.LoadData(Properties.Settings.Default.TestSetLocation, 17);


            //Loading a network should be like ...
            //var n = AA1_MLP.Utilities.ModelManager.LoadNetwork(PATH TO THE MODEL BINARY FILE);


            //Creating a backpropagation trainer
            BackPropagation br = new BackPropagation();

            //Calling the Train method of the trainer with the desired parameters
            var learningCurve = br.Train(n, ds, learningRate: 1, numberOfEpochs: 500, shuffle: false, debug: n.Debug, momentum: 0.4, resilient: true, resilientUpdateAccelerationRate: 0.5, resilientUpdateSlowDownRate: 0.2, regularization: AA1_MLP.Enums.Regularizations.None, regularizationRate: 0.01, validationSet: dt, batchSize: 7);

            //writing the learning curve data to desk (ugly for memory, but simple)
            File.WriteAllText(Properties.Settings.Default.LearningCurveLocation, string.Join("\n", learningCurve.Select(s => string.Join(",", s))));


            //saving the trained network to desk
            //AA1_MLP.Utilities.ModelManager.SaveNetowrk(n, path2SaveModel);


            //Testing the model and outputing the confusion matrix
            AA1_MLP.Utilities.ModelManager.TesterMonkClassification(dt, n, 0.5, Properties.Settings.Default.TestReportLocation);



            //computing the ROC plot parameters

            //var file = new System.IO.StreamWriter(Properties.Settings.Default.ROCParamsStorageLocation);
            //for (float i = 0.005f; i <= 1.0f; i += 0.005f)
            //{
            //    var TprFpr = AA1_MLP.Utilities.ModelManager.TesterMonkClassification(dt, n, i);
            //    file.WriteLine(string.Format("{0},{1},{2}", i, TprFpr[0], TprFpr[1]));
            //}
            //file.Close();


        }



    }
}
