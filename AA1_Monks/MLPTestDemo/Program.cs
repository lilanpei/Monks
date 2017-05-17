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
    class Program
    {
        static void Main(string[] args)
        {
            //path where the trained model shall be saved in or loaded from
            string path2SaveModel = Properties.Settings.Default.path2SaveModel;


            // Building a simple network
              Network n = new Network(new List<Layer>() {
                new Layer(new ActivationIdentity(),true,17),
                new Layer(new ActivationSigmoid(),true,2),
                new Layer(new ActivationSigmoid(),false,1),
                }, false);

                

            //loading monks training and testing datasets
            DataSet ds = DataManager.LoadMonksData(Properties.Settings.Default.TrainingSetLocation, 17);
            DataSet dt = DataManager.LoadMonksData(Properties.Settings.Default.TestSetLocation, 17);



            //Loading a network should be like ...
          //  var n = AA1_MLP.Utilities.ModelManager.LoadNetwork(Properties.Settings.Default.path2LoadModel);


            //Creating a backpropagation trainer
            BackPropagation br = new BackPropagation();

            //Calling the Train method of the trainer with the desired parameters
              var learningCurve = br.Train(n, ds, learningRate: 1, validationSplit: null, numberOfEpochs: 200, shuffle: false, debug: n.Debug, momentum: 0.5, resilient: true, resilientUpdateAccelerationRate: 1.2, resilientUpdateSlowDownRate: 0.5, regularization: AA1_MLP.Enums.Regularizations.None, regularizationRate: 0.01, testData: dt);

            //writing the learning curve data to desk
            // File.WriteAllText(Properties.Settings.Default.LearningCurveLocation, string.Join("\n", learningCurve.Select(s => (s.Length == 2) ? (s[0] + "," + s[1]) : s[0] + "")));


            //saving the trained network to desk
            //AA1_MLP.Utilities.ModelManager.SaveNetowrk(n, path2SaveModel);


            //Testing the model and outputing the confusion matrix
            AA1_MLP.Utilities.ModelManager.Tester(dt, n, 0.5, Properties.Settings.Default.TestReportLocation);



            //computing the ROC plot parameters

            var file = new System.IO.StreamWriter(Properties.Settings.Default.ROCParamsStorageLocation);
            for (float i = 0.005f; i <= 1.0f; i += 0.005f)
            {
                var TprFpr = AA1_MLP.Utilities.ModelManager.Tester(dt, n, i, Properties.Settings.Default.TestReportLocation);
                file.WriteLine(string.Format("{0},{1},{2}", i, TprFpr[0], TprFpr[1]));
            }
            file.Close();


        }



    }
}
