using System;
using System.Configuration;
using AA1_Monks.ModelManager;

namespace AA1_Monks
{
    class Program
    {
        static void Main(string[] args)
        {
            string trainSetLocation = Properties.Settings.Default.trainingDataLocation;
            string testSetLocation = Properties.Settings.Default.testDataLocation;


            var network = ModelManager.NeworkTrainer.TrainNetwork(trainSetLocation, @"model2.mo", @"c:\traininghsitory_" + DateTime.Now.ToString().Replace(":", "-").Replace("/", "_") + ".log", 5000, 0.001, NetworkTrainer.Quick);
          //  var network = ModelManager.NeworkTrainer.LoadNetwork(@"C:\Users\ahmad\Source\Repos\Monks\AA1_Monks\AA1_Monks\bin\Debug\model1.mo");

            ModelManager.NetworkTester.TestNetwork(testSetLocation, network,@"C:\out.log");
        }

    }
}
