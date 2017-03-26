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


            //var network = ModelManager.NeworkTrainer.TrainNetwork(trainSetLocation, @"model2.mo", @"c:\traininghsitory_" + DateTime.Now.ToString().Replace(":", "-").Replace("/", "_") + ".log", 5000, 0.001, NetworkTrainer.Quick);
            var network = ModelManager.NeworkTrainer.LoadNetwork(@"E:\Dropbox\Master Course\SEM-3\ML\Monk\AA1_Monks\AA1_Monks\bin\Debug\model2.mo");

            for (double i = 0; i <=  1; i += 0.05)
            {
                ModelManager.NetworkTester.PlotROC(testSetLocation, network, @"ROC.log", i);
            }           
        }
    }
}
