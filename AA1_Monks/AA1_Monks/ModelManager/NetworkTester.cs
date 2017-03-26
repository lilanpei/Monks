using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Encog.Neural.Networks;

namespace AA1_Monks.ModelManager
{
    class NetworkTester
    {
        public static void PlotROC(string testSetLocation, BasicNetwork network, string errorLogLocation, double threshold)
        {
            double TP = 0, FP = 0, TN = 0, FN = 0;
            double actualyes = 0;
            double actualNo = 0;
            double predictedYes = 0;
            double predictedNo = 0;
            var testingSet = DataManager.DataLoader.LoadMonksData(testSetLocation);
            using (System.IO.StreamWriter file =
                new System.IO.StreamWriter(errorLogLocation,true))
            {

                foreach (var pair in testingSet)
                {
                    var o = network.Compute(pair.Input);


                    if ((int)pair.Ideal[0] == 1)
                    {
                        actualyes++;
                    }
                    else
                    {
                        actualNo++;
                    }

                    if ((o[0] >= threshold ? 1 : 0) == 1)
                    {
                        predictedYes++;
                    }
                    else predictedNo--;

                    if ((o[0] >= threshold ? 1 : 0) == 1 && (int)pair.Ideal[0] == 1)
                    {
                        TP++;
                    }
                    else if ((o[0] >= threshold ? 1 : 0) == 0 && (int)pair.Ideal[0] == 0)
                    {
                        TN++;
                    }

                    else if ((o[0] >= threshold ? 1 : 0) == 1 && (int)pair.Ideal[0] == 0)
                    {
                        FP++;
                    }
                    else if ((o[0] >= threshold ? 1 : 0) == 0 && (int)pair.Ideal[0] == 1)
                    {
                        FN++;
                    }
                    //file.WriteLine("Actual=" + (o[0] >= threshold ? 1 : 0) + ", Ideal=" + pair.Ideal[0]);

                }
                //file.WriteLine("Accuracy:" + (TP + TN) / testingSet.Count);
                //file.WriteLine("Misclassification Rate:" + (FP + FN) / testingSet.Count);
                //file.WriteLine("True Positive Rate(Recall):" + TP / actualyes);
                //file.WriteLine("False Positive Rate:" + FP / actualNo);
                file.WriteLine(1 - TN / actualNo + "," + TP / actualyes);
                //file.WriteLine("Precision:" + TP / predictedYes);
                //file.WriteLine("Prevalence:" + actualyes / testingSet.Count);

                //file.WriteLine("predicted yes:" + predictedYes);
                //file.WriteLine("predicted np:" + predictedNo);

                //file.WriteLine("Consusion Matrix:");

                //file.WriteLine("True Positive:" + TP);
                //file.WriteLine("True Negative:" + TN);
                //file.WriteLine("False Positive:" + FP);
                //file.WriteLine("False Negative:" + FN);
            }
        }
        public static void TestNetwork(string testSetLocation, BasicNetwork network, string errorLogLocation)
        {
            double TP = 0, FP = 0, TN = 0, FN = 0;
            double actualyes = 0;
            double actualNo = 0;
            double predictedYes = 0;
            double predictedNo = 0;
            double threshold = 0.5;
            var testingSet = DataManager.DataLoader.LoadMonksData(testSetLocation);
            using (System.IO.StreamWriter file =
                new System.IO.StreamWriter(errorLogLocation))
            {

                foreach (var pair in testingSet)
                {
                    var o = network.Compute(pair.Input);


                    if ((int)pair.Ideal[0] == 1)
                    {
                        actualyes++;
                    }
                    else
                    {
                        actualNo++;
                    }

                    if ((o[0] >= threshold ? 1 : 0) == 1)
                    {
                        predictedYes++;
                    }
                    else predictedNo--;

                    if ((o[0] >= threshold ? 1 : 0) == 1 && (int)pair.Ideal[0] == 1)
                    {
                        TP++;
                    }
                    else if ((o[0] >= threshold ? 1 : 0) == 0 && (int)pair.Ideal[0] == 0)
                    {
                        TN++;
                    }

                    else if ((o[0] >= threshold ? 1 : 0) == 1 && (int)pair.Ideal[0] == 0)
                    {
                        FP++;
                    }
                    else if ((o[0] >= threshold ? 1 : 0) == 0 && (int)pair.Ideal[0] == 1)
                    {
                        FN++;
                    }
                    file.WriteLine("Actual=" + (o[0] >= threshold ? 1 : 0) + ", Ideal=" + pair.Ideal[0]);

                }
                file.WriteLine("Accuracy:" + (TP + TN) / testingSet.Count);
                file.WriteLine("Misclassification Rate:" + (FP + FN) / testingSet.Count);
                file.WriteLine("True Positive Rate(Recall):" + TP / actualyes);
                file.WriteLine("False Positive Rate:" + FP / actualNo);
                file.WriteLine("Specificity:" + TN / actualNo);
                file.WriteLine("Precision:" + TP / predictedYes);
                file.WriteLine("Prevalence:" + actualyes / testingSet.Count);

                file.WriteLine("predicted yes:" + predictedYes);
                file.WriteLine("predicted np:" + predictedNo);

                file.WriteLine("Consusion Matrix:");

                file.WriteLine("True Positive:" + TP);
                file.WriteLine("True Negative:" + TN);
                file.WriteLine("False Positive:" + FP);
                file.WriteLine("False Negative:" + FN);







            }
        }

    }
}
