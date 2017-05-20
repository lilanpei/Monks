using AA1_MLP.Entities;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text;
using System.Threading.Tasks;

namespace AA1_MLP.Utilities
{
    public sealed class ModelManager
    {
        public static Network LoadNetwork(string networkLocation)
        {
            Network network = null;
            var serializer = new BinaryFormatter();
            using (var s = new FileStream(networkLocation, FileMode.Open))
            {
                network = (Network)serializer.Deserialize(s);
            }
            return network;
        }


        public static void SaveNetowrk(Network n, string path2SaveModel)
        {
            var serializer = new BinaryFormatter();
            using (var s = new FileStream(path2SaveModel, FileMode.Create))
            {
                serializer.Serialize(s, n);
            }
        }
        //#$# TODO: Report the MSE for the test dataset!

        public static List<double[]> TesterRegression(DataSet testSet, Network n, out double MEE)
        {
            List<double[]> predictionsVSActuals = new List<double[]>();
            MEE = 0;
            for (int i = 0; i < testSet.Inputs.RowCount; i++)
            {
                var o = n.ForwardPropagation(testSet.Inputs.Row(i));
                predictionsVSActuals.Add(new double[] { o[0], o[1], testSet.Labels.Row(i)[0], testSet.Labels.Row(i)[1] });
                var loss = ((testSet.Labels.Row(i) - o).PointwiseMultiply(testSet.Labels.Row(i) - o)).Sum();
                MEE += Math.Sqrt(loss);

            }

            MEE /= testSet.Labels.RowCount;
            return predictionsVSActuals;


        }
        public static double[] TesterClassification(DataSet testingSet, Network n, double threshold = 0.5, string reportLocation = "", bool printActualVsIdeal = false)
        {
            double[] TPRateFPRate = new double[2];
            double TP = 0, FP = 0, TN = 0, FN = 0;
            double actualyes = 0;
            double actualNo = 0;
            double predictedYes = 0;
            double predictedNo = 0;
            System.IO.StreamWriter file = null;
            if (!string.IsNullOrWhiteSpace(reportLocation))
            {
                file = new System.IO.StreamWriter(reportLocation);
            }
            for (int i = 0; i < testingSet.Inputs.RowCount; i++)
            {
                var o = n.ForwardPropagation(testingSet.Inputs.Row(i));//network.Compute(pair.Input);
                if ((o[0] >= threshold ? 1 : 0) == 1 && (int)testingSet.Labels.Row(i)[0] == 1)
                {
                    TP++;
                    predictedYes++;
                    actualyes++;
                }
                else if ((o[0] >= threshold ? 1 : 0) == 0 && (int)testingSet.Labels.Row(i)[0] == 0)
                {
                    TN++;
                    predictedNo++;
                    actualNo++;
                }
                else if ((o[0] >= threshold ? 1 : 0) == 1 && (int)testingSet.Labels.Row(i)[0] == 0)
                {
                    FP++;
                    predictedYes++;
                    actualNo++;
                }
                else if ((o[0] >= threshold ? 1 : 0) == 0 && (int)testingSet.Labels.Row(i)[0] == 1)
                {
                    FN++;
                    predictedNo++;
                    actualyes++;
                }
                if (printActualVsIdeal && file != null)
                    file.WriteLine("Actual=" + (o[0] >= threshold ? 1 : 0) + ", Ideal=" + (int)testingSet.Labels.Row(i)[0]);
            }
            TPRateFPRate[0] = TP / actualyes;
            TPRateFPRate[1] = FP / actualNo;
            if (file != null)
            {
                file.WriteLine("Accuracy:" + (TP + TN) / testingSet.Inputs.RowCount);
                file.WriteLine("Misclassification Rate:" + (FP + FN) / testingSet.Inputs.RowCount);
                file.WriteLine("True Positive Rate(Recall):" + TP / actualyes);
                file.WriteLine("False Positive Rate:" + FP / actualNo);
                file.WriteLine("Specificity:" + TN / actualNo);
                file.WriteLine("Precision:" + TP / predictedYes);
                file.WriteLine("Prevalence:" + actualyes / testingSet.Inputs.RowCount);
                file.WriteLine("predicted yes:" + predictedYes);
                file.WriteLine("predicted np:" + predictedNo);
                file.WriteLine("Consusion Matrix:");
                file.WriteLine("True Positive:" + TP);
                file.WriteLine("True Negative:" + TN);
                file.WriteLine("False Positive:" + FP);
                file.WriteLine("False Negative:" + FN);
                file.WriteLine("Actual Yes:" + actualyes);
                file.WriteLine("Actual No:" + actualNo);



            }
            if (file != null)
            {
                file.Close();
            }
            return TPRateFPRate;
        }
    }
}
