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
    public  sealed class ModelManager
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

        public static void Tester(DataSet testingSet, Network n)
        {
            {
                double TP = 0, FP = 0, TN = 0, FN = 0;
                double actualyes = 0;
                double actualNo = 0;
                double predictedYes = 0;
                double predictedNo = 0;
                double threshold = 0.5;
                using (System.IO.StreamWriter file =
                    new System.IO.StreamWriter("error.txt"))
                {

                    for (int i = 0; i < testingSet.Inputs.RowCount; i++)
                    {
                        var o = n.ForwardPropagation(testingSet.Inputs.Row(i));//network.Compute(pair.Input);


                        if ((int)testingSet.Labels.Row(i)[0] == 1)
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
                        else predictedNo++;

                        if ((o[0] >= threshold ? 1 : 0) == 1 && (int)testingSet.Labels.Row(i)[0] == 1)
                        {
                            TP++;
                        }
                        else if ((o[0] >= threshold ? 1 : 0) == 0 && (int)testingSet.Labels.Row(i)[0] == 0)
                        {
                            TN++;
                        }

                        else if ((o[0] >= threshold ? 1 : 0) == 1 && (int)testingSet.Labels.Row(i)[0] == 0)
                        {
                            FP++;
                        }
                        else if ((o[0] >= threshold ? 1 : 0) == 0 && (int)testingSet.Labels.Row(i)[0] == 1)
                        {
                            FN++;
                        }
                        file.WriteLine("Actual=" + (o[0] >= threshold ? 1 : 0) + ", Ideal=" + (int)testingSet.Labels.Row(i)[0]);

                    }
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







                }
            }
        }
    }
}
