using AA1_MLP.Activations;
using AA1_MLP.Entities;
using AA1_MLP.Entities.Trainers;
using MathNet.Numerics.LinearAlgebra;
using System.Collections.Generic;
using System.IO;
using System.Linq;
namespace AA1_MLP
{
    class Program
    {
        static void Main(string[] args)
        {
            Network n = new Network(new List<Layer>() {

            new Layer(new ActivationIdentity(),true,17),
            new Layer(new ActivationSigmoid(),true,2*17),
            new Layer(new ActivationSigmoid(),false,1),
            }, false);




            DataSet ds = DataManager.DataManager.LoadMonksData(Properties.Settings.Default.TrainingSetLocation, 17);
            /*   DataSet ds = new DataSet(
                   CreateMatrix.Dense(4, 2, new double[] { 1, 0, 1, 0, 1, 1, 0, 0 }),
                   CreateMatrix.Dense(4, 1, new double[] { 0, 1, 1, 0 }));*/
            /*
                        DataSet ds = new DataSet(
                            CreateMatrix.Dense(1, 2, new double[] { 0.05, 0.1 }),
                            CreateMatrix.Dense(1, 2, new double[] { 0.01, 0.99 })

                            );*/
            for (int i = 0; i < ds.Inputs.RowCount; i++)
            {
                System.Console.WriteLine("Input");
                System.Console.WriteLine(ds.Inputs.Row(i));


                var result = n.ForwardPropagation(ds.Inputs.Row(i));
                System.Console.WriteLine("Final output");
                System.Console.WriteLine(result);
            }

            BackPropagation br = new BackPropagation();
            var learningCurve = br.Train(n, ds, learningRate: 0.5, numberOfEpochs: 10000, debug: n.Debug, momentum: 0.9);

            File.WriteAllText("learningcurve.txt", string.Join("\n", learningCurve.Select(s => (s.Length == 2) ? (s[0] + "," + s[1]) : s[0] + "")));


            System.Console.WriteLine("~~~~~~~~~~Printing Results:~~~~~~~~~~");

            for (int i = 0; i < ds.Inputs.RowCount; i++)
            {
                System.Console.WriteLine("Input");
                System.Console.WriteLine(ds.Inputs.Row(i));
                System.Console.WriteLine("Target");
                System.Console.WriteLine(ds.Labels.Row(i));
                checked
                {
                    var result = n.ForwardPropagation(ds.Inputs.Row(i));
                    
                    System.Console.WriteLine("Final output");
                    System.Console.WriteLine(result);
                }
                System.Console.WriteLine("Target");
                System.Console.WriteLine(ds.Labels.Row(i));

            }
            Tester(ds, n);


        }

        static void Tester(DataSet testingSet, Network n)
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
                        else predictedNo--;

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
