using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using Encog.Engine.Network.Activation;
using Encog.Neural.Networks;
using Encog.Neural.Networks.Layers;
using Encog.Neural.Networks.Training;
using Encog.Neural.Networks.Training.Propagation.Resilient;
using Encog.ML;
using Encog.Neural.Data.Basic;
using Encog.Neural.NeuralData;

namespace AA1_Monks
{
    class Program
    {
        static void Main(string[] args)
        {
            string trainSetLocation = @"dataset1train";
            string testSetLocation = @"dataset1test";
            var trainingSet = LoadData(trainSetLocation);
            var testingSet = LoadData(testSetLocation);
            Console.WriteLine("Hello Ahmad");
        }

        private static BasicNeuralDataSet LoadData(string datasetLocation)
        {
            string[] lines = File.ReadAllLines(datasetLocation);
            double[][] input = new double[lines.Length][];
            double[][] output = new double[lines.Length][];
            for (int i = 0; i < lines.Length; i++)
            {
                var line = lines[i].Trim().Split(' ');


                input[i] = new double[17];
                input[i][0] = line[1] == "1" ? 1 : 0;
                input[i][1] = line[1] == "2" ? 1 : 0;
                input[i][2] = line[1] == "3" ? 1 : 0;

                input[i][3] = line[2] == "1" ? 1 : 0;
                input[i][4] = line[2] == "2" ? 1 : 0;
                input[i][5] = line[2] == "3" ? 1 : 0;

                input[i][6] = line[3] == "1" ? 1 : 0;
                input[i][7] = line[3] == "2" ? 1 : 0;


                input[i][8] = line[4] == "1" ? 1 : 0;
                input[i][9] = line[4] == "2" ? 1 : 0;
                input[i][10] = line[4] == "3" ? 1 : 0;

                input[i][11] = line[5] == "1" ? 1 : 0;
                input[i][12] = line[5] == "2" ? 1 : 0;
                input[i][13] = line[5] == "3" ? 1 : 0;
                input[i][14] = line[5] == "4" ? 1 : 0;

                input[i][15] = line[6] == "1" ? 1 : 0;
                input[i][16] = line[6] == "2" ? 1 : 0;


                output[i] = new double[] { double.Parse((line[0])) };
            }

            BasicNeuralDataSet trainingSet = new BasicNeuralDataSet(input, output);
            return trainingSet;
        }
    }
}
