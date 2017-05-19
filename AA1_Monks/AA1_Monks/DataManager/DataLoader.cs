using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Encog.Neural.Data.Basic;

namespace AA1_Monks.DataManager
{
    public class DataLoader
    {
       
        public static BasicNeuralDataSet LoadMonksData(string datasetLocation)
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

        public static BasicNeuralDataSet LoadData(string datasetLocation)
        {
            string[] lines = File.ReadAllLines(datasetLocation).Where(s=>!string.IsNullOrEmpty(s)&&!s.StartsWith("#")).ToArray();
            double[][] input = new double[lines.Length][];
            double[][] output = new double[lines.Length][];
            for (int i = 0; i < lines.Length; i++)
            {
                var line = lines[i].Trim().Split(',');
                input[i] = new double[10];

                for (int x = 1; x <=10; x++)
                {
                    input[i][x-1] = double.Parse( line[x]);
                    
                }

                output[i] = new double[] { double.Parse((line[11])), double.Parse(line[12]) };
            }

            BasicNeuralDataSet trainingSet = new BasicNeuralDataSet(input, output);
            return trainingSet;
        }

    }
}
