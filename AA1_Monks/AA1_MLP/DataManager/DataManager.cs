using AA1_MLP.Entities;
using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AA1_MLP.DataManager
{
    /// <summary>
    /// Loads, parses and encodes the Monks dataset
    /// </summary>
    public class DataManager : IDataManager
    {

        public override DataSet LoadData(string datasetLocation, int featureVectorLength, int outputLength = 1, int? numberOfExamples = null)
        {

            string l;
            List<string> lines = new List<string>();
            System.IO.StreamReader file =
   new System.IO.StreamReader(datasetLocation);
            while ((l = file.ReadLine()) != null)
            {
                if (!string.IsNullOrWhiteSpace(l))
                {
                    lines.Add(l);
                    if (numberOfExamples != null)
                    {
                        numberOfExamples--;
                        if (numberOfExamples == 0)
                        {
                            break;
                        }

                    }

                }

            }

            file.Close();
            // string[] lines = File.ReadAllLines(datasetLocation);
            //  lines = lines.Select(s => !string.IsNullOrWhiteSpace(s));
            Matrix<double> input = CreateMatrix.Dense<double>(lines.Count, featureVectorLength, 0);
            Matrix<double> output = CreateMatrix.Dense<double>(lines.Count, 1, 0);
            for (int i = 0; i < lines.Count; i++)
            {
                var line = lines[i].Trim().Split(' ');


                input.SetRow(i, new double[]{
              line[1] == "1" ? 1 :  0,
              line[1] == "2" ? 1 :  0,
              line[1] == "3" ? 1 :  0,
                                     
              line[2] == "1" ? 1 :  0,
              line[2] == "2" ? 1 :  0,
              line[2] == "3" ? 1 :  0,
                                     
              line[3] == "1" ? 1 :  0,
              line[3] == "2" ? 1 :  0,
                                    
              
              line[4] == "1" ? 1 :  0,
              line[4] == "2" ? 1 :  0,
              line[4] == "3" ? 1 : 0,
                                     
               line[5] == "1" ? 1 : 0,
               line[5] == "2" ? 1 : 0,
               line[5] == "3" ? 1 : 0,
               line[5] == "4" ? 1 : 0,
                                     
               line[6] == "1" ? 1 : 0,
               line[6] == "2" ? 1 : 0
              });

                double[] opt = new double[] { double.Parse(line[0]) };
                output.SetRow(i, opt);
            }

            DataSet trainingSet = new DataSet(input, output);
            return trainingSet;
        }

    }

}
