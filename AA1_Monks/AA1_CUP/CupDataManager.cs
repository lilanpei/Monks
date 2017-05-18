﻿using AA1_MLP.DataManager;
using AA1_MLP.Entities;
using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AA1_CUP
{
    class CupDataManager : IDataManager
    {
        public override AA1_MLP.Entities.DataSet LoadData(string datasetLocation, int featureVectorLength, int outputLength = 1, int? numberOfExamples = null)
        {

            string l;
            System.IO.StreamReader file =
   new System.IO.StreamReader(datasetLocation);
            Matrix<double> input = CreateMatrix.Dense<double>(1, featureVectorLength, 0);
            Matrix<double> output = CreateMatrix.Dense<double>(1, outputLength, 0);
            int i = 0;
            while ((l = file.ReadLine()) != null)
            {

                if (!string.IsNullOrWhiteSpace(l) && !l.StartsWith("#"))
                {
                    
                    var line = l.Split(',');
                    if (i == 0)
                    {
                        input.SetRow(i, line.Skip(1).Take(10).Select(s => double.Parse(s, CultureInfo.InvariantCulture)).ToArray());
                        output.SetRow(i, line.Skip(11).Select(s => double.Parse(s, CultureInfo.InvariantCulture)).ToArray());

                    }
                    else
                    {
                        input = input.InsertRow(i, CreateVector.Dense(line.Skip(1).Take(10).Select(s => double.Parse(s)).ToArray()));
                        output = output.InsertRow(i, CreateVector.Dense(line.Skip(11).Select(s => double.Parse(s)).ToArray()));
                    }
                    i++;
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



            DataSet trainingSet = new DataSet(input, output);
            return trainingSet;
        }
    }
}
