using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AA1_MLP.Entities.Linear
{
    public class LinearModel : IModel
    {

        public Matrix<double> Weights { get; set; }
        public int Degree { get; set; }
        public bool bias { get; set; }
        public override MathNet.Numerics.LinearAlgebra.Vector<double> Predict(MathNet.Numerics.LinearAlgebra.Vector<double> input)
        {
            //Ax where A is the input and x are the weights

            if (Degree > 1)
            {
                double[] row = new double[input.Count * Degree];

                for (int j = 0; j < input.Count; j++)
                {
                    row[j] = input[j];
                    for (int k = 1; k < Degree; k++)
                    {
                        row[k * input.Count + j] = Math.Pow(row[j], k + 1);
                    }
                }
                input = CreateVector.Dense<double>(row);

            }

            if (bias)
            {
                List<double> data = input.ToList();
                data.Insert(0, 1.0);
                input = CreateVector.Dense<double>(data.ToArray<double>());
            }
            Vector<double> output = CreateVector.Dense<double>(new double[] { input.DotProduct(Weights.Column(0)) });
            return output;

        }
    }
}
