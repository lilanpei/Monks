using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AA1_MLP.Entities
{
   public class DataSet
    {
        public Matrix<double> Inputs { get; set; }
        public Matrix<double> Labels { get; set; }

        public DataSet(Matrix<double>inputs,Matrix<double>labels)
        {
            Inputs = inputs;
            Labels = labels;
        }
      
    }
}
