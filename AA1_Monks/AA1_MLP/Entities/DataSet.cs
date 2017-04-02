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
        public Matrix<double> Input { get; set; }
        public Matrix<double> Output { get; set; }

    }
}
