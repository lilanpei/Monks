using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AA1_MLP.Entities
{
    /// <summary>
    /// This class represents a dataset object, specifying the inputs and the corresponding labels/targets
    /// </summary>
   public class DataSet
    {
        public Matrix<double> Inputs { get; set; }
        public Matrix<double> Labels { get; set; }

       /// <summary>
       /// Default constructor loading the inputs and the corresponding labels
       /// </summary>
       /// <param name="inputs">a matrix of doubles representing the inputs</param>
       /// <param name="labels">a matrix of doubles represting the targets</param>
        public DataSet(Matrix<double>inputs,Matrix<double>labels)
        {
            Inputs = inputs;
            Labels = labels;
        }
      
    }
}
