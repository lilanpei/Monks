using AA1_MLP.Entities;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AA1_MLP.DataManager
{
    /// <summary>
    /// Class the loads the dataset
    /// </summary>
    public abstract class IDataManager
    {
        /// <summary>
        /// Loads parses and returns the dataset, any loader has to provide the implementation for this class
        /// </summary>
        /// <param name="datasetLocation"></param>
        /// <param name="featureVectorLength"></param>
        /// <param name="outputLength"></param>
        /// <param name="numberOfExamples"></param>
        /// <returns></returns>
        abstract public DataSet LoadData(string datasetLocation, int featureVectorLength, int outputLength = 1, int? numberOfExamples = null, bool reportOsutput = true);
    }
}
