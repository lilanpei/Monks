using AA1_MLP.Entities;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AA1_MLP.DataManager
{
    public abstract class IDataManager
    {
        abstract public DataSet LoadData(string datasetLocation, int featureVectorLength, int outputLength = 1, int? numberOfExamples = null);
    }
}
