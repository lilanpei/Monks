using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CustomExtensionMethods;
namespace AA1_MLP.Entities
{
    class BackPropagation : IOptimizer
    {
        public override void Iterate(Network network, DataSet wholeData, int numberOfEpochs, bool shuffle = false, int? batchSize = null, float? validationSplit = null, IOptimizer.Historian historian = null, IOptimizer.CheckPointer checkPointer = null)
        {
            List<int> indices = Enumerable.Range(0, wholeData.Labels.RowCount - 1).ToList();
            if (shuffle)
            {
                indices.Shuffle();
            }

            DataSet validation = new DataSet();
            if (validationSplit != null)
            {
                for (int i = 0; i < indices.Count * validationSplit; i++)
                {
                    validation.Inputs.Append(wholeData.Inputs.SubMatrix(indices[i], 1, 0, wholeData.Inputs.ColumnCount - 1));
                    validation.Labels.Append(wholeData.Labels.SubMatrix(indices[i], 1, 0, wholeData.Labels.ColumnCount - 1));

                }
                for (int i = (int)( indices.Count * validationSplit); i < indices.Count; i++)
                {
                    wholeData.Inputs.Append(wholeData.Inputs.SubMatrix(indices[i], 1, 0, wholeData.Inputs.ColumnCount - 1));
                    wholeData.Labels.Append(wholeData.Labels.SubMatrix(indices[i], 1, 0, wholeData.Labels.ColumnCount - 1));

                }

            }

            for (int i = 0; i < numberOfEpochs; i++)
            {
                if (batchSize != null)
                {

                }

            }
        }



    }
}
