using AA1_MLP.Activations;
using AA1_MLP.Entities;
using AA1_MLP.Entities.Trainers;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AA1_CUP
{
    public class Program
    {
        static void Main(string[] args)
        {
            // Building a simple network
            Network n = new Network(new List<Layer>() {
                new Layer(new ActivationIdentity(),true,10),
                new Layer(new ActivationSigmoid(),true,20),
                new Layer(new ActivationSigmoid(),true,20),

                new Layer(new ActivationSigmoid(),false,2),
                }, new MathNet.Numerics.Distributions.Normal(-0.7, 0.7), true, false);



            CupDataManager dm = new CupDataManager();
            DataSet wholeSet = dm.LoadData(Properties.Settings.Default.TrainingSetLocation, 10, 2);

            int trainSplit = (int)(0.7 * wholeSet.Inputs.RowCount);
            DataSet trainingSplit = new DataSet(
               inputs: wholeSet.Inputs.SubMatrix(0, trainSplit, 0, wholeSet.Inputs.ColumnCount),
          labels: wholeSet.Labels.SubMatrix(0, trainSplit, 0, wholeSet.Labels.ColumnCount));

            DataSet ValidationSplit = new DataSet(
              inputs: wholeSet.Inputs.SubMatrix(trainSplit, wholeSet.Inputs.RowCount - trainSplit, 0, wholeSet.Inputs.ColumnCount),
         labels: wholeSet.Labels.SubMatrix(trainSplit, wholeSet.Inputs.RowCount - trainSplit, 0, wholeSet.Labels.ColumnCount));



            BackPropagation bp = new BackPropagation();

            var learningCurve = bp.Train(n,
                     trainingSplit,
                     0.01,
                     500,
                     true,
                     regularizationRate: 0.1,
                     regularization: AA1_MLP.Enums.Regularizations.L2,
                     momentum: 0.9,
                     validationSet: ValidationSplit,
                     MEE: true
                    // resilient:true,resilientUpdateAccelerationRate:1.5,resilientUpdateSlowDownRate:0.5
                     );

            //writing the learning curve data to desk (ugly for memory, but simple)
            File.WriteAllText(Properties.Settings.Default.LearningCurveLocation, string.Join("\n", learningCurve.Select(s => string.Join(",", s))));


        }
    }
}
