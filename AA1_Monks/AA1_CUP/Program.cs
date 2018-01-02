using AA1_MLP.Activations;
using AA1_MLP.Entities;
using AA1_MLP.Entities.Trainers;
using AA1_MLP.Utilities;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AA1_CUP
{
    /// <summary>
    /// Performing an automated grid search for hyperparameters for the model for the Cup problem
    /// </summary>
    public class Program
    {
        static void Main(string[] args)
        {
            //Loading and parsing cup dataset
            CupDataManager dm = new CupDataManager();
            DataSet wholeSet = dm.LoadData(Properties.Settings.Default.TrainingSetLocation, 10, 2, permute: true, seed: 1);
            new KFoldValidation().Screen(wholeSet, 5);


            //Loading and parsing cup dataset

            //  CupDataManager dm = new CupDataManager();
            //Loading the test dataset
            //DataSet TestSet = dm.LoadData(Properties.Settings.Default.TestSetLocation, 10, reportOsutput: false);
            //Loading the trained model
            //var n = AA1_MLP.Utilities.ModelManager.LoadNetwork("Final_hidn18_reg0.01_mo0.5_lr9E-06_model.AA1");

            //double MEE = 0;
            //applying the model on the test data
            //var predictions = ModelManager.GeneratorCUP(TestSet, n);
            //writing the results
            // File.WriteAllText("OMG_LOC-OSM2-TS.txt", string.Join("\n", predictions.Select(s => string.Join(",", s))));



        }
    }
}
