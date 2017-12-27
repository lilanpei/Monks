using AA1_MLP.Activations;
using AA1_MLP.DataManager;
using AA1_MLP.Entities;
using AA1_MLP.Entities.Trainers;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using AA1_MLP.Entities.TrainersParams;
using AA1_MLP.Enums;
using AA1_MLP.Entities.Regression;
using AA1_MLP.Entities.RegressionTrainers;
using AA1_MLP.Entities.Linear;
using AA1_MLP.DataManagers;
namespace MLPTestDemo
{
    /// <summary>
    /// a simple demo on how to create a network, load, train and save it
    /// </summary>
    class Program
    {

        static void Main(string[] args)
        {
            //path where the trained model shall be saved in or loaded from
            string path2SaveModel = Properties.Settings.Default.path2SaveModel;


            // Building a simple network
            Network n = new Network(new List<Layer>() {
                new Layer(new ActivationIdentity(),true,17),
                new Layer(new ActivationSigmoid(),true,3),

                new Layer(new ActivationSigmoid(),false,1),
                }, false, AA1_MLP.Enums.WeightsInitialization.Xavier);


            //loading monks Dataset and testing datasets
            MonksDataManager dm = new MonksDataManager();
            DataSet ds = dm.LoadData(Properties.Settings.Default.TrainingSetLocation, 17);
            DataSet dt = dm.LoadData(Properties.Settings.Default.TestSetLocation, 17);


            //loading cup like data

            //CupDataManager dm = new CupDataManager();
            //DataSet ds = dm.LoadData(Properties.Settings.Default.TrainingSetLocation, 1);
            //DataSet dt = dm.LoadData(Properties.Settings.Default.TestSetLocation, 1);


            //Loading a network should be like ...
            //var n = AA1_MLP.Utilities.ModelManager.LoadNetwork(PATH TO THE MODEL BINARY FILE);


            //**Creating a backpropagation trainer
              var learningCurve = TrainWithSGD(n, ds, dt);


            // creating a linear model and training it with linear regression, need to move the model outside of the trainer!!!!
            //LinearModel model = new LinearModel();
            //var learningCurve = SolveWithLinearRegression(ds, dt, model);

            //LLS with normal equations solution
            //var tp = new TrainerParams();

            //**trying SVD
            //LinearLeastSquaresParams passedParams = new LinearLeastSquaresParams { model = model };
            //passedParams.trainingSet = ds;
            //passedParams.validationSet = dt;
            //var learningCurve = new LLSSVD().Train(passedParams);


            //var learningCurve = new LLSNormal().Train(passedParams);

            //creates an ADAM trainer
             //var learningCurve = TrainWithAdam(n, ds, dt);

            //writing the learning curve trainingdataWithBias to desk (ugly for memory, but simple)
            File.WriteAllText(Properties.Settings.Default.LearningCurveLocation, string.Join("\n", learningCurve.Select(s => string.Join(",", s))));


            //saving the trained network to desk
            //AA1_MLP.Utilities.ModelManager.SaveNetowrk(n, path2SaveModel);

            //CAUTION!!!****************$$$$$$$$$$$$$$$$$$$$$-----------------###############
            //CAUTION!!!****************$$$$$$$$$$$$$$$$$$$$$-----------------###############
            //CAUTION!!!****************$$$$$$$$$$$$$$$$$$$$$-----------------###############
            //CAUTION!!!****************$$$$$$$$$$$$$$$$$$$$$-----------------###############                                       
            //This uses only the network for testing, neeeeeed tooooo wrrrriiiitttteeee one for Linear Least Squares uhhhhhhhhhhhhhh
            //Testing the model and outputing the confusion matrix
            AA1_MLP.Utilities.ModelManager.TesterMonkClassification(dt, n, 0.5, Properties.Settings.Default.TestReportLocation);



            //computing the ROC plot parameters

            //var file = new System.IO.StreamWriter(Properties.Settings.Default.ROCParamsStorageLocation);
            //for (float batchIndex = 0.005f; batchIndex <= 1.0f; batchIndex += 0.005f)
            //{
            //    var TprFpr = AA1_MLP.Utilities.ModelManager.TesterMonkClassification(dt, n, batchIndex);
            //    file.WriteLine(string.Format("{0},{1},{2}", batchIndex, TprFpr[0], TprFpr[1]));
            //}
            //file.Close();


        }

        private static List<double[]> TrainWithSGD(Network n, DataSet ds, DataSet dt)
        {
            Gradientdescent br = new Gradientdescent();

            //Calling the Train method of the trainer with the desired parameters
            //n, ds, learningRate: .3, numberOfEpochs: 200, shuffle: false, debug: n.Debug, nestrov:false, momentum:0.9, resilient: false, resilientUpdateAccelerationRate: 0.3,
            //resilientUpdateSlowDownRate: 0.1, regularization: AA1_MLP.Enums.Regularizations.L2, regularizationRate: 0.001, validationSet: dt, batchSize: 7
            GradientDescentParams passedParams = new GradientDescentParams();
            passedParams.network = n;
            passedParams.trainingSet = ds;
            passedParams.learningRate = 0.8;
            passedParams.numberOfEpochs = 100;
            passedParams.shuffle = false;
            passedParams.debug = n.Debug;
            passedParams.nestrov = false;
            passedParams.momentum = 0.7;
            passedParams.resilient = false;
            passedParams.resilientUpdateAccelerationRate = 0.3;
            passedParams.resilientUpdateSlowDownRate = 0.1;
            passedParams.regularization = Regularizations.L2;
            passedParams.regularizationRate = 0.001;
            passedParams.validationSet = dt;
            passedParams.batchSize = null;



            var learningCurve = br.Train(passedParams);
            return learningCurve;
        }

        private static List<double[]> SolveWithLinearRegression(DataSet ds, DataSet dt, LinearModel model)
        {
            LLSGradientDescent gd = new LLSGradientDescent();

            //Calling the Train method of the trainer with the desired parameters
            //n, ds, learningRate: .3, numberOfEpochs: 200, shuffle: false, debug: n.Debug, nestrov:false, momentum:0.9, resilient: false, resilientUpdateAccelerationRate: 0.3,
            //resilientUpdateSlowDownRate: 0.1, regularization: AA1_MLP.Enums.Regularizations.L2, regularizationRate: 0.001, validationSet: dt, batchSize: 7


            LinearLeastSquaresParams passedParams = new LinearLeastSquaresParams { model = model };
            passedParams.trainingSet = ds;
            passedParams.learningRate = 1;
            passedParams.numOfIterations = 3000;
            passedParams.shuffle = false;
            passedParams.debug = false;
            passedParams.regularizationRate = 0.01;
            passedParams.regularizationType = Regularizations.None;
            passedParams.validationSet = dt;
            passedParams.degree = 30;


            var learningCurve = gd.Train(passedParams);
            return learningCurve;
        }



        private static List<double[]> TrainWithAdam(Network n, DataSet ds, DataSet dt)
        {
            Adam br = new Adam();

            //Calling the Train method of the trainer with the desired parameters
            //n, ds, learningRate: .3, numberOfEpochs: 200, shuffle: false, debug: n.Debug, nestrov:false, momentum:0.9, resilient: false, resilientUpdateAccelerationRate: 0.3,
            //resilientUpdateSlowDownRate: 0.1, regularization: AA1_MLP.Enums.Regularizations.L2, regularizationRate: 0.001, validationSet: dt, batchSize: 7
            AdamParams passedParams = new AdamParams();
            passedParams.network = n;
            passedParams.trainingSet = ds;
            passedParams.learningRate = 0.09;
            passedParams.numberOfEpochs = 200;
            passedParams.shuffle = true;
            passedParams.debug = n.Debug;
            passedParams.regularization = Regularizations.None;
            passedParams.regularizationRate = 0.0001;
            passedParams.validationSet = dt;
            passedParams.batchSize = 7;




            var learningCurve = br.Train(passedParams);
            return learningCurve;
        }



    }
}
