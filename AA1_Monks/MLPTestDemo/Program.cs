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
                new Layer(new ActivationSigmoid(),true,5),

                new Layer(new ActivationSigmoid(),false,1),
                }, false, AA1_MLP.Enums.WeightsInitialization.Xavier);


            //loading monks Dataset and testing datasets
            DataManager dm = new DataManager();
            DataSet ds = dm.LoadData(Properties.Settings.Default.TrainingSetLocation, 17);
            DataSet dt = dm.LoadData(Properties.Settings.Default.TestSetLocation, 17);


            //Loading a network should be like ...
            //var n = AA1_MLP.Utilities.ModelManager.LoadNetwork(PATH TO THE MODEL BINARY FILE);


            //Creating a backpropagation trainer
            //  var learningCurve = TrainWithSGD(n, ds, dt);


            // creating a linear model and training it with linear regression, need to move the model outside of the trainer!!!!
            var learningCurve = SolveWithLinearRegression(ds, dt);

            //LLS with normal equations solution
            //var tp = new TrainerParams();
            //tp.trainingSet = ds;
            //tp.validationSet = dt;
            //var learningCurve = new LLSNormal().Train(tp);
            //creates an ADAM trainer
            // var learningCurve = TrainWithAdam(n, ds, dt);

            //writing the learning curve trainingdataWithBias to desk (ugly for memory, but simple)
            File.WriteAllText(Properties.Settings.Default.LearningCurveLocation, string.Join("\n", learningCurve.Select(s => string.Join(",", s))));


            //saving the trained network to desk
            //AA1_MLP.Utilities.ModelManager.SaveNetowrk(n, path2SaveModel);


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
            passedParams.learningRate = 0.3;
            passedParams.numberOfEpochs = 500;
            passedParams.shuffle = false;
            passedParams.debug = n.Debug;
            passedParams.nestrov = false;
            passedParams.momentum = 0.9;
            passedParams.resilient = false;
            passedParams.resilientUpdateAccelerationRate = 0.3;
            passedParams.resilientUpdateSlowDownRate = 0.1;
            passedParams.regularization = Regularizations.None;
            passedParams.regularizationRate = 0.001;
            passedParams.validationSet = dt;
            passedParams.batchSize = 7;



            var learningCurve = br.Train(passedParams);
            return learningCurve;
        }

        private static List<double[]> SolveWithLinearRegression(DataSet ds, DataSet dt)
        {
            LLSGradientDescent gd = new LLSGradientDescent();

            //Calling the Train method of the trainer with the desired parameters
            //n, ds, learningRate: .3, numberOfEpochs: 200, shuffle: false, debug: n.Debug, nestrov:false, momentum:0.9, resilient: false, resilientUpdateAccelerationRate: 0.3,
            //resilientUpdateSlowDownRate: 0.1, regularization: AA1_MLP.Enums.Regularizations.L2, regularizationRate: 0.001, validationSet: dt, batchSize: 7
            LinearLeastSquaresParams passedParams = new LinearLeastSquaresParams();
            passedParams.trainingSet = ds;
            passedParams.learningRate = 0.007;
            passedParams.numOfIterations = 3000;
            passedParams.shuffle = false;
            passedParams.debug = false;
            passedParams.regularizationRate = 0.01;
            passedParams.regularizationType = Regularizations.None;
            passedParams.validationSet = dt;
            passedParams.degree = 10;


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
            passedParams.learningRate = 0.01;
            passedParams.numberOfEpochs = 2000;
            passedParams.shuffle = false;
            passedParams.debug = n.Debug;
            passedParams.regularization = Regularizations.None;
            passedParams.regularizationRate = 0.001;
            passedParams.validationSet = dt;
            passedParams.batchSize = 7;




            var learningCurve = br.Train(passedParams);
            return learningCurve;
        }



    }
}
