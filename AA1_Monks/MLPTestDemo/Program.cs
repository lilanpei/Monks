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
using AA1_MLP.Utilities;
namespace MLPTestDemo
{
    /// <summary>
    /// a simple demo on how to create a network, load, train and save it
    /// </summary>
    class Program
    {

        static void Main(string[] args)
        {
            // MonksTesting();


            //TrainAndPRoduceFinalResult();
            //  CupTestingLLS("D:\\dropbox\\Dropbox\\Master Course\\SEM-3\\ML\\CM_CUP_Datasets\\60percenttrain.txt", "D:\\dropbox\\Dropbox\\Master Course\\SEM-3\\ML\\CM_CUP_Datasets\\60percenttest.txt");

            //proving non-convexity:

            //  CheckConvexity();

            /*
                Solution:
                1- randomly pick a weight to play with [maybe start with the hidden layer? it should be more fun :D?]
             */

            /* 
               2- for that weight, change its value in a range [-1 to 1] maybe? or perhaps we define a range based on the weights values in the whole layer or maybe better, a range around the weights value?
               3- for each value change for the picked weight, computer the corresponding cost function
               4- enjoy the plots and redo for more weights
            */
        }

        private static void CheckConvexity()
        {
            AA1_MLP.DataManagers.CupDataManager dm = new AA1_MLP.DataManagers.CupDataManager();
            DataSet trainDS = dm.LoadData("D:\\dropbox\\Dropbox\\Master Course\\SEM-3\\ML\\CM_CUP_Datasets\\60percenttrain.txt", 10, 2, standardize: true);

            var nOriginal = //ModelManager.LoadNetwork(@"C:\Users\ahmad\Documents\monks\Monks\5kitr_mo0.5_100_final_sgdnestrov_hdn100_lr0.001_reg0.001.n");
            new Network(new List<Layer>() {
                     new Layer(new ActivationIdentity(),true,10),
                     new Layer(new ActivationTanh(),true,100),
                  //   new Layer(new ActivationLeakyRelu(),true,40),


                     new Layer(new ActivationIdentity(),false,2),
                     }, false, AA1_MLP.Enums.WeightsInitialization.Xavier);

            for (int i = 0; i < 100; i++)
            {
                Console.WriteLine(i);

                List<double[]> weightValVsCost = GeneratePlot(trainDS, nOriginal, -10, 10, 0.01);

                File.WriteAllLines(@"xcurve" + i + ".txt", weightValVsCost.OrderBy(s => s[0]).Select(x => string.Join(",", x)).ToArray());
            }
            Console.WriteLine();

        }

        private static List<double[]> GeneratePlot(DataSet trainDS, Network nOriginal, double start, double end, double step)
        {


            //building the architecture
            Network n = new Network(new List<Layer>() {
                     new Layer(new ActivationIdentity(),true,10),
                     new Layer(new ActivationTanh(),true,100),
                  //   new Layer(new ActivationLeakyRelu(),true,40),


                     new Layer(new ActivationIdentity(),false,2),
                     }, false, AA1_MLP.Enums.WeightsInitialization.Xavier);


            Network nRand = new Network(new List<Layer>() {
                     new Layer(new ActivationIdentity(),true,10),
                     new Layer(new ActivationTanh(),true,100),
                  //   new Layer(new ActivationLeakyRelu(),true,40),


                     new Layer(new ActivationIdentity(),false,2),
                     }, false, AA1_MLP.Enums.WeightsInitialization.Normal);





            double MSE = 0, MEE = 0;
            List<double[]> weightValVsCost = new List<double[]>();
            for (double i = start; i <= end; i += step)
            {
                for (int layerIdx = 0; layerIdx < n.Weights.Count; layerIdx++)
                {
                    n.Weights[layerIdx] = nOriginal.Weights[layerIdx] + (i * nRand.Weights[layerIdx]);

                }



                var log = ModelManager.TesterCUPRegression(trainDS, n, out MEE, out  MSE);
                weightValVsCost.Add(new double[] { i, MSE });
            }
            return weightValVsCost;
        }
        /// <summary>
        /// For outputting the final cup results
        /// </summary>
        private static void TrainAndPRoduceFinalResult()
        {
            AA1_MLP.DataManagers.CupDataManager dm = new AA1_MLP.DataManagers.CupDataManager();
            DataSet trainDS = dm.LoadData(@"D:\dropbox\Dropbox\Master Course\SEM-3\ML\CM_CUP_Datasets\ML-17-PRJ lecture  package-20171225\ML-CUP17-TR.csv", 10, 2, skip: 1, standardize: true);
            DataSet FinalTestDS = dm.LoadData(@"D:\dropbox\Dropbox\Master Course\SEM-3\ML\CM_CUP_Datasets\ML-17-PRJ lecture  package-20171225\ML-CUP17-TS.csv", 10, skip: 1, reportOsutput: false, standardize: true);




            /*AdamParams passedParams = new AdamParams();
            IOptimizer trainer = new Adam();*/
            GradientDescentParams passedParams = new GradientDescentParams();
            Gradientdescent trainer = new Gradientdescent();
            passedParams.numberOfEpochs = 5000;
            passedParams.batchSize = 10;
            passedParams.trainingSet = trainDS;
            passedParams.learningRate = 0.001;
            passedParams.regularization = Regularizations.L2;
            passedParams.regularizationRate = 0.001;
            passedParams.nestrov = true;
            passedParams.resilient = false;
            passedParams.resilientUpdateAccelerationRate = 2;
            passedParams.resilientUpdateSlowDownRate = 0.5;

            passedParams.momentum = 0.5;
            passedParams.NumberOfHiddenUnits = 100;
            passedParams.trueThreshold = null;

            string path = "cupTrain" + passedParams.NumberOfHiddenUnits + "_lr" + passedParams.learningRate + "_reg" + passedParams.regularizationRate;
            //building the architecture
            Network n = new Network(new List<Layer>() {
                     new Layer(new ActivationIdentity(),true,10),
                     new Layer(new ActivationTanh(),true,passedParams.NumberOfHiddenUnits),
                  //   new Layer(new ActivationLeakyRelu(),true,40),


                     new Layer(new ActivationIdentity(),false,2),
                     }, false, AA1_MLP.Enums.WeightsInitialization.Xavier);
            passedParams.network = n;
            var watch = System.Diagnostics.Stopwatch.StartNew();
            List<double[]> learningCurve = trainer.Train(passedParams);
            watch.Stop();
            var elapsedMs = watch.ElapsedMilliseconds;
            Console.WriteLine("elapsed Time:{0} ms", elapsedMs);



            File.WriteAllText(path + ".txt", string.Join("\n", learningCurve.Select(s => string.Join(",", s))));


            ModelManager.SaveNetowrk(n, path + ".n");

            var predictions = ModelManager.GeneratorCUP(FinalTestDS, n);
            File.WriteAllText("OMG_LOC-OSM2-TS.txt", string.Join("\n", predictions.Select(s => string.Join(",", s))));
        }
        /// <summary>
        /// For outputting the LLS solution results
        /// </summary>
        /// <param name="trainsetpath"></param>
        /// <param name="testsetpath"></param>
        private static void CupTestingLLS(string trainsetpath, string testsetpath)
        {
            CupDataManager dm = new CupDataManager();
            DataSet trainset = dm.LoadData(trainsetpath, 10, 2, standardize: true);
            DataSet testset = dm.LoadData(testsetpath, 10, 2, standardize: true);



            LinearModel model = new LinearModel { bias = true };

            //**trying SVD
            LinearLeastSquaresParams passedParams = new LinearLeastSquaresParams { model = model, numOfIterations = 5000, learningRate = 0.1, degree = 1 };
            passedParams.trainingSet = trainset;
            passedParams.validationSet = testset;
            Console.WriteLine("SVD solution:");
            var svdlearningCurve = new LLSSVD().Train(passedParams);
            Console.WriteLine("Normal Equations Solution:");
            var normallearningCurve = new LLSNormal().Train(passedParams);
            Console.WriteLine("GD  Solution:");
            var gdlearningCurve = new LLSGradientDescent().Train(passedParams);

        }

        private static void MonksTesting()
        {
            //path where the trained model shall be saved in or loaded from
            string path2SaveModel = Properties.Settings.Default.path2SaveModel;


            // Building a simple network
            Network n = new Network(new List<Layer>() {
                new Layer(new ActivationIdentity(),true,17),
                new Layer(new ActivationSigmoid(),true,3),

                new Layer(new ActivationSigmoid(),false,1),
                }, false, AA1_MLP.Enums.WeightsInitialization.He);


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
            //var learningCurve = TrainWithSGD(n, ds, dt);


            // creating a linear model and training it with linear regression, need to move the model outside of the trainer!!!!
            LinearModel model = new LinearModel();
            var learningCurve = SolveWithLinearRegression(ds, dt, model);

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
            AA1_MLP.Utilities.ModelManager.TesterMonkClassification(dt, model, 0.5, Properties.Settings.Default.TestReportLocation);



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
