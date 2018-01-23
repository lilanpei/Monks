
# Hello and Welcome

## Please refer to the detailed report for information about the project, general usage and contents

### Report File CM2017Rerport.pdf

#### here we very briefely shed some light on the major parts of the project and supported files.
The core algorithms implementations are in project AA1_MLP
Folder Entities>> 

NeuralTrainers contains the MLP trainer algorithms implementations
Adam.cs for our implementation of adam and Gradientdescent.cs for our SGD+Momentum implementation


RegressionTrainers contains the LLS solvers implementations

LLSGradientDescent.cs for a very simple gradient descent implementation

LLSNormals.cs for our normal quations solver

LLSSVD.cs for our SVD solver 

Simple examples for reproducing the results are available in the report.

### Supported files
Supported files are the same as the AA1 CUP challenge as input
Output files are CSV files with columns of learning loss, validation loss and the last line is the validation MEE and MSE, to corectly plot a file you output, please feel free to use your plotting software of choice, for us, we wold use python notebooks available in usedFiles/Notebooks
 https://github.com/lilanpei/Monks/tree/master/UsedFiles/Notebooks

 A .n file is a trained model binary serialization saved using the ModelManager static method
 ModelManager.SaveNetowrk
 there is also a ModelManager.LoadNetwork to load a trained model in a .n file

 examples of .n files are present in 
 UsedFiles\FurtherExperiments\60Train40Validation\TrainedModels

 Examples of learning curves csv files are present in 
 UsedFiles\FurtherExperiments\60Train40Validation\LearningCurves

some predicted vs actual comparison for some trained models are present in 
UsedFiles\FurtherExperiments\60Train40Validation\PredictedVsActual

### Update, for reducing verbosity, we now have a flag called PrintLoss in the parameters passed for the MLP adam and sgd trainers, please set it to true in the settings in case you wish to see a realtime update of the training and validation losses

as this section for Example would only be triggered if the parameter is set to True!
 if (passedParams.PrintLoss)
                {
                    Console.ForegroundColor = ConsoleColor.Green;
                    Console.WriteLine("Epoch:{0} train loss:{1} - validation loss:{2}", epoch, epochLoss, validationError);
                }

