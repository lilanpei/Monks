using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.Serialization.Formatters.Binary;
using AA1_Monks.Entities;
using Encog.Engine.Network.Activation;
using Encog.ML.Data;
using Encog.Neural.Networks;
using Encog.Neural.Networks.Training;
using Encog.Neural.Networks.Training.Propagation.Back;
using Encog.Neural.Networks.Training.Propagation.Quick;
using Encog.Neural.Networks.Training.Propagation.Resilient;
using Encog.Neural.Networks.Training.Propagation.SCG;

namespace AA1_Monks.ModelManager
{

    enum NetworkTrainer
    {
        BackProp,
        ResilientBackProp,
        Quick,
        ScaledConjugateGradient
    }

    class NeworkTrainer
    {

        public static List<NeuralLayerDescriptor> NetworkTrainingInitializer(string trainSetLocation, out IMLDataSet TrainingSet)
        {
            TrainingSet = DataManager.DataLoader.LoadMonksData(trainSetLocation);

            var descriptor = new List<NeuralLayerDescriptor>()
            {
                new NeuralLayerDescriptor() {NumberOfNeurons = TrainingSet.InputSize},
                new NeuralLayerDescriptor()
                {
                    Activation = new ActivationTANH(),
                    Bias = true,
                    NumberOfNeurons = 2*TrainingSet.InputSize
                },
                new NeuralLayerDescriptor() {Activation = new ActivationSigmoid(), Bias = true, NumberOfNeurons = 1}
            };


            return descriptor;

        }



        public static BasicNetwork LoadNetwork(string networkLocation)
        {
            BasicNetwork network = null;
            var serializer = new BinaryFormatter();
            using (var s = new FileStream(networkLocation, FileMode.Open))
            {
                network = (BasicNetwork)serializer.Deserialize(s);
            }
            return network;
        }

        public static BasicNetwork TrainNetwork(string trainSetLocation, string path2SaveModel, string trainingHistoryLocation, int numberOfEpochs, double errorThreshold, NetworkTrainer trainer)
        {
            IMLDataSet trainingDataSet = null;
            List<NeuralLayerDescriptor> descriptor = NetworkTrainingInitializer(trainSetLocation, out trainingDataSet);
            var network = NetworkCreator.CreateNetwork(descriptor);
            ITrain train = null;
            switch (trainer)
            {
                case NetworkTrainer.BackProp: train = new Backpropagation(network, trainingDataSet);
                    break;
                case NetworkTrainer.ResilientBackProp: train = new ResilientPropagation(network, trainingDataSet);
                    break;
                case NetworkTrainer.Quick:
                    train = new QuickPropagation(network, trainingDataSet);
                    break;
                case NetworkTrainer.ScaledConjugateGradient: train = new ScaledConjugateGradient(network, trainingDataSet);
                    break;
            }


            int epoch = 0;
            using (System.IO.StreamWriter file =
                new System.IO.StreamWriter(trainingHistoryLocation))
            {
                do
                {
                    train.Iteration();
                    Console.WriteLine("Epoch #" + epoch + " Error:" + train.Error);
                    file.WriteLine(train.Error);
                    epoch++;
                } while ((epoch < numberOfEpochs) && (train.Error > errorThreshold));

            }

            var serializer = new BinaryFormatter();
            using (var s = new FileStream(path2SaveModel, FileMode.Create))
            {
                serializer.Serialize(s, network);
            }
            return network;
        }


    }
}
