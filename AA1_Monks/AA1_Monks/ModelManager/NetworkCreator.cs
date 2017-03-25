using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using AA1_Monks.Entities;
using Encog.Neural.Networks;
using Encog.Neural.Networks.Layers;

namespace AA1_Monks.ModelManager
{
    class NetworkCreator
    {
        public static BasicNetwork CreateNetwork(List<NeuralLayerDescriptor> descritor)
        {
            BasicNetwork network = new BasicNetwork();
            network.AddLayer(new BasicLayer(descritor[0].NumberOfNeurons));

            for (int i = 1; i < descritor.Count; i++)
            {
                network.AddLayer(new BasicLayer(descritor[i].Activation, descritor[i].Bias, descritor[i].NumberOfNeurons));
            }
            network.Structure.FinalizeStructure();
            network.Reset();
            return network;
        }

    }
}
