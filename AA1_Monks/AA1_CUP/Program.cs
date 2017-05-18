using AA1_MLP.Activations;
using AA1_MLP.Entities;
using System;
using System.Collections.Generic;
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
                new Layer(new ActivationSigmoid(),false,2),
                }, new MathNet.Numerics.Distributions.Normal(-0.7, 0.7), false, false);



            CupDataManager dm = new CupDataManager();
            DataSet ds = dm.LoadData(Properties.Settings.Default.TrainingSetLocation, 10,2);


        }
    }
}
