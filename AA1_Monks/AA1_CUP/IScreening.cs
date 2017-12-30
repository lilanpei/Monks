using AA1_MLP.Entities;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace AA1_CUP
{
    interface IScreening
    {
        void Screen(DataSet wholeSet, int k = 0);

    }
}
