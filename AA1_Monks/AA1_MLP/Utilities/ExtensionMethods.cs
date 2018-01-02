using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;

namespace AA1_MLP.CustomExtensionMethods
{
    /// <summary>
    /// Some extension methods we needed to provide for the Math.NET Library
    /// </summary>
    public static class ExtensionMethods
    {
        private static Random rng = new Random();

        /// <summary>
        /// Shuffles a given C#.NET List
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="list"></param>
        public static void Shuffle<T>(this IList<T> list, int? seed=null)
        {
            if (seed!=null)
            {
                rng = new Random((int)seed);
            }

            int n = list.Count;
            while (n > 1)
            {
                n--;
                int k = rng.Next(n + 1);
                T value = list[k];
                list[k] = list[n];
                list[n] = value;
            }
        }

        public static Matrix<double> Mtrx2Vecmultiply(this Matrix<double> lhs, Vector<double> rhs)
        {
            var mul = lhs * rhs;
            return mul.ToColumnMatrix();
        }


        public static Matrix<double> Vec2Mtrxmultiply(this Vector<double> lhs, Matrix<double> rhs)
        {
            var mul = rhs * lhs.ToRowMatrix();
            return mul;
        }




    }
}
