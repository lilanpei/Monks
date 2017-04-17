using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;

namespace CustomExtensionMethods
{
    static class ExtensionMethods
    {
        private static Random rng = new Random();

        public static void Shuffle<T>(this IList<T> list)
        {
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

        public static Matrix<double> Vec2Vecmultiply(this Vector<double> lhs, Vector<double> rhs)
        {
            var mul = lhs.ToColumnMatrix() * rhs.ToRowMatrix();
            return mul.Transpose();//CreateMatrix.Dense(1, rhs.Count, mul.ToArray()).Transpose();
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
