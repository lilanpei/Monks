# Hello
In this repository we extend the work we did back in the machine learning foundation course of unipi to cover more implementation requirements for unipi's most awesome Computational Mathematics course, our goal is to enhance the current implementation and provide a linear regression model as well:

Requirements:

a NN with topology and activation function of your choice (provided it is differentiable)

a standard linear regression (min last square)

Implement yourself two training algorithms of the NN using:

a standard momentum descent approach [references: http://www.cs.toronto.edu/~fritz/absps/momentum.pdf]

an algorithm of the class of accelerated gradient methods [reference: https://www.cs.cmu.edu/~ggordon/10725-F12/slides/09-acceleration.pdf, http://www.cs.toronto.edu/~adeandrade/assets/aconntmftc.pdf, https://arxiv.org/pdf/1412.6980.pdf]

and

a basic version of the linear least squares solver of your choice.
using the programming language of your choice (C/C++, Python, Matlab) but no ready-to-use optimization libraries. For the avoidance of doubt, this means that you may use library functions (Matlab ones or otherwise) if an inner step of the algorithm requires them as a subroutine, but your final implementation should not be a single library call. Also, blatant copying from existing material, either provided by the teachers or found on the Internet, will be mercilessly crushed upon. Ask the teachers if you are uncertain about what this means in the context of your project (for instance, you are using a SVD, for which details were not seen in the lectures, a full implementation will not be required).

Required output of the project are:

A PDF document (LaTeX typesetting advised but not mandatory) describing in details:

the optimization problem to be solved

the implemented solution methods, with a discussion of all relevant details (stopping criterion employed, line search used, algorithmic parameters and their setting) for both

a summary of the known theoretical convergence results for the approaches and a discussion about whether or not they apply to the problem at hand and why

a discussion of which method among the ones seen during the lectures (normal equations, with which inner solution method, QR, SVD) rates to be more effective for the linear least square problem, on grounds of stability and computational cost

the description of experiments aimed at finding the best algorithmic parameters (comprised different accelerated gradient formulae if tested) for solving the problem at hand

the description of the behavior of the approaches on the provided data, evaluating the effectiveness (capability of finding good solutions) and efficiency (convergence rate and running time) when compared to each other;

the comparison of the behavior of the implemented min least square approach with the off-the-shelf implementation available in your programming language in terms of speed and accuracy

optionally, a comparison with efficiency and effectiveness of available off-the-shelf tools (factoring in elements like difference of programming language if necessary) is appreciated

optionally, a comparison of utility of the obtained solutions in terms of Machine Learning performances (generalization capabilities of the linear regression and the NN) is also appreciated; note that this is mandatory if the project is used for the ML course, too

The source code of the implemented approach, comprised any batch or auxiliary file required to run the experiments, properly documented and with README files describing structure and use of the package

Results of the experiments in spreadsheet/databases/text files


# Stay Tuned! ...



# -----------

# old note from The ML foundations project...

# Hello and Welcome
In this repository we provide our results of the Monks problems and AA1 CUP  using our Multilayer Perceptron implementation in C#.NET

We have implemented a lightweight library for building, training and testing multilayer perceptrons. And tested it on the Monks datasets available on https://archive.ics.uci.edu/ml/datasets/MONK's+Problems


### In the project MLPTestDemo.Program.cs
a demo for building, training,saving,loading and testing a MLP
### In AA1_CUP.Screening.cs
a demo for the screening process
### In AA1_CUP.Program.cs
a demo for using the code to perform the regression on the test data

# Report And details:
https://docs.google.com/document/d/1v0lOPbnfBgjSr-iws1u0cb9STbOj7EJMAdh5VG40dBA/edit?usp=sharing
