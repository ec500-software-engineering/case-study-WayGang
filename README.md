# case-study-WayGang
case-study-WayGang created by GitHub Classroom

Let's start from a joke:
-----
Machine Learning, Neural Network...., this is what people think of it:
![image](https://github.com/ec500-software-engineering/case-study-WayGang/blob/master/machine1.jpeg)


![image](https://github.com/ec500-software-engineering/case-study-WayGang/blob/master/machine2.jpeg)
-----
Actually, by keras, what we are doing is like:
![image](https://github.com/ec500-software-engineering/case-study-WayGang/blob/master/mysight.jpeg)

Keras-case-study
=====
Keras is a high-level neural network API. It supports us to do quick-test, 
which means we can have our ideas tested immediately.
Keras would be a good choice when we have these kinds of need:

1.Modular, simple to have a runnable prototype.

2.Support CNN and RNN, or combined.

3.Able to switch between CPU & GPU.



Technology and Platform used for development
-----------------------------------
1.Coding languages: Python. 
Actually C++ has better performance, because c++ is compiled language whereas 
Python is interpreted language. Our computer process python file takes more steps than processing C++ files.
However:
Keras is designed and aimed for its friendly using style. 
Python is such a language that is very comfortable to write. 
So I believe that python would be best choice no matter python is 
not as fast as C++ when it comes to performance comparison.
Thus, if Keras is developped recently, still, I think python would be the better choice.

2.Build system & build tools environment needed: None.
Keras is capable with any python IDE.

3.Frameworks / libraries used in the project: Numpy,CNTK,Tensorflow...
![image](https://github.com/ec500-software-engineering/case-study-WayGang/blob/master/Install_Requirements_Keras.png)


Testing
-----------------------------------
Travis CI as their testing platform. 
They have code coverage metrics for example:https://coveralls.io/github/phreeza/keras, 
to ensure its meaningful.





Defects
-----------------------------------
Analyze two defects in the project--e.g. open GitHub issue, support request tickets or feature request for the project
Does the issue require an architecture change, or is it just adding a new function or?
 make a patch / pull request for the project to fix problem / add feature

1.Why keras apps using multi_gpu_model is slower than single gpu? #9204
multi_gpu_model is from keras.utils and it wraps the application model to use multiple GPU to train. However, it seems that using multi_gpu_model makes the training heavier and slower.
It seems that CPU-side data-preprocessing can be one of the reason that greatly slow down the multi-GPU training
Besides, the current version of multi_gpu_model seems to benefit large NN-models only, such as Xception, since weights synchronization is not the bottleneck. When it is wrapped to simple model such as mnist_cnn and cifar_cnn, weights synchronization is pretty frequent and makes the whole time much slower.

2.Removing layers with layers.pop() doesn't work? #2371
A new batch method is needed. Like a method to pop layers off and manage all the links correctly.


Architecture
---------------------------------


![image](https://github.com/ec500-software-engineering/case-study-WayGang/blob/master/StackedLSTM.png)


Demonstration
-----------------------------------
Please run demo.py
