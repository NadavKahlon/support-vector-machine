# Support Vector Machine (SVM)

This library provides an implementation of Support 
Vector Machines: a powerful machine-learning 
classification model, that was considered among the 
state-of-the-art before the rise of artificial neural 
networks. 

I worked on this project in 2021, and uploaded it to 
GitHub in 2024.

The models' training process is based on a custom
implementation of the Sequential Minimal Optimization
(SMO) algorithm, designed by John C. Platt, and 
described in his 1998 paper: _Sequential Minimal 
Optimization: A Fast Algorithm for Training Support 
Vector Machines_. 

Though SVMs are binary classifiers by nature, we also
include an implementation of an SVM-based _multiclass_
classification model, based on the "1-versus-all" 
scheme.

An example usage of the library, fit on the Iris dataset,
can be found at `examples/iris_classification.py`.