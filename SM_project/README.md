# SM_project
The presentation should include the following:
- a description of the project’s aim
- an exploratory data analysis, including a possible data-cleaning phase
- selection, description, and possibly comparison of the most suitable statistical models
- comments on results.

As an example, look at: https://www.kaggle.com/code/janiobachmann/patient-charges-clustering-and-regression
For some data files, you will likely find some analyses on the net carried out by others. You can look at them for inspiration, but we ask you to find your original key to the analysis.
Each presentation must last 25 minutes. All group members must be aware of all the project’s parts and take part in the presentation.
You are free to choose the kind of file to organize your presentation, eg, slides, report.


document for random things: https://docs.google.com/document/d/1VjBD6tnAic3ViBbBV862q7X1WaDpXu8-MnTCUmYmIkI/edit?usp=sharing

# The response variable is the dvisits, in the data set its named doctorco??

# TODO

- exploratory analysis (everyone)
- data cleaning, feature extraction, feature importance
- compare results
- try to predict the response variable with various models (Linear, Poisson, GLM, GAM, Trees, Random Forest, MARS, with and without Bootstrap)
- report in Latex
- presentation in RMD

# models to try

 - zero-inflated poisson- christian
 - zero-inflated NB - uroš
 - hurdle-poisson -tanja
 - hurdle-NB -andrea 
 - machine learning tree or svm in two parts: bin + count  -omar

# metrics

 - binary: accuracy, AUC, F1-score, 
 - count: RMSE

# Last meeting

- valuate logistic regression with 0 and non-zero, interpret coefficients, try to implement ROSE ( https://cran.r-project.org/web/packages/ROSE/ROSE.pdf ) (FROM TORELLI!!!!)
- try to update last and better model of ZINB.


Both of them says that we implement a lot of things for the project, and this is good for us. 
They also said to give more attention to the interpretation of coefficients. 


# Presentation of the report

Powerpoint 

----- 0'

1 Andrea: Introduction - EDA analysis - First Cluster Analysis (from test_2)

----- 5'

2 Uros: Second Cluster Analysis - MCA 

----- 10 '

3 Christian glm-gam **

----- 15'

4 Tanja glm-gam **

----- 20'

5 Omar magic machine learning

----- 25'

6 Applause and Standing Ovation 

----- Finish!!!!

(** All statistical models)



