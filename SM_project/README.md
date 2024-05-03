# Demand for health care in Australia
_____

## About the project

In this final project for the course "344SM - Statistical Methods" we were assigned a dataset concerning australian health survey from 1977 to 1978. The dataset contains information of 5190 individuals and 19 variables, including the response variable `doctorco`, which is the number of doctor consulations with a doctor or specialist in the past 2 weeks. 

For the purpose of this project, after the initial Explanatory Data Analysis, which includes a data-cleaning and data visualization, we will focus on the prediction of the number of consulations with a doctor or specialist in the past 2 weeks. We will use different statistical models and some of machine learning techniques, such as:

* Negative Binomial Regression
* Zero-Inflated Poisson Regression (ZIPB)
* Zero-Inflated Negative Binomial Regression (ZINB)
* Hurdle Poisson Regression
* Hurdle Negative binomial Regression
* Logistic Regression 
* Random Forest
* Naive Bayes
* Neural Networks
* K-Nearest Neighbors

The whole project was implemented with two programming languages, R and Python, thereby harnessing the specialized capabilities of each to create a robust, efficient, and versatile analytical workflow. This dual-language approach facilitated the leveraging of R's advanced statistical analysis and visualization tools anlogside Python's superior data manipulation, machine learning, and integration capabilities. As a result, tjhe project benefited from the rich and diverse libraries of both ecosystems, ensuring that each stage of the project, from data cleaning to complex statistical modeling and even deployment, was handled with the most effective tools available. This method not only enhanced the overall quality and depth of the analysis but also fostered a flexible and innovative environment for tackling the multifaceted challenges of the project.

________

## Considerations before starting the analysis - Levy in Australia health care in 70-80s

Health care in Australia during 1977-1978 was is a transitional phase, with major focus on a universal health insurance system called Medibank. The Medibank scheme was introduced by the Whitlam Government in 1975 through the Health Insurance Act 1973. The scheme was intended to provide universal health insurance to all Australians and to provide free treatment in public hospitals. The scheme was to be funded by a 2.5% levy on taxable incomes, an additional levy on high-income earners, as well as government funding. The Medibank scheme was later abolished by the Fraser Government in 1981, and replaced by a government-subsidised private insurance scheme, called Medibank Private, which exists to this day. The dataset we are using is from the period when the Medibank scheme was in place, and it is interesting to see how the health care system was functioning at that time.

The Australian health care system has a federal structure, with responsibilities split between federal and state/territory governments. The federal government was primarly responsible for funding and policy, while the state/territory governments were responsible for the delivery of health care services. The cost of running Medibank was a signiﬁcant concerns for the government during this period. There were debates aboyt whether Medibank provided truly equitable access for all Australians and concerns about longer waiting times in the public system.

________

Structure of the project:

```

⦿--ABSTRACT
|-- Explanatory Data Analysis
|   |-- Dataset Analysis
|   |   |-- Response Variable 'doctorco'
|   |   |-- Variables
|   |   └-- Considerations
|   |-- Bivariate Analysis: 'doctorco' vs. other variables
|   |-- Correlation Matrix
|   |   |-- 'doctorco' vs. 'sex'
|   |   |-- 'doctorco' vs. 'age'
|   |   |-- 'doctorco' vs. 'income'
|   |   |-- 'doctorco' vs. 'levyplus'
|   |   |-- 'doctorco' vs. 'freepoor'
|   |   |-- 'doctorco' vs. 'freepera'
|   |   |-- 'doctorco' vs. 'illness'
|   |   |-- 'doctorco' vs. 'actdays'
|   |   |-- 'doctorco' vs. 'hscore'
|   |   |-- 'doctorco' vs. 'chcond1'
|   |   |-- 'doctorco' vs. 'chcond2'
|   |   |-- 'doctorco' vs. 'nondocco'
|   |   |-- 'doctorco' vs. 'hospadmi'
|   |   |-- 'doctorco' vs. 'hospdays'
|   |   |-- 'doctorco' vs. 'medicine'
|   |   |-- 'doctorco' vs. 'prescrib'
|   |   |-- 'doctorco' vs. 'nonpresc'
|   |   └-- Summary
|   |-- Variable combination - Interaction effect on response variable
|   |   |-- Interaction between 'actdays' and 'illness'
|   |   |-- Interaction between 'age' and 'prescrib'
|   |   |-- Interaction between 'income' and 'freepoor'
|   |   |-- Interaction between 'income' and 'levyplus'
|   |   └-- Interaction between 'sex' and 'hscore'
|   └-- Cluster Analysis
|       |-- Elbow Method
|       |-- Cluster profiling
|       |-- Parallel Coordinates Plot
|       |-- Projecting the Data to 2 Dimensions
|       |-- Explaining MCA
|       |-- Correspondence Analysis
|       |-- PCA Based Explanation
|       |-- Down-projecting with PCA
|       |-- Down-projecting with MCA
|       |-- Visualizing the Data using MCA
|       |-- Looking at Different Principal Components
|       |-- 3D Down-Projection
|       |-- Additional Data Exploration
|       └-- Visualizing Higher Dimensional Clusters
|-- Binary Classification Problem
|   └-- Balancing the dataset with ROSE
|       |-- GAM with ROSE
|       └-- GLM with ROSE
|-- Zero-Inflated Negative Binomial (ZINB) Regression
|-- Hurdle Negative Binomial Regression
|-- Zero-Inflated VS Hurdle
|-- Lookup model
|-- Random Forest
|-- Neural Networks
|-- Poisson Regression
|-- Logistic Regression
|-- MCA-kNN
└-- Trained distance

```

________

## References:

- https://www.health.gov.au/medicare-turns-40/history#:~:text=The%20Australian%20Government%2C%20under%20Prime,Australian%20Labor%20Party%20formed%20government

- https://www.nma.gov.au/defining-moments/resources/medicare#:~:text=The%20incoming%20Fraser%20government%20modified,had%20blocked%20while%20in%20opposition

- Young DS, Roemmele ES, Yeh P. Zero-inflated modeling part I: Traditional zero-inflated count regression models, their applications, and computational tools. WIREs Comput Stat. 2022; 14:e1541. https://doi.org/10.1002/wics.1541

- Lambert, Diane. 1992. “Zero-Inflated Poisson Regression, with an Application to Defects in Manufacturing.” Technometrics 34 (1).

- Fletcher, David & MacKenzie, Darryl & Villouta Stengl, Eduardo. (2005). Modelling skewed data with many zeros: A simple approach combining ordinary and logistic regression. Environmental and Ecological Statistics. 12. 45-54. 10.1007/s10651-005-6817-1 (https://www.researchgate.net/publication/226071827_Modelling_skewed_data_with_many_zeros_A_simple_approach_combining_ordinary_and_logistic_regression)

- Farbmacher, Helmut. "Estimation of hurdle models for overdispersed count data." The Stata Journal 11.1 (2011): 82-94.

- Ana Gonzalez-Blanks, Jessie M. Bridgewater & Tuppett M. Yates (2020) Statistical Approaches for Highly Skewed Data: Evaluating Relations between Maltreatment and Young Adults’ Non-Suicidal Self-injury, Journal of Clinical Child & Adolescent Psychology, 49:2, 147-161, DOI: 10.1080/15374416.2020.1724543

- Dixit SK, Sambasivan M. A review of the Australian healthcare system: A policy perspective. SAGE Open Medicine. 2018;6. doi:10.1177/2050312118769211

- C. Ford, 2016. "Getting started with Hurdle Models." UVA Library, StatLab. https://library.virginia.edu/data/articles/getting-started-with-hurdle-models

- Feng, C.X. A comparison of zero-inflated and hurdle models for modeling zero-inflated count data. J Stat Distrib App 8, 8 (2021). https://doi.org/10.1186/s40488-021-00121-4

- Packages used: `ggplot2`, `Hmisc`, `mgcv`, `pROC`, `PRROC`, `pscl`, `reticulate`, `ROSE`, `tidyverse`, `base`, `datasets`, `dplyr`, `forcats`, `graphics`, `grDevices`, `lattice`, `lubridate`, `MASS`, `methods`, `nlme`, `purrr`, `readr`, `rlang`, `stats`, `stringr`, `tibble`, `tidyr`, `utils`.

________
### Notes

This project is W.I.P. and the README file will be updated as the project progresses.

