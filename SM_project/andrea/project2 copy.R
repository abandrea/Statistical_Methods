load("HealthCareAustralia.rda")
data<-ex3.health

#Data_analysis
str(data)
head(data)
summary(data)

###multicolinearity
library(car)

# Check VIF values
vif_model <- lm(doctorco ~ age + sex + income + levyplus + freepoor + freepera + illness + actdays + hscore + chcond1 + chcond2 + nondocco + hospadmi + hospdays + prescrib + nonpresc , data = data)
vif(vif_model)



#######################GROUPING 3 AND MORE VISITS

# Create a new variable 'doctorco_grouped' with the specified grouping
data$doctorco_grouped <- ifelse(data$doctorco >= 3, 3, as.character(data$doctorco))


#il migliore
updated_model10_NBZ <- zeroinfl(doctorco_grouped  ~ income*levyplus + freepera + illness*actdays + hscore + chcond1 + age:chcond2 + hospadmi  + prescrib + nonpresc|
                                levyplus + freepoor + freepera+ illness * actdays +nondocco+ hospdays + prescrib, data = data, dist = "negbin")
AIC(updated_model10_NBZ, initial_model_NBZ)





###ZERO INFLATED-POISSON

library(MASS)
library(pscl)

# Fit an initial model (you might start with a full model or a null model)
initial_model_poisson <- zeroinfl(doctorco ~ age + sex + income + levyplus + freepoor + freepera + illness + actdays + hscore + chcond1 + chcond2 + nondocco + hospadmi + hospdays + prescrib + nonpresc | age + sex + income + levyplus + freepoor + freepera + illness + actdays + hscore + chcond1 + chcond2 + nondocco + hospadmi + hospdays + prescrib + nonpresc, data = data, dist = "poisson")
updated_model_poisson<- zeroinfl(doctorco ~ +levyplus + freepera + illness + actdays + hscore + chcond1 + chcond2 + hospadmi + prescrib + nonpresc|
                                   levyplus + freepoor + freepera+ illness + actdays + hscore + nondocco + hospdays+ prescrib+nonpresc, data = data, dist = "poisson")
AIC(initial_model_poisson, updated_model_poisson)





###ZERO INFLATED-NB

library(MASS)
library(pscl)


# Fit an initial model (you might start with a full model or a null model)
initial_model_NBZ <- zeroinfl(doctorco ~ age + sex + income + levyplus + freepoor + freepera + illness + actdays + hscore + chcond1 + chcond2 + nondocco + hospadmi + hospdays + prescrib + nonpresc | age + sex + income + levyplus + freepoor + freepera + illness + actdays + hscore + chcond1 + chcond2 + nondocco + hospadmi + hospdays + prescrib + nonpresc, data = data, dist = "negbin")

#il migliore
updated_model_NBZ <- zeroinfl(doctorco ~ income*levyplus + freepera + illness*actdays + hscore + chcond1 + age:chcond2 + hospadmi  + prescrib + nonpresc|
                                levyplus + freepoor + freepera+ illness * actdays +nondocco+ hospdays + prescrib, data = data, dist = "negbin")


AIC(updated_model_NBZ, initial_model_NBZ)
summary(updated_model_NBZ)


##################################################

#trying to add age with splines but is worse
library(splines)
# Fit a ZINB model with spline terms for age
updated_model7_NBZ <- zeroinfl(doctorco ~ income * levyplus + freepera + illness * actdays + hscore + chcond1 + bs(age, knots=c(0.32, 0.52)):chcond2 + hospadmi + prescrib + nonpresc |
                                 +levyplus + freepoor + freepera + illness * actdays + nondocco + hospdays + prescrib, data = data, dist = "negbin")


AIC(updated_model7_NBZ, initial_model_NBZ)

###################################################

#trying to add income for the groups and with splines but its worse

data$income_0_21_0_45 <- ifelse(data$income >= 0.21 & data$income <= 0.45, 1, 0)

# Update your ZINB model to include the new indicator variable
updated_model8_NBZ <- zeroinfl(doctorco ~ income * levyplus + freepera + illness * actdays + hscore +
                                chcond1 + age * chcond2 + hospadmi + prescrib + nonpresc + income_0_21_0_45 |
                                levyplus + freepoor + freepera + illness * actdays + nondocco +
                                hospdays + prescrib, data=data, dist="negbin")

updated_model9_NBZ <- zeroinfl(doctorco ~ bs(income, knots=c(0.21, 0.45)) : levyplus + freepera + illness * actdays +
                                hscore + chcond1 + age * chcond2 + hospadmi + prescrib + nonpresc |
                                levyplus + freepoor + freepera + illness * actdays + nondocco +
                                hospdays + prescrib, data = data, dist = "negbin")


AIC(updated_model9_NBZ, initial_model_NBZ)

####################################################

updated2_model_NBZ <- zeroinfl(doctorco ~  income*levyplus + freepera + illness*actdays + illness* hscore + chcond1 + age*chcond2 + hospadmi  + prescrib + nonpresc|
                         levyplus + freepoor + freepera+ illness + actdays + hscore + nondocco + hospdays+ age*prescrib+ nonpresc, data = data, dist = "negbin")


AIC(updated_model_NBZ, updated2_model_NBZ)


################################################3333


#updating model 

library(splines)


# Adding polynomial terms
updated_model4_NBZ <- update(updated_model_NBZ, . ~ . + I(hscore^2))
AIC(updated_model4_NBZ, updated_model_NBZ)


# Using splines for a flexible non-linear relationship
updated_model5_NBZ <- update(updated_model_NBZ, . ~ . + bs(age, degree = 3))
AIC(updated_model5_NBZ, updated_model_NBZ)

updated_model6_NBZ <- update(updated_model_NBZ, . ~ . - age)
AIC(updated_model5_NBZ, updated_model_NBZ)

#summary
summary(updated_model_NBZ)




##########################residuals 

# Predicted counts
predicted_counts <- predict(updated_model_NBZ, type = "response")

# Residuals for the count model
residuals_count <- data$doctorco - predicted_counts

# Plotting residuals
plot(residuals_count, ylab = "Residuals", xlab = "Predicted Counts", main = "Residuals vs. Predicted Counts")
abline(h = 0, col = "red")

#the mean to the variance of the residuals.
mean_res <- mean(residuals_count)
var_res <- var(residuals_count)

overdispersion_stat <- var_res / mean_res
print(overdispersion_stat)



resid_count <- residuals(updated_model2_NBZ, type = "pearson")
fitted_count <- fitted(updated_model2_NBZ)
plot(fitted_count, resid_count, main = "Residuals vs Fitted (Count Part)", xlab = "Fitted Values", ylab = "Residuals")
abline(h = 0, col = "red")



#################################################33
#update with step aic
updated_model2_NBZ <- zeroinfl(doctorco ~ levyplus + freepoor + freepera + illness + actdays + 
                                 hscore + hospadmi + prescrib + nonpresc, data = data, dist = "negbin")
AIC(updated_model2_NBZ, initial_model_NBZ)


# Perform stepwise selection based on AIC
best_model_NBZ <- stepAIC(initial_model_NBZ, direction = "both")

# Show the summary of the best model
summary(best_model)
AIC(best_model_NBZ, initial_model_NBZ)





###HURDLE-poisson

library(pscl)

initial_hurdle_poisson <- hurdle(doctorco ~ age + sex + income + levyplus + freepoor + freepera + illness + actdays + hscore + chcond1 + chcond2 + nondocco + hospadmi + hospdays + prescrib + nonpresc | age + sex + income + levyplus + freepoor + freepera + illness + actdays + hscore + chcond1 + chcond2 + nondocco + hospadmi + hospdays + prescrib + nonpresc, 
                               data = data, dist = "poisson")


updated_hurdle_poisson <- hurdle(doctorco ~ income+ freepera +illness + actdays + hospadmi + nonpresc| 
                          sex+levyplus + freepoor + freepera+ actdays + illness + hscore +hospadmi +  prescrib+ nonpresc, 
                        data = data, dist = "poisson")

# Compare AIC values
AIC(initial_hurdle_poisson, updated_hurdle_poisson)





###HURDLE-NB
initial_hurdle_NB <- hurdle(doctorco ~ age + sex + income + levyplus + freepoor + freepera + illness + actdays + hscore + chcond1 + chcond2 + nondocco + hospadmi + hospdays + prescrib + nonpresc | age + sex + income + levyplus + freepoor + freepera + illness + actdays + hscore + chcond1 + chcond2 + nondocco + hospadmi + hospdays + prescrib + nonpresc, 
                               data = data, dist = "negbin")

updated1_hurdle_NB <- hurdle(doctorco ~ age +sex+ income +levyplus+ freepoor + freepera + illness + actdays + hscore + chcond1 + chcond2 + nondocco + hospadmi + hospdays + prescrib + nonpresc | 
                                              sex+levyplus + freepoor + freepera+ actdays + illness + hscore +hospadmi +  prescrib+ nonpresc , 
                                  data = data, dist = "negbin")

updated2_hurdle_NB <- hurdle(doctorco ~ income+ freepera +illness + actdays + hospadmi + nonpresc| 
                                     sex+levyplus + freepoor + freepera+ actdays + illness + hscore +hospadmi +  prescrib+ nonpresc , 
                                   data = data, dist = "negbin")

updated3_hurdle_NB <- hurdle(doctorco ~ income*levyplus + freepera + illness*actdays + hscore + chcond1 + age:chcond2 + hospadmi  + prescrib + nonpresc|
                               levyplus + freepoor + freepera+ illness * actdays +nondocco+ hospdays + prescrib, data = data, dist = "negbin")

#updated2 is better than updated3

AIC(initial_hurdle_NB, updated2_hurdle_NB)
AIC(updated_model_NBZ, updated3_hurdle_NB)
BIC(updated_model_NBZ, updated3_hurdle_NB)
summary(initial_hurdle_model)












###evaluation


load_split_data <- function(dataset_loc) {
  # Load the dataset train and test splits. Returns train, test, and the whole
  # df. Input to the function is the path to the dataset.
  set.seed(42)
  load(dataset_loc)
  df <- ex3.health
  sample <- sample(c(TRUE, FALSE), nrow(df), replace=TRUE, prob=c(0.8,0.2))
  train  <- df[sample, ]
  test   <- df[!sample, ]
  return(list("train"=train, "test"=test, "df"=df))
}


data_splits <- load_split_data("HealthCareAustralia.rda")
train <- data_splits$train
test <- data_splits$test

model_NBZ<- zeroinfl(doctorco ~ age + sex + income + levyplus + freepoor + freepera + illness + actdays + hscore + chcond1 + chcond2 + nondocco + hospadmi + hospdays + prescrib + nonpresc | age + sex + income + levyplus + freepoor + freepera + illness + actdays + hscore + chcond1 + chcond2 + nondocco + hospadmi + hospdays + prescrib + nonpresc, data = data, dist = "poisson")
#model_NBZ <- zeroinfl(doctorco ~ levyplus + freepera + illness + actdays + hscore + chcond1 + chcond2 + hospadmi  + prescrib + nonpresc|
                                #levyplus + freepoor + freepera+ illness + actdays + hscore + nondocco + hospdays+ prescrib+nonpresc, data = train, dist = "negbin")

hurdle_NB <- hurdle(doctorco ~ income+ freepera +illness + actdays + hospadmi + nonpresc| 
                               sex+levyplus + freepoor + freepera+ actdays + illness + hscore +hospadmi +  prescrib+ nonpresc , 
                             data = train, dist = "negbin")



###binary predictions

library(pROC)


# For ZINB model
actual_binary <- ifelse(test$doctorco > 0, 1, 0)
predicted_binary_zinb <- predict(model_NBZ, test, type = "zero")

# Calculate AUC
roc_result <- roc(actual_binary, predicted_binary_zinb)
auc_value <- auc(roc_result)

# Calculate accuracy
predicted_class_zinb <- ifelse(predicted_binary_zinb > 0.5, 1, 0)
accuracy <- sum(predicted_class_zinb == actual_binary) / length(actual_binary)

print(paste("AUC: ", auc_value))
print(paste("Accuracy: ", accuracy))


# For hurdle model
actual_binary <- ifelse(test$doctorco > 0, 1, 0)
predicted_binary_hurdle <- predict(hurdle_NB, test, type = "zero")

# Calculate AUC
roc_result_hurdle <- roc(actual_binary, predicted_binary_hurdle)
auc_value_hurdle <- auc(roc_result_hurdle)

# Calculate accuracy
predicted_class_hurdle<- ifelse(predicted_binary_hurdle > 0.5, 1, 0)
accuracy_hurdle <- sum(predicted_class_hurdle == actual_binary) / length(actual_binary)

print(paste("AUC: ", auc_value_hurdle))
print(paste("Accuracy: ", accuracy_hurdle))




##evaluate count

# For ZINB model (similar process for Hurdle model)
predicted_counts_zinb <- predict(model_NBZ, test, type = "response")

# Calculate RMSE
rmse <- sqrt(mean((predicted_counts_zinb - test$doctorco)^2))

print(paste("RMSE: ", rmse))


# For ZINB model (similar process for Hurdle model)
predicted_counts_hurdle <- predict(hurdle_NB, test, type = "response")

# Calculate RMSE
rmse_hurdle <- sqrt(mean((predicted_counts_hurdle - test$doctorco)^2))

print(paste("RMSE: ", rmse_hurdle))

