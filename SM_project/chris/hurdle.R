load("/home/sebas/Documents/chris/statistica/FINAL_PROJECT/HealthCareAustralia.rda")
data <- ex3.health


thresholds <- c(0, 0.3, 0.7, 1)

data$age_factor <- cut(data$age, breaks = thresholds, 
                       labels = c("young", "adult", "old"), include.lowest = TRUE)

set.seed(42)
train_indices <- sample(seq_len(nrow(data)), 0.8 * nrow(data))
train_data <- data[train_indices, ]
test_data <- data[-train_indices, ]
###HURDLE-poisson

library(pscl)

initial_hurdle_poisson <- hurdle(doctorco ~ age + sex + income + levyplus + freepoor + freepera + illness + actdays + hscore + chcond1 + chcond2 + nondocco + hospadmi + hospdays + prescrib + nonpresc | age + sex + income + levyplus + freepoor + freepera + illness + actdays + hscore + chcond1 + chcond2 + nondocco + hospadmi + hospdays + prescrib + nonpresc, 
                                 data = train_data, dist = "poisson")


updated_hurdle_poisson <- hurdle(doctorco ~ income+ freepera +illness + actdays + hospadmi + nonpresc| 
                                   sex+levyplus + freepoor + freepera+ actdays + illness + hscore +hospadmi +  prescrib+ nonpresc, 
                                 data = train_data, dist = "poisson")

# Compare AIC values
AIC(initial_hurdle_poisson, updated_hurdle_poisson)





###HURDLE-NB
initial_hurdle_NB <- hurdle(doctorco ~ age + sex + income + levyplus + freepoor + freepera + illness + actdays + hscore + chcond1 + chcond2 + nondocco + hospadmi + hospdays + prescrib + nonpresc | age + sex + income + levyplus + freepoor + freepera + illness + actdays + hscore + chcond1 + chcond2 + nondocco + hospadmi + hospdays + prescrib + nonpresc, 
                            data = train_data, dist = "negbin")
AIC(initial_hurdle_NB)


updated1_hurdle_NB <- hurdle(doctorco ~ age +sex+ income +levyplus+ freepoor + freepera + illness + actdays + hscore + chcond1 + chcond2 + nondocco + hospadmi + hospdays + prescrib + nonpresc | 
                               sex+levyplus + freepoor + freepera+ actdays + illness + hscore +hospadmi +  prescrib+ nonpresc , 
                             data = train_data, dist = "negbin")
AIC(updated1_hurdle_NB)

hurdle_tanja <- hurdle(doctorco ~ income+ freepera +illness + actdays + hospadmi + nonpresc| 
                               sex+levyplus + freepoor + freepera+ actdays + illness + hscore +hospadmi +  prescrib+ nonpresc , 
                             data = train_data, dist = "negbin")

AIC(updated2_hurdle_NB)

#trying to find interactions, 
best_hurdle_NB <- hurdle(doctorco ~ illness +actdays + hospadmi| 
                              income:freepoor +  actdays *illness + sex*hscore +
                           hospadmi +  age*prescrib+ nonpresc, 
                            data = train_data, dist = "negbin")
AIC(best_hurdle_NB)
summary(best_hurdle_NB)

predicted_counts <- round(predict(best_hurdle_NB, newdata=test_data, type = "response"))
counts_best <- ifelse(predicted_counts == 0, 0, 1)
true_counts <- test_data$doctorco
rmse <- mean(abs(predicted_counts - true_counts))
print(paste("MAE:",rmse)) # 0.27167

mse <- mean((true_counts - counts_best)^2)
print(paste("Mean Squared Error (best):", mse))
print(predicted_counts)
par(mfrow=c(1,2))
hist(predicted_counts)
hist(test_data$doctorco)
par(mfrow=c(1,1))
# Residuals for the count model
residuals_count <- test_data$doctorco - predicted_counts
residuals_count
# Calculate residuals
residuals <- residuals(best_hurdle_NB, type = "pearson")

# Plot residuals against fitted values
fitted_vals <- fitted(best_hurdle_NB)
plot(fitted_vals, residuals, xlab = "Fitted Values", ylab = "Residuals", main = "Residuals vs Fitted")
abline(h = 0, col = "red")
pred_vis <- sum(predicted_counts)
true_vis <- sum(test_data$doctorco)
print(paste("True total number of visits:",true_vis))
print(paste("Total number of predicted visits:", pred_vis))
print(paste("Rate of visits predicted:", pred_vis/true_vis ))





try_hurdle_NB <- hurdle(doctorco ~ income*freepoor + hospadmi + nonpresc + 
                          hscore + illness * actdays + prescrib*age_factor| 
                          income:freepoor +  actdays *illness + sex*hscore +
                          hospadmi +  age*prescrib+ nonpresc, 
                        data = train_data, dist = "negbin")
AIC(try_hurdle_NB)
summary(try_hurdle_NB)

predicted_counts <- round(predict(try_hurdle_NB, newdata=test_data, type = "response"))
counts_try <- ifelse(predicted_counts == 0, 0, 1)
true_counts <- ifelse(test_data$doctorco == 0, 0, 1)
mse <- mean((true_counts - counts_try)^2)
print(paste("Mean Squared Error (try):", mse))
rmse <- mean(abs(counts_try - true_counts))
print(paste("MAE:",rmse))
print(predicted_counts)
par(mfrow=c(1,2))
hist(predicted_counts)
hist(test_data$doctorco)
par(mfrow=c(1,1))
# Residuals for the count model
residuals_count <- test_data$doctorco - predicted_counts
residuals_count
# Calculate residuals
residuals <- residuals(try_hurdle_NB, type = "pearson")

# Plot residuals against fitted values
fitted_vals <- fitted(try_hurdle_NB)
plot(fitted_vals, residuals, xlab = "Fitted Values", ylab = "Residuals", main = "Residuals vs Fitted")
abline(h = 0, col = "red")
pred_vis <- sum(predicted_counts)
true_vis <- sum(test_data$doctorco)
print(paste("True total number of visits:",true_vis))
print(paste("Total number of predicted visits:", pred_vis))
print(paste("Rate of visits predicted:", pred_vis/true_vis ))
