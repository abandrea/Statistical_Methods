load("/home/sebas/Documents/chris/statistica/FINAL_PROJECT/HealthCareAustralia.rda")
data <- ex3.health

set.seed(42)
train_indices <- sample(seq_len(nrow(data)), 0.8 * nrow(data))
train_data <- data[train_indices, ]
test_data <- data[-train_indices, ]

library("pscl")
updated_model38_NBZ <- zeroinfl(doctorco ~ 
                                  illness*actdays + hscore + chcond1 + age: chcond2 
                                + hospadmi  +prescrib + nonpresc|
                                  levyplus + age:income:freepoor + freepera+ 
                                  illness * actdays +prescrib, 
                                data = train_data, dist = "negbin")

AIC(updated_model38_NBZ)
summary(updated_model38_NBZ)


predicted_counts <- round(predict(updated_model38_NBZ, newdata = test_data, type = "response"))
true_counts <- test_data$doctorco
rmse <- mean(abs(predicted_counts - true_counts))
print(paste("MAE:",rmse)) #0.26975
par(mfrow=c(1,2))
hist(predicted_counts)
hist(test_data$doctorco)
par(mfrow=c(1,1))
# Residuals for the count model
residuals_count <- test_data$doctorco - predicted_counts

# Calculate residuals
residuals <- residuals(updated_model38_NBZ, type = "pearson")

# Plot residuals against fitted values
fitted_vals <- fitted(updated_model38_NBZ)
plot(fitted_vals, residuals, xlab = "Fitted Values", ylab = "Residuals", main = "Residuals vs Fitted")
abline(h = 0, col = "red")
pred_vis <- sum(predicted_counts)
true_vis <- sum(test_data$doctorco)
print(paste("True total number of visits:",true_vis))
print(paste("Total number of predicted visits:", pred_vis))
print(paste("Rate of visits predicted:", pred_vis/true_vis ))