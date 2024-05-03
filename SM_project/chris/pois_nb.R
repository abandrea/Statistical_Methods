load("/home/sebas/Documents/chris/statistica/FINAL_PROJECT/HealthCareAustralia.rda")
data <- ex3.health

thresholds <- c(0, 0.3, 0.7, 1)

data$age_factor <- cut(data$age, breaks = thresholds, 
                       labels = c("young", "adult", "old"), include.lowest = TRUE)

set.seed(42)
train_indices <- sample(seq_len(nrow(data)), 0.8 * nrow(data))
train_data <- data[train_indices, ]
test_data <- data[-train_indices, ]

poisson_model <- glm(doctorco ~ income*freepoor+ illness*actdays + age*prescrib + nondocco + nonpresc+
                        +prescrib*hospadmi, data = train_data, family = "poisson")
summary(poisson_model)

#overdispersion test
library(AER)
overdispersion_test <- dispersiontest(poisson_model)
print(overdispersion_test)


predicted_counts <- round(predict(poisson_model, newdata=test_data, type = "response"))
counts_best <- ifelse(predicted_counts == 0, 0, 1)
true_counts <- test_data$doctorco
mse <- mean((true_counts - counts_best)^2)
print(paste("Mean Squared Error (pois):", mse))
rmse <- mean(abs(predicted_counts - true_counts))
print(paste("MAE (pois):",rmse)) # 0.27167

par(mfrow=c(1,2))
hist(predicted_counts)
hist(test_data$doctorco)
par(mfrow=c(1,1))
# Residuals for the count model
residuals_count <- test_data$doctorco - predicted_counts
# Calculate residuals
residuals <- residuals(poisson_model, type = "pearson")

# Plot residuals against fitted values
fitted_vals <- fitted(poisson_model)
plot(fitted_vals, residuals, xlab = "Fitted Values", ylab = "Residuals", main = "Residuals vs Fitted")
abline(h = 0, col = "red")
pred_vis <- sum(predicted_counts)
true_vis <- sum(test_data$doctorco)
print(paste("True total number of visits:",true_vis))
print(paste("Total number of predicted visits:", pred_vis))
print(paste("Rate of visits predicted:", pred_vis/true_vis ))





library(MASS)
nb_model <- glm.nb(doctorco ~ illness*actdays + age*prescrib + nondocco + nonpresc+
                     prescrib+ hospadmi + freepoor, data = train_data)
summary(nb_model)


predicted_counts <- round(predict(nb_model, newdata=test_data, type = "response"))
counts_best <- ifelse(predicted_counts == 0, 0, 1)
true_counts <- test_data$doctorco
mse <- mean((true_counts - counts_best)^2)
print(paste("Mean Squared Error (nb):", mse))
rmse <- mean(abs(predicted_counts - true_counts))
print(paste("MAE (nb):",rmse)) #0.28131

par(mfrow=c(1,2))
hist(predicted_counts)
hist(test_data$doctorco)
par(mfrow=c(1,1))
# Residuals for the count model
residuals_count <- test_data$doctorco - predicted_counts
# Calculate residuals
residuals <- residuals(nb_model, type = "pearson")

# Plot residuals against fitted values
fitted_vals <- fitted(nb_model)
plot(fitted_vals, residuals, xlab = "Fitted Values", ylab = "Residuals", main = "Residuals vs Fitted")
abline(h = 0, col = "red")
pred_vis <- sum(predicted_counts)
true_vis <- sum(test_data$doctorco)
print(paste("True total number of visits:",true_vis))
print(paste("Total number of predicted visits:", pred_vis))
print(paste("Rate of visits predicted:", pred_vis/true_vis ))
