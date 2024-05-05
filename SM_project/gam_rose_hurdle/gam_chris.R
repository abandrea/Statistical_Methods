load("/home/sebas/Documents/chris/statistica/FINAL_PROJECT/HealthCareAustralia.rda")
data <- ex3.health
head(data)
names(data)

data$ifvisit <- ifelse(data$doctorco == 0, 0, 1)
hist(data$ifvisit)
perc_nonzero <- (sum(data$ifvisit)/length(data$ifvisit))

data$sex <- factor(data$sex)

thresholds <- c(0, 0.3, 0.7, 1)

data$age_factor <- cut(data$age, breaks = thresholds, 
                       labels = c("young", "adult", "old"), include.lowest = TRUE)

set.seed(42)
train_indices <- sample(seq_len(nrow(data)), 0.8 * nrow(data))
train_data <- data[train_indices, ]
test_data <- data[-train_indices, ]

library("mgcv")
model3 <- gam(ifvisit ~ sex*hscore +  illness*actdays +
                prescrib, data = train_data, 
             family = poisson(link = "log"))

summary(model3)
AIC(model3)
BIC(model3)

data$ifvisit <- ifelse(data$doctorco == 0, 0, 1)

model <- gam(ifvisit ~ s(actdays)+income*freepoor + hospadmi + nonpresc + hscore + illness * actdays + 
               prescrib*age_factor , data=train_data, family = binomial(link = "logit"))
summary(model)
AIC(model)


model3 <- glm(ifvisit ~ I(hscore^2)+income*freepoor + hospadmi + nonpresc + hscore + illness * actdays + 
                prescrib*age_factor, data = train_data, family = binomial)
predicted_counts <- round(predict(model, newdata = test_data, type = "response"))
true_counts <- test_data$ifvisit
mse <- mean((true_counts - predicted_counts)^2)
print(paste("Mean Squared Error (MSE):", mse))
rmse <- mean(abs(predicted_counts - true_counts))
print(paste("MAE:",rmse))