load("/home/sebas/Documents/chris/statistica/FINAL_PROJECT/HealthCareAustralia.rda")
data <- ex3.health
head(data)
names(data)

data$ifvisit <- ifelse(data$doctorco == 0, 0, 1)
hist(data$ifvisit)
perc_nonzero <- (sum(data$ifvisit)/length(data$ifvisit))

data$pre_act0 <- ifelse((data$prescrib == 0) * (data$actdays ==0), 0, 1)
data$pre0 <- ifelse(data$prescrib ==0, 0, 1)
data$pre_nond0 <- ifelse((data$nondocco == 0) * (data$prescrib ==0), 0, 1)


data$sex <- factor(data$sex)

thresholds <- c(0, 0.3, 0.7, 1)

data$age_factor <- cut(data$age, breaks = thresholds, 
                       labels = c("young", "adult", "old"), include.lowest = TRUE)

set.seed(42)
train_indices <- sample(seq_len(nrow(data)), 0.8 * nrow(data))
train_data <- data[train_indices, ]
test_data <- data[-train_indices, ]

library("mgcv")
model_gam <- gam(ifvisit ~ s(actdays)+income*freepoor + hospadmi + nonpresc + 
                   hscore + illness * actdays + prescrib*age_factor + pre_act0, 
                 data=train_data, family = binomial(link = "logit"))

summary(model_gam)
AIC(model_gam)
BIC(model_gam)

model_glm <- glm(ifvisit ~  hospadmi + nonpresc + hscore + 
                   illness * actdays + prescrib*age_factor + pre_act0, 
                 data = train_data, family = binomial)
summary(model_glm)
AIC(model_glm)
BIC(model_glm)

predicted_counts <- round(predict(model_glm, newdata = test_data, type = "response"))
true_counts <- test_data$ifvisit
mse <- mean((true_counts - predicted_counts)^2)
print(paste("Mean Squared Error (MSE):", mse))
rmse <- mean(abs(predicted_counts - true_counts))
print(paste("MAE (binary):",rmse))

print(paste("Number of true 0:",sum(true_counts == 0)))
print(paste("Number of correct predictions:",sum((true_counts - predicted_counts)==0)))
print(paste("Number of true ifvisit:",sum(true_counts == 1)))
print(paste("Number of ifvisit predicted:", sum(predicted_counts==1)))

par(mfrow = c(1, 2))
hist(predicted_counts,ylim = c(0, 1000))
hist(true_counts,ylim = c(0, 1000))
par(mfrow = c(1, 1))

# Blob is data where doctorco is not 0
blob_data <- data[data$doctorco != 0, ]
blob_data$doctorco <- blob_data$doctorco -1

library(MASS)
model_nb <- glm.nb(doctorco ~ actdays + hospadmi, data = blob_data)
summary(model_nb)

data$doctorco <- data$doctorco-1

predicted_nb <- round(predict(model_nb, newdata = test_data, type = "response")) + 1
true_counts <- test_data$doctorco
pred_final <- predicted_nb * predicted_counts 
rmse <- mean(abs(pred_final - true_counts))
print(paste("MAE (whole):",rmse))
# GAM <- 0.28227
# GLM <- 0.29480

true_vis <- sum(test_data$doctorco)
pred_vis <- sum(pred_final)
print(paste("Number of true visits:",true_vis))
print(paste("Number of predicted visits:", pred_vis))

print(paste("Rate of visits predicted:", pred_vis/true_vis ))