library(ROSE)
load("/home/sebas/Documents/chris/statistica/FINAL_PROJECT/HealthCareAustralia.rda")
data <- ex3.health

# Constructing the binary response variable
data$ifvisit <- ifelse(data$doctorco == 0, 0, 1)
barplot(table(data$ifvisit))

# Using ROSE to balance the dataset
data.rose <- ROSE(ifvisit ~ ., data = data, seed = 1, hmult.mino = 0)$data
data.rose <- abs(data.rose)
# we left out hmult.mino because we want it's default value set in the library
barplot(table(data.rose$ifvisit))

data.rose$hospadmi <- round(data.rose$hospadmi)
data.rose$actdays <- round(data.rose$actdays)
data.rose$nondocco <- round(data.rose$nondocco)
data.rose$hospdays <- round(data.rose$hospdays)
data.rose$sex <- round(data.rose$sex)
data.rose$sex <- data.rose$sex - ifelse(data.rose$sex == 2, 1, 0)
data.rose$levyplus <- round(data.rose$levyplus)
data.rose$levyplus <- data.rose$levyplus - ifelse(data.rose$levyplus == 2, 1, 0)
data.rose$freepoor <- round(data.rose$freepoor)
data.rose$freepoor <- data.rose$freepoor - ifelse(data.rose$freepoor == 2, 1, 0)
data.rose$freepera <- round(data.rose$freepera)
data.rose$freepera <- data.rose$freepera - ifelse(data.rose$freepera == 2, 1, 0)
data.rose$illness <- round(data.rose$illness)
data.rose$hscore <- round(data.rose$hscore)
data.rose$chcond1 <- round(data.rose$chcond1)
data.rose$chcond1 <- data.rose$chcond1 - ifelse(data.rose$chcond1 == 2, 1, 0)
data.rose$chcond2 <- round(data.rose$chcond2)
data.rose$chcond2 <- data.rose$chcond2 - ifelse(data.rose$chcond2 == 2, 1, 0)
data.rose$doctorco <- round(data.rose$doctorco)
data.rose$medicine <- round(data.rose$medicine)
data.rose$prescrib <- round(data.rose$prescrib)
data.rose$nonpresc <- round(data.rose$nonpresc)
data.rose$ifvisit <- round(data.rose$ifvisit)
data.rose$ifvisit <- data.rose$ifvisit - ifelse(data.rose$ifvisit == 2, 1, 0)
data.rose$constant <- round(data.rose$constant)
data.rose$constant <- data.rose$constant - ifelse(data.rose$constant == 2, 1, 0)


# Splitting into train / test
set.seed(42)
train_indices <- sample(seq_len(nrow(data.rose)), 0.8 * nrow(data.rose))
train_data <- data.rose[train_indices, ]
test_data <- data.rose[-train_indices, ]

# Fitting a GAM model
library("mgcv")
model_gam <- gam(ifvisit ~ s(age) + s(actdays) + s(hscore) + s(nondocco) + 
                   s(medicine), data=train_data, family = binomial(link = "logit"))
summary(model_gam)
AIC(model_gam)
BIC(model_gam)

par(mfrow=c(1,2))
plot(model_gam, select=3)
plot(model_gam, select=4)
plot(model_gam, residuals = TRUE, pch = 19)
par(mfrow=c(1,1))

# Fitting a GLM model
model_glm <- glm(ifvisit ~  hospadmi + nondocco +
                   illness + actdays + prescrib + age + nonpresc + sex
                 , 
                 data = train_data, family = binomial)
summary(model_glm)
AIC(model_glm)
BIC(model_glm)

par(mfrow=c(1,3))
for(j in 1:3) plot(model_glm, select=j)


# GAM results
predicted_counts <- round(predict(model_gam, newdata = test_data, type = "response"))
true_counts <- test_data$ifvisit

rmse <- mean(abs(predicted_counts - true_counts))
print(paste("MAE (gam):",rmse))
print(paste("Number of true ifvisit:",sum(true_counts)))
print(paste("Number of ifvisit predicted:", sum(predicted_counts)))

par(mfrow = c(1, 2))
hist(predicted_counts,ylim = c(0, 1000))
hist(true_counts,ylim = c(0, 1000))
par(mfrow = c(1, 1))

# GLM results
predicted_counts <- round(predict(model_glm, newdata = test_data, type = "response"))
true_counts <- test_data$ifvisit

rmse <- mean(abs(predicted_counts - true_counts))
print(paste("MAE (glm):",rmse))
print(paste("Number of true ifvisit:",sum(true_counts)))
print(paste("Number of ifvisit predicted:", sum(predicted_counts)))

par(mfrow = c(1, 2))
hist(predicted_counts,ylim = c(0, 1000))
hist(true_counts,ylim = c(0, 1000))
par(mfrow = c(1, 1))
