library(car)
library(MASS)
library(glmnet)
library(dplyr)
library(leaps)

options(scipen=999)
hist <- readr::read_csv("data.csv", show_col_types = FALSE)
summary(hist)

scaled.hist <- as.data.frame(scale(hist))
lm.fit1 = lm(diff ~ ., data=scaled.hist)
summary(lm.fit1)

# stepwise
lm.both.aic = step(lm.fit1, direction="both")
summary(lm.both.aic)
anova(lm.both.aic)

# lasso
y <- hist$diff
x <- data.matrix(hist[, c('white', 'black', 'aian', 'aapi', 'others', 'nva', 'va', 'nh', 'h', 'total')])

lambdas_to_try <- 10^seq(-2, 1, length.out = 100)
cv_model <- cv.glmnet(x, y, alpha=1, lambda=lambdas_to_try)
plot(cv_model)
lambda_cv <- ridge_cv$lambda.min
best_model <- glmnet(x, y, alpha = 0, lambda = lambda_cv, standardize = TRUE)
coef(best_model)

lm.fit1 = lm(diff ~ white + black + aian + aapi + others + nva + va + nh + h, data=scaled.hist)
summary(lm.fit1)
