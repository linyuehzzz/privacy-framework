library(car)
library(MASS)
library(glmnet)
library(dplyr)

options(scipen=999)
hist <- readr::read_csv("data.csv", show_col_types = FALSE)
scaled.hist <- as.data.frame(scale(hist))
lm.fit1 = lm(diff ~ ., data=scaled.hist)
summary(lm.fit1)

par(mfrow=c(1, 2))
plot(fitted.values(lm.fit1), residuals(lm.fit1), pch=16, 
     xlab="Predicted Value", ylab="Residual", main="")
abline(h=0, lty=2)
qqPlot(rstandard(lm.fit1))

lm.both.aic = step(lm.fit1, direction="both")
summary(lm.both.aic)


