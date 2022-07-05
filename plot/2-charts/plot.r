library(ggplot2)
library(dplyr)


########------------- Read data -------------############

options(scipen=999)
bi.data <- readr::read_csv("bi_objective.csv", show_col_types = FALSE)


########------------- Objectives -------------############
bi.data0 <- bi.data %>% filter(predicate == "VER")
bi.data0$f2[bi.data0$lambda == 1] <- as.numeric(bi.data0[bi.data0$lambda == 1, ]$f2) * 100 / 740.725589257187
bi.data0$f2[bi.data0$lambda == 2] <- as.numeric(bi.data0[bi.data0$lambda == 2, ]$f2) * 100 / 1313.37213444108
bi.data0$f2[bi.data0$lambda == 3] <- as.numeric(bi.data0[bi.data0$lambda == 3, ]$f2) * 100 / 2839.76182586111
bi.data0$lambda <- factor(bi.data0$lambda , levels = c("3", "2", "1"))

f0 <- ggplot(data = bi.data0, aes(x=f2, y=f1, fill=as.factor(lambda))) +
  geom_vline(xintercept=25,linetype=2) +
  geom_vline(xintercept=50,linetype=2) +
  geom_vline(xintercept=75,linetype=2) +
  geom_vline(xintercept=100,linetype=2) +
  #geom_line(aes(linetype=as.factor(lambda), color=as.factor(lambda))) +
  geom_line(aes(color=as.factor(lambda))) +
  geom_point(aes(shape=as.factor(lambda), color=as.factor(lambda)), size=1) +
  labs(y="Weighted Number of At-risk Individuals Assigned", x="Percentage of Maximum Error Introduced") +
  scale_colour_manual(values = c("red", "blue", "green")) +
  #scale_colour_grey(start = 0.1, end = 0.8) +
  theme_bw() + 
  theme(plot.margin = margin(0.5, 0.5, 0.5, 0.5, unit = "cm"),
        strip.placement = "outside",
        axis.text = element_text(size = 8),
        panel.grid.major.y = element_blank())
f0 


########------------- Metrics -------------############

bi.data$predicate <- factor(bi.data$predicate, levels = c("VER", "ER", "R"))
bi.data$lambda <- factor(bi.data$lambda , levels = c("3", "2", "1"))

f1 <- ggplot(data = bi.data, aes(x=smape, y=risk_1, fill=as.factor(lambda))) +
  #geom_line(aes(linetype=as.factor(lambda), color=as.factor(lambda))) +
  geom_line(aes(color=as.factor(lambda))) +
  #geom_point(aes(shape=as.factor(lambda), color=as.factor(lambda)), size=1) +
  labs(x="SME", y=expression("Global Disclosure Risk ("~tau~')')) +
  facet_grid(cols=vars(predicate)) +
  scale_colour_manual(values = c("red", "blue", "green")) +
  #scale_colour_grey(start = 0.1, end = 0.8) +
  theme_bw() + 
  theme(plot.margin = margin(0.5, 0.5, 0.5, 0.5, unit = "cm"),
        strip.placement = "outside",
        axis.text = element_text(size = 8),
        panel.grid.major.y = element_blank())
f1 

f2 <- ggplot(data = bi.data, aes(x=smape, y=risk_2, fill=as.factor(lambda))) +
  #geom_line(aes(linetype=as.factor(lambda), color=as.factor(lambda))) +
  geom_line(aes(color=as.factor(lambda))) +
  #geom_point(aes(shape=as.factor(lambda), color=as.factor(lambda)), size=1) +
  labs(x="SME", y=expression("Population Uniqueness ("~phi~')')) +
  facet_grid(cols=vars(predicate)) +
  scale_colour_manual(values = c("red", "blue", "green")) +
  #scale_colour_grey(start = 0.1, end = 0.8) +
  theme_bw() + 
  theme(plot.margin = margin(0.5, 0.5, 0.5, 0.5, unit = "cm"),
        strip.placement = "outside",
        axis.text = element_text(size = 8),
        panel.grid.major.y = element_blank())
f2
