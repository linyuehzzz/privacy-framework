library(ggplot2)
library(dplyr)

########------------- Read data -------------############

options(scipen=999)
bi.data <- readr::read_csv("bi_objective.csv")


########------------- Objectives -------------############
bi.data0 <- bi.data %>% filter(predicate == "VER")

f0 <- ggplot(data = bi.data0, aes(x=f1, y=f2, fill=as.factor(lambda))) +
  geom_line(aes(linetype=as.factor(lambda), color=as.factor(lambda))) +
  geom_point(aes(shape=as.factor(lambda), color=as.factor(lambda))) +
  labs(y = expression("Total Percentage Error"), x = "Number of Individuals Assigned") +
  scale_colour_grey(start = 0.1, end = 0.8) +
  theme_bw() + 
  theme(plot.margin = margin(0.5, 0.5, 0.5, 0.5, unit = "cm"),
        strip.placement = "outside",
        axis.text = element_text(size = 8),
        panel.grid.major.y = element_blank())
f0 


########------------- Metrics -------------############

bi.data$predicate <- factor(bi.data$predicate, levels = c("VER", "ER", "R"))

f1 <- ggplot(data = bi.data, aes(x=risk_1, y=smape, fill=as.factor(lambda))) +
  geom_line(aes(linetype=as.factor(lambda), color=as.factor(lambda))) +
  geom_point(aes(shape=as.factor(lambda), color=as.factor(lambda))) +
  labs(x = expression("Disclosure Risk:" ~tau), y = "SMAPE") +
  facet_grid(cols=vars(predicate)) +
  scale_colour_grey(start = 0.1, end = 0.8) +
  theme_bw() + 
  theme(plot.margin = margin(0.5, 0.5, 0.5, 0.5, unit = "cm"),
        strip.placement = "outside",
        axis.text = element_text(size = 8),
        panel.grid.major.y = element_blank())
f1 

f2 <- ggplot(data = bi.data, aes(x=risk_2, y=smape, fill=as.factor(lambda))) +
  geom_line(aes(linetype=as.factor(lambda), color=as.factor(lambda))) +
  geom_point(aes(shape=as.factor(lambda), color=as.factor(lambda))) +
  labs(x = expression("Disclosure Risk:" ~phi), y = "SMAPE") +
  facet_grid(cols=vars(predicate)) +
  scale_colour_grey(start = 0.1, end = 0.8) +
  theme_bw() + 
  theme(plot.margin = margin(0.5, 0.5, 0.5, 0.5, unit = "cm"),
        strip.placement = "outside",
        axis.text = element_text(size = 8),
        panel.grid.major.y = element_blank())
f2
