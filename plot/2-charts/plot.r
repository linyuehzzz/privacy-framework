library(ggplot2)
library(dplyr)


########------------- Read data -------------############

options(scipen=999)
bi.data <- readr::read_csv("bi_objective.csv", show_col_types = FALSE)


########------------- Objectives -------------############
bi.data0 <- bi.data %>% filter(predicate == "VER")
# bi.data0$f2[bi.data0$lambda == 1] <- as.numeric(bi.data0[bi.data0$lambda == 1, ]$f2) * 100 / 740.725589257187
# bi.data0$f2[bi.data0$lambda == 2] <- as.numeric(bi.data0[bi.data0$lambda == 2, ]$f2) * 100 / 1313.37213444108
# bi.data0$f2[bi.data0$lambda == 3] <- as.numeric(bi.data0[bi.data0$lambda == 3, ]$f2) * 100 / 2839.76182586111
bi.data0$lambda <- factor(bi.data0$lambda , levels = c("3", "2", "1"))

f0 <- ggplot(data = bi.data0, aes(x=f2, y=f1, fill=as.factor(lambda))) +
  # geom_vline(xintercept=25,linetype=3) +
  # geom_vline(xintercept=50,linetype=3) +
  # geom_vline(xintercept=75,linetype=3) +
  # geom_vline(xintercept=100,linetype=3) +
  #geom_line(aes(linetype=as.factor(lambda), color=as.factor(lambda))) +
  geom_line(linetype=3, aes(color=as.factor(lambda))) +
  geom_point(aes(shape=as.factor(lambda), color=as.factor(lambda)), size=1) +
  #labs(y=expression(italic(f)[1]), x=expression(italic(f)[2]/'max('~italic(f)[2]~')� 100 (%)')) +
  labs(y=expression(italic(f)[1]), x=expression(italic(f)[2])) +
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
  geom_line(aes(linetype=as.factor(lambda), color=as.factor(lambda))) +
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
  geom_line(aes(linetype=as.factor(lambda), color=as.factor(lambda))) +
  #geom_point(aes(shape=as.factor(lambda), color=as.factor(lambda)), size=1) +
  labs(x="SME", y=expression("Population Uniqueness Rate ("~phi~')')) +
  facet_grid(cols=vars(predicate)) +
  scale_colour_manual(values = c("red", "blue", "green")) +
  #scale_colour_grey(start = 0.1, end = 0.8) +
  theme_bw() + 
  theme(plot.margin = margin(0.5, 0.5, 0.5, 0.5, unit = "cm"),
        strip.placement = "outside",
        axis.text = element_text(size = 8),
        panel.grid.major.y = element_blank())
f2


########------------- Relations -------------############
bi.data0 <- bi.data %>% filter(predicate == "VER")
bi.data0$lambda <- factor(bi.data0$lambda , levels = c("3", "2", "1"))

f3 <- ggplot(data = bi.data0, aes(x=f1, y=risk_1, fill=as.factor(lambda))) +
  #geom_line(aes(linetype=as.factor(lambda), color=as.factor(lambda))) +
  geom_line(linetype=3, aes(color=as.factor(lambda))) +
  geom_point(aes(shape=as.factor(lambda), color=as.factor(lambda)), size=1) +
  labs(y=expression("Global Disclosure Risk ("~tau~')'), x=expression(italic(f)[1])) +
  scale_colour_manual(values = c("red", "blue", "green")) +
  #scale_colour_grey(start = 0.1, end = 0.8) +
  theme_bw() + 
  theme(plot.margin = margin(0.5, 0.5, 0.5, 0.5, unit = "cm"),
        strip.placement = "outside",
        axis.text = element_text(size = 8),
        panel.grid.major.y = element_blank())
f3 

f4 <- ggplot(data = bi.data0, aes(x=f1, y=risk_2, fill=as.factor(lambda))) +
  #geom_line(aes(linetype=as.factor(lambda), color=as.factor(lambda))) +
  geom_line(linetype=3, aes(color=as.factor(lambda))) +
  geom_point(aes(shape=as.factor(lambda), color=as.factor(lambda)), size=1) +
  labs(y=expression("Population Uniqueness Rate ("~phi~')'), x=expression(italic(f)[1])) +
  scale_colour_manual(values = c("red", "blue", "green")) +
  #scale_colour_grey(start = 0.1, end = 0.8) +
  theme_bw() + 
  theme(plot.margin = margin(0.5, 0.5, 0.5, 0.5, unit = "cm"),
        strip.placement = "outside",
        axis.text = element_text(size = 8),
        panel.grid.major.y = element_blank())
f4 

f5 <- ggplot(data = bi.data0, aes(x=f2, y=smape, fill=as.factor(lambda))) +
  #geom_line(aes(linetype=as.factor(lambda), color=as.factor(lambda))) +
  geom_line(linetype=3, aes(color=as.factor(lambda))) +
  geom_point(aes(shape=as.factor(lambda), color=as.factor(lambda)), size=1) +
  labs(y="SME", x=expression(italic(f)[2])) +
  scale_colour_manual(values = c("red", "blue", "green")) +
  #scale_colour_grey(start = 0.1, end = 0.8) +
  theme_bw() + 
  theme(plot.margin = margin(0.5, 0.5, 0.5, 0.5, unit = "cm"),
        strip.placement = "outside",
        axis.text = element_text(size = 8),
        panel.grid.major.y = element_blank())
f5
