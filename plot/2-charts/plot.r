library(ggplot2)
library(dplyr)


########------------- Read data -------------############

options(scipen=999)
bi.data <- readr::read_csv("bi_objective.csv", show_col_types = FALSE)


########------------- Original -------------############
ori.data <- readr::read_csv("ori.csv", show_col_types = FALSE)
ori.data$predicate <- factor(ori.data$predicate, levels = c("VER", "ER", "R"))
ggplot(ori.data, aes(x=predicate, y=risk, fill=as.factor(risk_type))) + 
  geom_bar(position="dodge", stat="identity") +
  geom_text(aes(label = round(risk, 3)), 
            position = position_dodge(0.9)) +
  theme_bw() + 
  theme(plot.margin = margin(0.5, 0.5, 0.5, 0.5, unit = "cm"),
        strip.placement = "outside",
        axis.text = element_text(size = 8),
        panel.grid.major.y = element_blank()) 


########------------- Time -------------############
bi.data0 <- bi.data %>% filter(predicate == "VER")
bi.data0$lambda <- factor(bi.data0$lambda , levels = c("3", "2", "1"))

ggplot(data = bi.data0, aes(x=q/100, y=cumsum(time), fill=as.factor(lambda))) +
  geom_point(aes(shape=as.factor(lambda), color=as.factor(lambda)), size=1) +
  labs(y="Runtime (s)") +
  scale_colour_manual(values = c("red", "blue", "green")) +
  theme_bw() + 
  theme(plot.margin = margin(0.5, 0.5, 0.5, 0.5, unit = "cm"),
        strip.placement = "outside",
        axis.text = element_text(size = 8),
        panel.grid.major.y = element_blank())



########------------- Objectives -------------############
bi.data0 <- bi.data %>% filter(predicate == "VER")
bi.data0$lambda <- factor(bi.data0$lambda , levels = c("3", "2", "1"))

f0 <- ggplot(data = bi.data0, aes(x=f2, y=f1, fill=as.factor(lambda))) +
  geom_point(aes(shape=as.factor(lambda), color=as.factor(lambda)), size=1) +
  labs(y=expression(italic(f)[1]), x=expression(italic(f)[2])) +
  scale_colour_manual(values = c("red", "blue", "green")) +
  theme_bw() + 
  theme(plot.margin = margin(0.5, 0.5, 0.5, 0.5, unit = "cm"),
        strip.placement = "outside",
        axis.text = element_text(size = 8),
        panel.grid.major.y = element_blank())
f0 

########------------- Metrics -------------############

bi.data$predicate <- factor(bi.data$predicate, levels = c("VER", "ER", "R"))
bi.data$lambda <- factor(bi.data$lambda , levels = c("3", "2", "1"))

f1 <- ggplot(data = bi.data, aes(x=utility, y=risk_1, fill=as.factor(lambda))) +
  geom_point(aes(shape=as.factor(lambda), color=as.factor(lambda)), size=1) +
  labs(x="Data utility (U)", y=expression("Global disclosure risk ("~tau~')')) +
  facet_grid(cols=vars(predicate)) +
  scale_colour_manual(values = c("red", "blue", "green")) +
  theme_bw() + 
  theme(plot.margin = margin(0.5, 0.5, 0.5, 0.5, unit = "cm"),
        strip.placement = "outside",
        axis.text = element_text(size = 8),
        panel.grid.major.y = element_blank())
f1 

f2 <- ggplot(data = bi.data, aes(x=utility, y=risk_2, fill=as.factor(lambda))) +
  geom_point(aes(shape=as.factor(lambda), color=as.factor(lambda)), size=1) +
  labs(x="Data utility (U)", y=expression("Population uniqueness rate ("~phi~')')) +
  facet_grid(cols=vars(predicate)) +
  scale_colour_manual(values = c("red", "blue", "green")) +
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
  geom_point(aes(shape=as.factor(lambda), color=as.factor(lambda)), size=1) +
  labs(y=expression("Global Disclosure Risk ("~tau~')'), x=expression(italic(f)[1])) +
  scale_colour_manual(values = c("red", "blue", "green")) +
  theme_bw() + 
  theme(plot.margin = margin(0.5, 0.5, 0.5, 0.5, unit = "cm"),
        strip.placement = "outside",
        axis.text = element_text(size = 8),
        panel.grid.major.y = element_blank())
f3 

f4 <- ggplot(data = bi.data0, aes(x=f1, y=risk_2, fill=as.factor(lambda))) +
  geom_point(aes(shape=as.factor(lambda), color=as.factor(lambda)), size=1) +
  labs(y=expression("Population Uniqueness Rate ("~phi~')'), x=expression(italic(f)[1])) +
  scale_colour_manual(values = c("red", "blue", "green")) +
  theme_bw() + 
  theme(plot.margin = margin(0.5, 0.5, 0.5, 0.5, unit = "cm"),
        strip.placement = "outside",
        axis.text = element_text(size = 8),
        panel.grid.major.y = element_blank())
f4 

f5 <- ggplot(data = bi.data0, aes(x=f2, y=utility, fill=as.factor(lambda))) +
  geom_point(aes(shape=as.factor(lambda), color=as.factor(lambda)), size=1) +
  labs(y="SME", x=expression(italic(f)[2])) +
  scale_colour_manual(values = c("red", "blue", "green")) +
  theme_bw() + 
  theme(plot.margin = margin(0.5, 0.5, 0.5, 0.5, unit = "cm"),
        strip.placement = "outside",
        axis.text = element_text(size = 8),
        panel.grid.major.y = element_blank())
f5
