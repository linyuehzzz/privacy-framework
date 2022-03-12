library(ggplot2)
library(dplyr)

########------------- Read data -------------############

options(scipen=999)
scm.data <- readr::read_csv("set_coverage.csv")
mcm.data <- readr::read_csv("max_coverage.csv")


########------------- Set coverage -------------############

scm.data$predicate <- factor(scm.data$predicate, levels = c("VER", "ER", "R"))

g1 <- ggplot(data = scm.data, aes(x=predicate, y=risk_1, fill=predicate)) +
  geom_bar(stat = "identity", color = "black", width=0.3) +
  labs(y = expression("Disclosure Risk:" ~tau), x = "") +
  scale_fill_grey(start = 0.7, end = 0.9) +
  facet_grid(coverage ~ lambda) +
  theme_bw() + 
  theme(plot.margin = margin(0.5, 0.5, 0.5, 0.5, unit = "cm"),
        strip.placement = "outside",
        axis.text = element_text(size = 8),
        panel.grid.major.y = element_blank())
g1

g2 <- ggplot(data = scm.data, aes(x=predicate, y=smape, fill=predicate)) +
  geom_bar(stat = "identity", color = "black", width=0.3) + 
  labs(y = "SMAPE", x = "") +
  scale_fill_grey(start = 0.7, end = 0.9) +
  facet_grid(coverage ~ lambda) +
  theme_bw() + 
  theme(plot.margin = margin(0.5, 0.5, 0.5, 0.5, unit = "cm"),
        strip.placement = "outside",
        axis.text = element_text(size = 8),
        panel.grid.major.y = element_blank())
g2

########------------- Max coverage -------------############

mcm.data$predicate <- factor(mcm.data$predicate, levels = c("VER", "ER", "R"))

f1 <- ggplot(data = mcm.data, aes(x=u, y=risk_1, fill=predicate)) +
  geom_line(aes(linetype=predicate, color=predicate)) +
  geom_point(aes(shape=predicate, color=predicate)) +
  labs(y = expression("Disclosure Risk:" ~tau), x = "") +
  facet_grid(coverage ~ lambda) +
  scale_colour_grey(start = 0.1, end = 0.8) +
  theme_bw() + 
  theme(plot.margin = margin(0.5, 0.5, 0.5, 0.5, unit = "cm"),
        strip.placement = "outside",
        axis.text = element_text(size = 8),
        panel.grid.major.y = element_blank())
f1 

f2 <- ggplot(data = mcm.data, aes(x=u, y=risk_2, fill=predicate)) +
  geom_line(aes(linetype=predicate, color=predicate)) +
  geom_point(aes(shape=predicate, color=predicate)) +
  labs(y = expression("Disclosure Risk:" ~phi), x = "") +
  facet_grid(coverage ~ lambda) +
  scale_colour_grey(start = 0.1, end = 0.8) +
  theme_bw() + 
  theme(plot.margin = margin(0.5, 0.5, 0.5, 0.5, unit = "cm"),
        strip.placement = "outside",
        axis.text = element_text(size = 8),
        panel.grid.major.y = element_blank())
f2

f3 <- ggplot(data = mcm.data, aes(x=u, y=p_mape, fill=predicate)) +
  geom_line(aes(linetype=predicate, color=predicate)) +
  geom_point(aes(shape=predicate, color=predicate)) +
  labs(y = "p-MAPE", x = "") +
  facet_grid(coverage ~ lambda) +
  scale_colour_grey(start = 0.1, end = 0.8) +
  theme_bw() + 
  theme(plot.margin = margin(0.5, 0.5, 0.5, 0.5, unit = "cm"),
        strip.placement = "outside",
        axis.text = element_text(size = 8),
        panel.grid.major.y = element_blank())
f3

f4 <- ggplot(data = mcm.data, aes(x=u, y=smape, fill=predicate)) +
  geom_line(aes(linetype=predicate, color=predicate)) +
  geom_point(aes(shape=predicate, color=predicate)) +
  labs(y = "SMAPE", x = "") +
  facet_grid(coverage ~ lambda) +
  scale_colour_grey(start = 0.1, end = 0.8) +
  theme_bw() + 
  theme(plot.margin = margin(0.5, 0.5, 0.5, 0.5, unit = "cm"),
        strip.placement = "outside",
        axis.text = element_text(size = 8),
        panel.grid.major.y = element_blank())
f4
