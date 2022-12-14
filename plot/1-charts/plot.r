library(ggplot2)
library(dplyr)

########------------- Read data -------------############

options(scipen=999)
scm1.data <- readr::read_csv("set_coverage_franklin.csv",show_col_types = FALSE)
mcm1.data <- readr::read_csv("max_coverage_franklin.csv",show_col_types = FALSE)
scm2.data <- readr::read_csv("set_coverage_guernsey.csv",show_col_types = FALSE)
mcm2.data <- readr::read_csv("max_coverage_guernsey.csv",show_col_types = FALSE)

########------------- Set coverage -------------############

scm1.data$predicate <- factor(scm1.data$predicate, levels = c("VER", "ER", "R"))
scm1.data$new <- paste(scm1.data$predicate, '_', scm1.data$coverage)
scm2.data$predicate <- factor(scm2.data$predicate, levels = c("VER", "ER", "R"))
scm2.data$new <- paste(scm2.data$predicate, '_', scm2.data$coverage)

f1 <- ggplot(data = scm1.data, aes(x=factor(lambda), y=risk_1, fill=new)) +
  geom_point(aes(shape=new, color=new), size=2) +
  labs(y = expression("Disclosure Risk:" ~tau), x = "") +
  ylim(0, 0.1) +
  scale_shape_manual(values=c(3, 3, 16, 16, 17, 17)) +
  scale_colour_manual(values=c("red", "green", "red", "green", "red", "green")) +
  theme_bw() + 
  theme(plot.margin = margin(0.5, 0.5, 0.5, 0.5, unit = "cm"),
        strip.placement = "outside",
        axis.text = element_text(size = 8),
        panel.grid.major.y = element_blank())
f1

g1 <- ggplot(data = scm2.data, aes(x=factor(lambda), y=risk_1, fill=new)) +
  geom_point(aes(shape=new, color=new), size=2) +
  labs(y = expression("Disclosure Risk:" ~tau), x = "") +
  ylim(0, 0.1) +
  scale_shape_manual(values=c(3, 3, 16, 16, 17, 17)) +
  scale_colour_manual(values=c("red", "green", "red", "green", "red", "green")) +
  theme_bw() + 
  theme(plot.margin = margin(0.5, 0.5, 0.5, 0.5, unit = "cm"),
        strip.placement = "outside",
        axis.text = element_text(size = 8),
        panel.grid.major.y = element_blank())
g1

f2 <- ggplot(data = scm1.data, aes(x=factor(lambda), y=smape, fill=new)) +
  geom_point(aes(shape=new, color=new), size=2) + 
  labs(y = "SME", x = "") +
  ylim(0, 0.4) +
  scale_shape_manual(values=c(3, 3, 16, 16, 17, 17)) +
  scale_colour_manual(values=c("red", "green", "red", "green", "red", "green")) +
  theme_bw() + 
  theme(plot.margin = margin(0.5, 0.5, 0.5, 0.5, unit = "cm"),
        strip.placement = "outside",
        axis.text = element_text(size = 8),
        panel.grid.major.y = element_blank())
f2

g2 <- ggplot(data = scm2.data, aes(x=factor(lambda), y=smape, fill=new)) +
  geom_point(aes(shape=new, color=new), size=2) + 
  labs(y = "SME", x = "") +
  ylim(0, 0.4) +
  scale_shape_manual(values=c(3, 3, 16, 16, 17, 17)) +
  scale_colour_manual(values=c("red", "green", "red", "green", "red", "green")) +
  theme_bw() + 
  theme(plot.margin = margin(0.5, 0.5, 0.5, 0.5, unit = "cm"),
        strip.placement = "outside",
        axis.text = element_text(size = 8),
        panel.grid.major.y = element_blank())
g2

########------------- Max coverage -------------############

mcm1.data$predicate <- factor(mcm1.data$predicate, levels = c("VER", "ER", "R"))
mcm1.data$new <- paste(mcm1.data$predicate, '_', mcm1.data$coverage)
mcm2.data$predicate <- factor(mcm2.data$predicate, levels = c("VER", "ER", "R"))
mcm2.data$new <- paste(mcm2.data$predicate, '_', mcm2.data$coverage)

f3 <- ggplot(data = mcm1.data, aes(x=u, y=risk_1, fill=new)) +
  geom_point(aes(shape=new, color=new)) +
  labs(y = expression("Disclosure Risk:" ~tau), x = "") +
  ylim(0, 0.25) +
  facet_grid(cols=vars(lambda)) +
  scale_shape_manual(values=c(3, 3, 16, 16, 17, 17)) +
  scale_colour_manual(values=c("red", "green", "red", "green", "red", "green")) +
  theme_bw() + 
  theme(plot.margin = margin(0.5, 0.5, 0.5, 0.5, unit = "cm"),
        strip.placement = "outside",
        axis.text = element_text(size = 8),
        panel.grid.major.y = element_blank())
f3 

g3 <- ggplot(data = mcm2.data, aes(x=u, y=risk_1, fill=new)) +
  geom_point(aes(shape=new, color=new)) +
  labs(y = expression("Disclosure Risk:" ~tau), x = "") +
  ylim(0, 0.25) +
  facet_grid(cols=vars(lambda)) +
  scale_shape_manual(values=c(3, 3, 16, 16, 17, 17)) +
  scale_colour_manual(values=c("red", "green", "red", "green", "red", "green")) +
  theme_bw() + 
  theme(plot.margin = margin(0.5, 0.5, 0.5, 0.5, unit = "cm"),
        strip.placement = "outside",
        axis.text = element_text(size = 8),
        panel.grid.major.y = element_blank())
g3

f4 <- ggplot(data = mcm1.data, aes(x=u, y=risk_2, fill=new)) +
  geom_point(aes(shape=new, color=new)) +
  labs(y = expression("Disclosure Risk:" ~phi), x = "") +
  ylim(0, 0.15) +
  facet_grid(cols=vars(lambda)) +
  scale_shape_manual(values=c(3, 3, 16, 16, 17, 17)) +
  scale_colour_manual(values=c("red", "green", "red", "green", "red", "green")) +
  theme_bw() + 
  theme(plot.margin = margin(0.5, 0.5, 0.5, 0.5, unit = "cm"),
        strip.placement = "outside",
        axis.text = element_text(size = 8),
        panel.grid.major.y = element_blank())
f4 

g4 <- ggplot(data = mcm2.data, aes(x=u, y=risk_2, fill=new)) +
  geom_point(aes(shape=new, color=new)) +
  labs(y = expression("Disclosure Risk:" ~phi), x = "") +
  ylim(0, 0.15) +
  facet_grid(cols=vars(lambda)) +
  scale_shape_manual(values=c(3, 3, 16, 16, 17, 17)) +
  scale_colour_manual(values=c("red", "green", "red", "green", "red", "green")) +
  theme_bw() + 
  theme(plot.margin = margin(0.5, 0.5, 0.5, 0.5, unit = "cm"),
        strip.placement = "outside",
        axis.text = element_text(size = 8),
        panel.grid.major.y = element_blank())
g4 

# f5 <- ggplot(data = mcm1.data, aes(x=u, y=p_mape, fill=predicate)) +
#   geom_line(aes(linetype=predicate, color=predicate)) +
#   geom_point(aes(shape=predicate, color=predicate)) +
#   labs(y = "p-MAPE", x = "") +
#   facet_grid(coverage ~ lambda) +
#   scale_colour_grey(start = 0.1, end = 0.8) +
#   theme_bw() + 
#   theme(plot.margin = margin(0.5, 0.5, 0.5, 0.5, unit = "cm"),
#         strip.placement = "outside",
#         axis.text = element_text(size = 8),
#         panel.grid.major.y = element_blank())
# f5

f6 <- ggplot(data = mcm1.data, aes(x=u, y=smape, fill=new)) +
  geom_point(aes(shape=new, color=new)) +
  labs(y = "SME") +
  ylim(0, 0.3) +
  facet_grid(cols=vars(lambda)) +
  scale_shape_manual(values=c(3, 3, 16, 16, 17, 17)) +
  scale_colour_manual(values=c("red", "green", "red", "green", "red", "green")) +
  theme_bw() + 
  theme(plot.margin = margin(0.5, 0.5, 0.5, 0.5, unit = "cm"),
        strip.placement = "outside",
        axis.text = element_text(size = 8),
        panel.grid.major.y = element_blank())
f6

g6 <- ggplot(data = mcm2.data, aes(x=u, y=smape, fill=new)) +
  geom_point(aes(shape=new, color=new)) +
  labs(y = "SME") +
  ylim(0, 0.3) +
  facet_grid(cols=vars(lambda)) +
  scale_shape_manual(values=c(3, 3, 16, 16, 17, 17)) +
  scale_colour_manual(values=c("red", "green", "red", "green", "red", "green")) +
  theme_bw() + 
  theme(plot.margin = margin(0.5, 0.5, 0.5, 0.5, unit = "cm"),
        strip.placement = "outside",
        axis.text = element_text(size = 8),
        panel.grid.major.y = element_blank())
g6
