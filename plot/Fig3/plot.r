library(ggplot2)
library(dplyr)

options(scipen=999)
scm.data <- readr::read_csv("set_coverage.csv")
mcm.data <- readr::read_csv("max_coverage.csv")

g2 <- ggplot(data = scm.data, aes(x=p_mape, y=risk_1, color=predicate)) +
  geom_point() + 
  labs(y = "Disclosure risk", x = "") +
  scale_fill_grey(start = 0.7, end = 0.9) +
  facet_grid(coverage ~ k) +
  theme_bw()
g2

g1 <- ggplot(data = mcm.data, aes(x=u, y=risk_1, fill=predicate)) +
  geom_line(aes(linetype=predicate, color=predicate)) +
  geom_point(aes(shape=predicate, color=predicate)) +
  labs(y = "Disclosure risk", x = "") +
  facet_grid(coverage ~ k) +
  theme_bw()
g1