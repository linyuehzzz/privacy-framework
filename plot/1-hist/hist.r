# Create the data for the chart
H <- c(3,0,1,2,0,2,0)
M <- c("(Single, <15)", "(Single, 15-24)", 
       "(Married, 15-24)", "(Single, 25-64)", 
       "(Married, 25-64)", "(Single, >64)", 
       "(Married, >64)")

par(mar = c(7, 4, 2, 2) + 0.2)

x <- barplot(H, cex.axis=.8, ylim=c(0, 3))
text(cex=.8, x=x-1, y=-0.9, M, xpd=TRUE, srt=45)

