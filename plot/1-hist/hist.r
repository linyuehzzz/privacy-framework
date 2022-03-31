# Create the data for the chart
H <- c(0,2,0,3,1,0,0,2)
M <- c("(Male, N/A)", "(Female, N/A)", 
       "(Male, 1997)", "(Female, 1997)", 
       "(Male, 2003)", "(Female, 2003)", 
       "(Male, 2011)", "(Female, 2011)")

par(mar = c(7, 4, 2, 2) + 0.2)

x <- barplot(H, cex.axis=.8, ylim=c(0, 3))
text(cex=.8, x=x-1, y=-0.9, M, xpd=TRUE, srt=45)

