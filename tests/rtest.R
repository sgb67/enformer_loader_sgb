# Test R code to check connection to a node
print("Hello, R is running successfully!")

# Check the R version
R.version.string

# Check the current working directory
getwd()
# Generate a simple plot
x <- rnorm(100)
y <- rnorm(100)
plot(x, y, main="Scatter plot of random points", xlab="X-axis", ylab="Y-axis")
quartz()  # Opens a new window for the plot (on Linux or macOS)
plot(1:10, 1:10)
