# download datasets from Time Series Data Libraries to csv files

install.packages("devtools")
devtools::install_github("FinYang/tsdl")
library(tsdl)

tsp <- c()
class <- c()
source <- c()
description <- c()
subject <- c()

for (i in 1:648)
{  
    dataset = tsdl[[i]]

    ifill = paste("00", i, sep="")
    ifill = substr(ifill,(nchar(ifill)+1)-3,nchar(ifill))
    filename = paste("dataset_", ifill, ".csv", sep="")

    write.csv(dataset, file=filename) 
    
    tsp <- c(tsp, attributes(dataset)$tsp)
    class <- c(class, attributes(dataset)$class)
    source <- c(source, attributes(dataset)$source)
    description <- c(description, attributes(dataset)$description)
    subject <- c(subject, attributes(dataset)$subject)
}

write.csv(tsp, file="tsp.csv") 
write.csv(class, file="class.csv") 
write.csv(source, file="source.csv") 
write.csv(description, file="description.csv") 
write.csv(subject, file="subject.csv") 


