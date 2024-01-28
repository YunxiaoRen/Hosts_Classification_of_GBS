# Env setting
options("repos"= c(CRAN="https://mirrors.tuna.tsinghua.edu.cn/CRAN/"))
options(BioC_mirror="http://mirrors.ustc.edu.cn/bioc/")
#install circlize packages
#install.packages("circlize")

#load packages
library(circlize)
library(ggsci)
library(parallel)
library(scales)
Sys.setenv(LANGUAGE = "en") #show error message in English
options(stringsAsFactors = FALSE) #forbid chr were changed into factor


# function for plot host_sero
genecor_circleplot_Sero <- function(x){
  Corr <- data.frame(rbind(data.frame(Gene=x[,1], Correlation=x[,3]), 
                           data.frame(Gene=x[,2], Correlation=x[,3])), stringsAsFactors = F)      
  Corr$Index <- seq(1,nrow(Corr),1) #Record the original order of the gene to the Index column
  Corr <- Corr[order(Corr[,1]),] #order based on genes
  corrsp <- split(Corr,Corr$Gene)
  corrspe <- lapply(corrsp, function(x){x$Gene_Start<-0
  
  #calculate the sum of correlation coefficients for each gene in turn, it is used as the gene termination site
  if (nrow(x)==1){x$Gene_End<-1}else{
    x$Gene_End<-sum(abs(x$Correlation))} 
  x})
  GeneID <- do.call(rbind,corrspe)
  GeneID <- GeneID[!duplicated(GeneID$Gene),]
  
  #color setting for gene
  #mycol <- pal_d3("category20c")(20)
  mycol <- pal_d3("category20")(20)
  #mycol <- pal_npg("nrc")(10)
  n <- nrow(GeneID)
  GeneID$Color <- mycol[c(11,1:9)]
  
  # The width of the connecting line is the absolute value of the correlation coefficient
  Corr[,2] <- abs(Corr[,2])
  corrsl <- split(Corr,Corr$Gene)
  aaaaa <- c()
  corrspl <- lapply(corrsl,function(x){nn<-nrow(x)
  for (i in 1:nn){
    aaaaa[1] <- 0
    aaaaa[i+1] <- x$Correlation[i]+aaaaa[i]}
  bbbbb <- data.frame(V4=aaaaa[1:nn],V5=aaaaa[2:(nn+1)])
  bbbbbb <- cbind(x,bbbbb)
  bbbbbb
  })
  Corr <- do.call(rbind,corrspl)
  
  #according to Index column,returning genes to their original order
  Corr <- Corr[order(Corr$Index),]
  
  #V4 is start site,V5 is end site
  #save into Links,start_1 and end_1 related to Gene_1,start_2 and end_2 related Gene_2
  x$start_1 <- Corr$V4[1:(nrow(Corr)/2)]
  x$end_1 <- Corr$V5[1:(nrow(Corr)/2)]
  x$start_2 <- Corr$V4[(nrow(Corr)/2 + 1):nrow(Corr)]
  x$end_2 <- Corr$V5[(nrow(Corr)/2 + 1):nrow(Corr)]
  
  #link color
  mycol2 <- c("#BCBDDCFF","#DBDB8DFF","#C6DBEFFF","#FDAF91FF","#9EDAE5FF","#C7C7C7FF","#31A354FF","#0000CCFF","#D0DFE6FF","#E762D7FF","#CC0000FF","#FFFF00FF",
              "#FF9896FF","#0000CCFF","#E762D7FF","#CC0000FF","#C49C94FF","#FFFF00FF","#FFFF00FF","#FFFF00FF","#FFFF00FF","#C49C94FF","#FFFF00FF","#FFFF00FF")
  
 
  n2 = nrow(x)
  x$color = mycol2[1:n2]
  
  #plot area setting
  #par(mar=rep(0,4))
  circos.clear()
  circos.par(start.degree = 90, #start,counter-clockwise order
             gap.degree = 5, #Space size between gene bars
             track.margin = c(0,0.23), #The larger the value, the smaller the interval between the gene and the linkage
             cell.padding = c(0,0,0,0)
  )
  circos.initialize(factors = GeneID$Gene,
                    xlim = cbind(GeneID$Gene_Start, GeneID$Gene_End))
  
  #plot genes first
  circos.trackPlotRegion(ylim = c(0, 1), factors = GeneID$Gene, 
                         track.height = 0.05, 
                         panel.fun = function(x, y) {
                           name = get.cell.meta.data("sector.index") 
                           i = get.cell.meta.data("sector.numeric.index") 
                           xlim = get.cell.meta.data("xlim")
                           ylim = get.cell.meta.data("ylim")
                           circos.text(x = mean(xlim), y = 1,
                                       labels = name,
                                       cex = 1, 
                                       niceFacing = TRUE, 
                                       facing = "bending",
                                       adj = c(0.5, -2.8), 
                                       font = 2 
                           )
                           circos.rect(xleft = xlim[1], 
                                       ybottom = ylim[1],
                                       xright = xlim[2], 
                                       ytop = ylim[2],
                                       col = GeneID$Color[i],
                                       border = GeneID$Color[i])
                           
                           circos.axis(labels.cex = 0.7, 
                                       direction = "outside"
                           )})
  
  #Plot link
  for(i in 1:nrow(x)){
    circos.link(sector.index1 = x[,1][i], 
                point1 = c(x[i, 4], x[i, 5]),
                sector.index2 = x[,2][i], 
                point2 = c(x[i, 6], x[i, 7]),
                col = x$color[i],
                border = FALSE, 
                rou = 0.7
    )}
}

# Plot
## Serotype
data1 <- read.table("./Number_host_Serotype.txt", header = TRUE, sep = "\t",as.is = T)
data1
data1$Hosts <- factor(data1$Hosts,levels = c("Human","Bovine","Fish","Pig"))
data1$Serotype <- factor(data1$Serotype,levels = c("Ia","Ib","II","III","IV","V"))
data1
genecor_circleplot_Sero(data1)

# ?????????PDF??????
pdf("Host_Serotype_correlation.pdf", width = 5, height = 5)
genecor_circleplot_Sero(data1)
dev.off()

