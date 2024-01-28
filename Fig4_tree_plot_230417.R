# if (!require("BiocManager", quietly = TRUE))
#   install.packages("BiocManager")
# BiocManager::install("ggtree")

rm(list = ls())
library(ggtree)
library("treeio")
library(ggsci)
library(scales)


mypal2 = pal_jco(alpha = 1)(9)
show_col(mypal2)
col2 = c("#868686FF","#EFC000FF","#CD534CFF","#0072B5FF")



tree <- read.tree("false_pre_tree_input.nwk.txt")
tree
group_info <- data.frame(label=c("Fish1","Pig","Bovine1","Bovine2","Fish2","Human"),Truth=c("Fish","Pig","Bovine","Bovine","Fish","Human"),
                         Predition=c("Human","Human","Human","Human","Human","Pig"))
tree1 <- full_join(tree,group_info,by="label")
as_tibble(tree1)
d <- data.frame(node=c(1,2,3,4,5,6), type=c("Human","Human","Human","Human","Pig","Human"))

ggtree(tree1, branch.length="none")+
  geom_tiplab(size=5, color=c("#0072B5FF","#CD534CFF","#CD534CFF","#0072B5FF","black","#EFC000FF")) +
  geom_tippoint(color= c("#0072B5FF","#CD534CFF","#CD534CFF","#0072B5FF","black","#EFC000FF"), alpha=1/2, size=3)+
  #geom_nodepoint(color="#b5e521", alpha=1/4, size=10) +
  xlim(NA, 8)
  #geom_hilight(data=d, aes(node=node, fill=type),type = "roundrect")

ggsave("false_pre_phy_tree.pdf",width = 5,height =5,dpi = 300)




