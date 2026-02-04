library(lubridate)
library(dplyr)
library(tidyr)
library(ggplot2)

bench_result <- read.csv("././graphMatching/data/benchmark.tsv", sep="\t")

for (drp in c("Alice","Both")){

  for(f in unique(bench_result$AliceAlgo)){

    full_info <- bench_result %>% filter(DropFrom==drp, AliceAlgo == f) %>%
      mutate(DataName = paste(as.numeric(gsub("\\D", "", Data))*1000,"Fake Names")) %>%
      mutate(DataName = replace(DataName, Data=="./data/titanic_full.tsv", "Titanic")) %>%
      mutate(DataName = replace(DataName, Data=="./data/euro_full.tsv", "Euro")) %>%
      mutate(DataName = replace(DataName, Data=="./data/ncvoter.tsv", "NCVoter")) %>%
      select(Overlap, Data, success_rate, correct, n_alice, n_eve, DataName, DropFrom, AliceAlgo, RegWS)


    if(drp=="Alice"){
      full_info$DiceOverlap = (2*full_info$n_alice)/(full_info$n_eve+full_info$n_alice)
    }else{
      full_info$DiceOverlap = round((2*(round(full_info$n_eve*full_info$Overlap)))/(full_info$n_eve+full_info$n_alice), 2)
    }

    DataNameord <- factor(full_info$DataName, levels=c('Titanic', 'Euro', 'NCVoter',
                                                       '1000 Fake Names', '2000 Fake Names', '5000 Fake Names',
                                                       '10000 Fake Names', '20000 Fake Names', '50000 Fake Names'))

    full_info$DataName <- DataNameord

    fullinfo_successplot <- full_info %>% ggplot(aes(x=Overlap, y=success_rate, group=DataName, color=DataName)) +
      geom_line(linewidth=1) + geom_point() +
      xlim(0,1) + ylim(0,1) +
      labs(y="Success Rate", color="Dataset")+
      scale_color_manual(values=c('Titanic'='#33CCCC', 'Euro'='#FF66FF', 'NCVoter' = '#990033',
                                  '1000 Fake Names' = '#3366CC', '2000 Fake Names' = '#CCCC33',
                                  '5000 Fake Names'= '#00CC00', '10000 Fake Names'='#FF9900',
                                  '20000 Fake Names'='#FF3300', '50000 Fake Names'='#666666'))+
      theme_minimal()

    fullinfo_successplot


    fname <- tolower(paste0("success_",full_info$AliceAlgo[1], "_drop_", full_info$DropFrom[1],"_full.eps"))

    ggsave(paste0("./plots/",fname), plot=fullinfo_successplot, device = "eps",
           width = 2200, height = 1000, units = "px")

  }
}
