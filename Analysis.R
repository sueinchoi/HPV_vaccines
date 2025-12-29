library(tidyverse)

data <- read.csv('Data/한국 HPV 코호트 자료를 이용한 자_병리검사 (복구됨).CSV', fileEncoding = 'CP949')

sample_data <- head(data, 100)

write.csv(sample_data, 'Data/pathology_sample.csv', row.names = F)
