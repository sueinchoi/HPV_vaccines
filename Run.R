# Run

source("scripts/extract_pathology_outcomes.R")
result <- main()

# 또는 개별 함수 사용
df <- load_pathology_data("Data/pathology.csv")
df <- extract_outcomes(df)
summary_df <- get_patient_outcomes_summary(df)

summary_df %>% head(10) %>% view()
