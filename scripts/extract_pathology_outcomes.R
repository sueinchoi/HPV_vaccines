#' 병리 검사 데이터에서 주요 결과 변수 추출
#'
#' 연구 계획서 7.1 주요 결과변수 (Primary Outcomes):
#' 1. 병변 재발: 조직검사로 확인된 HSIL/CIN3 이상 병변 재발
#' 2. 새로운 고위험 HPV 감염: Index date 이후 HPV 양성 전환
#'    - 고위험 HPV 유형: 16, 18, 31, 33, 45, 52, 58, 35, 39, 51, 56, 59, 66, 68

library(dplyr)
library(stringr)
library(readr)

# 고위험 HPV 유형 정의
HIGH_RISK_HPV_TYPES <- c(16, 18, 31, 33, 45, 52, 58, 35, 39, 51, 56, 59, 66, 68)


#' 병리 검사 데이터 로드
#'
#' @param file_path CSV 파일 경로
#' @return data.frame
load_pathology_data <- function(file_path) {
  df <- read_csv(file_path, locale = locale(encoding = "UTF-8"), show_col_types = FALSE)

  # 날짜 컬럼 변환
  date_columns <- c("처방일자", "실시일자", "판독일자")
  for (col in date_columns) {
    if (col %in% names(df)) {
      df[[col]] <- as.Date(as.character(df[[col]]), format = "%Y%m%d")
    }
  }

  return(df)
}


#' 판독결과에서 HSIL/CIN3 이상 병변 재발 여부 확인
#'
#' @param result_text 판독결과 텍스트
#' @return list with is_hsil_cin3_or_higher, detected_lesion, severity_level
detect_hsil_cin3_recurrence <- function(result_text) {
  # NA 처리
  if (is.na(result_text) || result_text == "") {
    return(list(
      is_hsil_cin3_or_higher = FALSE,
      detected_lesion = NA_character_,
      severity_level = NA_character_
    ))
  }

  result_text <- toupper(as.character(result_text))

  # 1. Carcinoma (가장 심각) - 자궁경부암
  carcinoma_patterns <- c(
    "SQUAMOUS\\s*CELL\\s*CARCINOMA",
    "ADENOCARCINOMA",
    "CERVICAL\\s*CANCER",
    "INVASIVE\\s*CARCINOMA",
    "CARCINOMA\\s*IN\\s*SITU",
    "\\bCIS\\b"
  )

  for (pattern in carcinoma_patterns) {
    match <- str_extract(result_text, pattern)
    if (!is.na(match)) {
      return(list(
        is_hsil_cin3_or_higher = TRUE,
        detected_lesion = match,
        severity_level = "Carcinoma"
      ))
    }
  }

  # 2. CIN3 / CIN III / CINIII 패턴
  cin3_patterns <- c(
    "CIN\\s*3\\b",
    "CIN\\s*III\\b",
    "CINIII\\b",
    "CIN-3\\b",
    "CERVICAL\\s*INTRAEPITHELIAL\\s*NEOPLASIA\\s*3",
    "CERVICAL\\s*INTRAEPITHELIAL\\s*NEOPLASM[A]?\\s*\\(?CIN\\)?\\s*3"
  )

  for (pattern in cin3_patterns) {
    match <- str_extract(result_text, pattern)
    if (!is.na(match)) {
      return(list(
        is_hsil_cin3_or_higher = TRUE,
        detected_lesion = match,
        severity_level = "CIN3"
      ))
    }
  }

  # 3. HSIL (High-grade Squamous Intraepithelial Lesion)
  hsil_patterns <- c(
    "\\bHSIL\\b",
    "\\bH-SIL\\b",
    "HIGH[\\s-]*GRADE\\s*SQUAMOUS\\s*INTRAEPITHELIAL\\s*LESION",
    "HIGH[\\s-]*GRADE\\s*SIL\\b"
  )

  for (pattern in hsil_patterns) {
    match <- str_extract(result_text, pattern)
    if (!is.na(match)) {
      return(list(
        is_hsil_cin3_or_higher = TRUE,
        detected_lesion = match,
        severity_level = "HSIL"
      ))
    }
  }

  # 4. CIN2 / CIN II / CINII
  cin2_patterns <- c(
    "CIN\\s*2\\b",
    "CIN\\s*II\\b",
    "CINII\\b",
    "CIN-2\\b",
    "CERVICAL\\s*INTRAEPITHELIAL\\s*NEOPLASIA\\s*2",
    "CIN\\s*2/3\\b",
    "CIN\\s*II/III\\b"
  )

  for (pattern in cin2_patterns) {
    match <- str_extract(result_text, pattern)
    if (!is.na(match)) {
      return(list(
        is_hsil_cin3_or_higher = TRUE,
        detected_lesion = match,
        severity_level = "CIN2"
      ))
    }
  }

  return(list(
    is_hsil_cin3_or_higher = FALSE,
    detected_lesion = NA_character_,
    severity_level = NA_character_
  ))
}


#' 판독결과에서 고위험 HPV 감염 여부 확인
#'
#' @param result_text 판독결과 텍스트
#' @return list with is_high_risk_hpv_positive, detected_hpv_types, hpv_result_detail
detect_high_risk_hpv <- function(result_text) {
  # NA 처리
  if (is.na(result_text) || result_text == "") {
    return(list(
      is_high_risk_hpv_positive = FALSE,
      detected_hpv_types = list(),
      hpv_result_detail = NA_character_
    ))
  }

  result_text_upper <- toupper(as.character(result_text))
  result_text_orig <- as.character(result_text)

  detected_types <- c()
  result_detail <- NA_character_

  # 1. 개별 HPV 유형 양성 확인
  for (hpv_type in HIGH_RISK_HPV_TYPES) {
    patterns <- c(
      sprintf("POSITIVE\\s*\\(\\s*%d\\s*\\)", hpv_type),
      sprintf("HPV\\s*%d\\s*:\\s*POSITIVE", hpv_type),
      sprintf("TYPE\\s*%d\\s*:\\s*POSITIVE", hpv_type)
    )

    for (pattern in patterns) {
      if (str_detect(result_text_upper, pattern)) {
        if (!(hpv_type %in% detected_types)) {
          detected_types <- c(detected_types, hpv_type)
        }
      }
    }
  }

  # 2. "Positive(other)" - 16, 18 외 고위험 유형
  other_patterns <- c(
    "POSITIVE\\s*\\(\\s*OTHER\\s*(?:TYPE)?\\s*\\)",
    "POSITIVE\\s*\\(\\s*OTHER\\s*\\)"
  )

  for (pattern in other_patterns) {
    if (str_detect(result_text_upper, pattern)) {
      result_detail <- "Other high-risk types detected"
      other_types <- HIGH_RISK_HPV_TYPES[!(HIGH_RISK_HPV_TYPES %in% c(16, 18))]
      for (t in other_types) {
        if (!(t %in% detected_types)) {
          detected_types <- c(detected_types, t)
        }
      }
    }
  }

  # 3. Pool 그룹 양성 확인 (P1, P2, P3)
  pool_types <- list(
    P1 = c(33, 58),
    P2 = c(56, 59, 66),
    P3 = c(35, 39, 68)
  )

  for (pool in names(pool_types)) {
    pattern <- sprintf("POSITIVE\\s*\\(\\s*%s\\s*\\)", pool)
    if (str_detect(result_text_upper, pattern)) {
      for (t in pool_types[[pool]]) {
        if (!(t %in% detected_types)) {
          detected_types <- c(detected_types, t)
        }
      }
    }
  }

  # 4. 일반적인 고위험 HPV 양성 표시
  general_positive_patterns <- c(
    "HPV\\s*:\\s*POSITIVE\\s*\\(\\s*HIGH[\\s-]*RISK\\s*\\)",
    "HIGH[\\s-]*RISK\\s*HPV\\s*:\\s*POSITIVE",
    "HPV\\s*HIGH[\\s-]*RISK\\s*:\\s*POSITIVE",
    "HIGH[\\s-]*RISK\\s*:\\s*POSITIVE"
  )

  for (pattern in general_positive_patterns) {
    if (str_detect(result_text_upper, pattern)) {
      if (length(detected_types) == 0) {
        result_detail <- "High-risk HPV positive (unspecified type)"
        detected_types <- c("high_risk_unspecified")
      }
    }
  }

  # 5. HPV DNA Chip 결과에서 유형 추출
  if (str_detect(result_text_upper, "HPV\\s*DNA\\s*CHIP.*?POSITIVE")) {
    type_pattern <- paste0("\\b(", paste(HIGH_RISK_HPV_TYPES, collapse = "|"), ")\\b")
    type_matches <- str_extract_all(result_text_orig, type_pattern)[[1]]

    if (str_detect(result_text_upper, "POSITIVE") && length(type_matches) > 0) {
      for (t in type_matches) {
        t_int <- as.integer(t)
        if (t_int %in% HIGH_RISK_HPV_TYPES && !(t_int %in% detected_types)) {
          detected_types <- c(detected_types, t_int)
        }
      }
    }
  }

  # 6. Negative 결과 명시적 확인
  negative_patterns <- c(
    "HPV\\s*GENOTYPING\\s*REAL[\\s-]*TIME\\s*PCR\\s*:\\s*NEGATIVE",
    "HPV\\s*:\\s*NEGATIVE",
    "\\[\\[RESULT\\]\\]\\s*NEGATIVE",
    "HIGH[\\s-]*RISK\\s*HPV\\s*:\\s*NEGATIVE"
  )

  is_explicitly_negative <- any(sapply(negative_patterns, function(p) str_detect(result_text_upper, p)))

  if (is_explicitly_negative && length(detected_types) == 0) {
    return(list(
      is_high_risk_hpv_positive = FALSE,
      detected_hpv_types = list(),
      hpv_result_detail = "Negative"
    ))
  }

  # 결과 정리
  is_positive <- length(detected_types) > 0

  # 숫자 유형만 필터링 및 정렬
  numeric_types <- sort(detected_types[sapply(detected_types, is.numeric)])

  return(list(
    is_high_risk_hpv_positive = is_positive,
    detected_hpv_types = if (length(numeric_types) > 0) as.list(numeric_types) else as.list(detected_types),
    hpv_result_detail = result_detail
  ))
}


#' 전체 병리 데이터에서 결과 변수 추출
#'
#' @param df 병리 데이터 data.frame
#' @return data.frame with outcome columns added
extract_outcomes <- function(df) {
  # 병변 재발 (HSIL/CIN3+) 추출
  lesion_results <- lapply(df$판독결과, detect_hsil_cin3_recurrence)

  df$is_hsil_cin3_or_higher <- sapply(lesion_results, function(x) x$is_hsil_cin3_or_higher)
  df$detected_lesion <- sapply(lesion_results, function(x) x$detected_lesion)
  df$severity_level <- sapply(lesion_results, function(x) x$severity_level)

  # 고위험 HPV 감염 추출
  hpv_results <- lapply(df$판독결과, detect_high_risk_hpv)

  df$is_high_risk_hpv_positive <- sapply(hpv_results, function(x) x$is_high_risk_hpv_positive)
  df$detected_hpv_types <- sapply(hpv_results, function(x) {
    types <- x$detected_hpv_types
    if (length(types) == 0) return("")
    paste(unlist(types), collapse = ",")
  })
  df$hpv_result_detail <- sapply(hpv_results, function(x) x$hpv_result_detail)

  return(df)
}


#' 환자별 결과 변수 요약
#'
#' @param df 결과 변수가 추가된 data.frame
#' @return data.frame with patient-level summary
get_patient_outcomes_summary <- function(df) {

  summary_df <- df %>%
    arrange(연구번호, 실시일자) %>%
    group_by(연구번호) %>%
    summarise(
      # 병변 재발 관련
      has_hsil_cin3_recurrence = any(is_hsil_cin3_or_higher, na.rm = TRUE),
      first_hsil_cin3_date = {
        hsil_dates <- 실시일자[is_hsil_cin3_or_higher == TRUE]
        if (length(hsil_dates) > 0) min(hsil_dates, na.rm = TRUE) else NA
      },
      first_detected_lesion = {
        hsil_rows <- which(is_hsil_cin3_or_higher == TRUE)
        if (length(hsil_rows) > 0) detected_lesion[hsil_rows[1]] else NA_character_
      },
      first_severity_level = {
        hsil_rows <- which(is_hsil_cin3_or_higher == TRUE)
        if (length(hsil_rows) > 0) severity_level[hsil_rows[1]] else NA_character_
      },
      total_hsil_cin3_events = sum(is_hsil_cin3_or_higher, na.rm = TRUE),

      # 고위험 HPV 관련
      has_high_risk_hpv = any(is_high_risk_hpv_positive, na.rm = TRUE),
      first_high_risk_hpv_date = {
        hpv_dates <- 실시일자[is_high_risk_hpv_positive == TRUE]
        if (length(hpv_dates) > 0) min(hpv_dates, na.rm = TRUE) else NA
      },
      detected_hpv_types = {
        all_types <- detected_hpv_types[is_high_risk_hpv_positive == TRUE]
        all_types <- all_types[all_types != ""]
        if (length(all_types) > 0) {
          unique_types <- unique(unlist(strsplit(all_types, ",")))
          unique_types <- unique_types[unique_types != "" & !is.na(unique_types)]
          numeric_types <- suppressWarnings(as.numeric(unique_types))
          numeric_types <- numeric_types[!is.na(numeric_types)]
          if (length(numeric_types) > 0) {
            paste0("[", paste(sort(numeric_types), collapse = ", "), "]")
          } else {
            "[]"
          }
        } else {
          "[]"
        }
      },
      total_hpv_positive_tests = sum(is_high_risk_hpv_positive, na.rm = TRUE),
      .groups = "drop"
    )

  return(summary_df)
}


#' 메인 실행 함수
main <- function() {
  # 데이터 경로 설정
  script_dir <- dirname(sys.frame(1)$ofile)
  if (is.null(script_dir) || script_dir == "") {
    script_dir <- "."
  }
  base_path <- file.path(script_dir, "..")

  data_path <- file.path(base_path, "Data", "pathology_sample.csv")
  output_path <- file.path(base_path, "Data", "pathology_outcomes_R.csv")
  summary_path <- file.path(base_path, "Data", "patient_outcomes_summary_R.csv")

  cat("============================================================\n")
  cat("병리 검사 데이터 결과 변수 추출 (R version)\n")
  cat("============================================================\n")

  # 데이터 로드
  cat(sprintf("\n1. 데이터 로드: %s\n", data_path))
  df <- load_pathology_data(data_path)
  cat(sprintf("   - 전체 레코드 수: %s\n", format(nrow(df), big.mark = ",")))
  cat(sprintf("   - 환자 수: %s\n", format(n_distinct(df$연구번호), big.mark = ",")))

  # 결과 변수 추출
  cat("\n2. 결과 변수 추출 중...\n")
  df <- extract_outcomes(df)

  # 결과 요약 출력
  cat("\n3. 추출 결과 요약:\n")
  cat("----------------------------------------\n")

  # 병변 재발 (HSIL/CIN3+)
  hsil_count <- sum(df$is_hsil_cin3_or_higher, na.rm = TRUE)
  hsil_patients <- n_distinct(df$연구번호[df$is_hsil_cin3_or_higher == TRUE])
  cat("   [병변 재발 (HSIL/CIN3+)]\n")
  cat(sprintf("   - 양성 레코드 수: %s\n", format(hsil_count, big.mark = ",")))
  cat(sprintf("   - 양성 환자 수: %s\n", format(hsil_patients, big.mark = ",")))

  if (hsil_count > 0) {
    severity_counts <- table(df$severity_level[df$is_hsil_cin3_or_higher == TRUE])
    cat("   - 중증도별 분포:\n")
    for (level in names(severity_counts)) {
      cat(sprintf("     * %s: %d\n", level, severity_counts[level]))
    }
  }

  # 고위험 HPV 감염
  hpv_count <- sum(df$is_high_risk_hpv_positive, na.rm = TRUE)
  hpv_patients <- n_distinct(df$연구번호[df$is_high_risk_hpv_positive == TRUE])
  cat("\n   [고위험 HPV 감염]\n")
  cat(sprintf("   - 양성 레코드 수: %s\n", format(hpv_count, big.mark = ",")))
  cat(sprintf("   - 양성 환자 수: %s\n", format(hpv_patients, big.mark = ",")))

  # 환자별 요약 생성
  cat("\n4. 환자별 결과 요약 생성 중...\n")
  summary_df <- get_patient_outcomes_summary(df)

  # 결과 저장
  cat("\n5. 결과 저장:\n")

  write_csv(df, output_path)
  cat(sprintf("   - 상세 결과: %s\n", output_path))

  write_csv(summary_df, summary_path)
  cat(sprintf("   - 환자별 요약: %s\n", summary_path))

  cat("\n============================================================\n")
  cat("추출 완료!\n")
  cat("============================================================\n")

  return(list(df = df, summary_df = summary_df))
}


# 스크립트 직접 실행 시
if (!interactive()) {
  result <- main()
}
