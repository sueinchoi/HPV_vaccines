"""
HPV 백신 효과 연구를 위한 매칭 코호트 구축

연구 계획서에 따른 코호트 구축:
1. 수술 기록이 있는 환자에서 첫 번째 수술 날짜 추출
2. 수술 후 HPV 백신 투여 대상자 찾기 (접종군)
3. 접종군의 index date = 백신 접종일
4. 비접종군은 접종군과 매칭:
   - 매칭 변수: 수술시점 (calendar year ±1년), 수술 시 나이 (±5세), 수술 방법 (원추절제술/자궁절제술)
5. 비접종군의 index date = 수술일 + 매칭된 접종군의 수술-접종 간격(T)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional

# 재현성을 위한 시드 고정
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def load_data(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """데이터 로드"""
    # 수술 데이터
    surgery_df = pd.read_csv(
        data_dir / '한국 HPV 코호트 자료를 이용한 자_수술처방_수술종류구분완료.csv',
        encoding='cp949'
    )

    # 처방 데이터 (백신)
    prescription_df = pd.read_csv(
        data_dir / '한국 HPV 코호트 자료를 이용한 자_처방정보.csv',
        encoding='cp949'
    )

    # 코호트 데이터 (생년월일)
    cohort_df = pd.read_csv(
        data_dir / '한국 HPV 코호트 자료를 이용한 자_코호트.csv',
        encoding='cp949'
    )

    return surgery_df, prescription_df, cohort_df


def preprocess_surgery_data(surgery_df: pd.DataFrame) -> pd.DataFrame:
    """수술 데이터 전처리"""
    df = surgery_df.copy()

    # 수술 종류가 '제외'인 경우 제거
    df = df[df['수술 종류'] != '제외'].copy()

    # 수술 종류 매핑 (1: 원추절제술, 3: 자궁절제술)
    df['수술방법'] = df['수술 종류'].map({1: '원추절제술', '1': '원추절제술',
                                      3: '자궁절제술', '3': '자궁절제술'})

    # 날짜 변환
    df['수술일자'] = pd.to_datetime(df['수술처방일자'], format='%Y%m%d', errors='coerce')
    df['수술연도'] = df['수술일자'].dt.year

    return df


def get_first_surgery(surgery_df: pd.DataFrame) -> pd.DataFrame:
    """환자별 첫 번째 수술 정보 추출"""
    df = surgery_df.sort_values(['연구번호', '수술일자'])

    first_surgery = df.groupby('연구번호').first().reset_index()
    first_surgery = first_surgery[['연구번호', '수술일자', '수술연도', '수술방법', '수술처방코드', '수술처방한글명']]
    first_surgery.columns = ['연구번호', '첫수술일자', '수술연도', '수술방법', '수술코드', '수술명']

    return first_surgery


def extract_hpv_vaccines(prescription_df: pd.DataFrame) -> pd.DataFrame:
    """HPV 백신 처방 추출"""
    df = prescription_df.copy()

    # HPV 백신 코드 패턴
    hpv_vaccine_codes = ['DV-JHP', 'DV-HPF', 'DV-9HPF']  # Cervarix, Gardasil, Gardasil 9

    # HPV 백신만 필터링
    vaccine_mask = df['처방코드'].str.contains('|'.join(hpv_vaccine_codes), na=False)
    vaccine_df = df[vaccine_mask].copy()

    # 날짜 변환
    vaccine_df['접종일자'] = pd.to_datetime(vaccine_df['처방일자'], format='%Y%m%d', errors='coerce')

    # 백신 종류 구분
    def get_vaccine_type(code):
        if 'DV-9HPF' in str(code):
            return 'Gardasil9'
        elif 'DV-HPF' in str(code):
            return 'Gardasil'
        elif 'DV-JHP' in str(code):
            return 'Cervarix'
        return 'Unknown'

    vaccine_df['백신종류'] = vaccine_df['처방코드'].apply(get_vaccine_type)

    return vaccine_df[['연구번호', '접종일자', '백신종류', '처방코드', '처방한글명']]


def get_first_vaccine_after_surgery(first_surgery_df: pd.DataFrame,
                                     vaccine_df: pd.DataFrame) -> pd.DataFrame:
    """수술 후 첫 번째 백신 접종 정보 추출"""
    # 수술 환자와 백신 데이터 조인
    merged = first_surgery_df.merge(vaccine_df, on='연구번호', how='inner')

    # 수술 후 접종만 필터링
    merged = merged[merged['접종일자'] > merged['첫수술일자']].copy()

    # 수술-접종 간격 계산
    merged['수술_접종_간격일'] = (merged['접종일자'] - merged['첫수술일자']).dt.days

    # 환자별 첫 번째 접종만 선택
    merged = merged.sort_values(['연구번호', '접종일자'])
    first_vaccine = merged.groupby('연구번호').first().reset_index()

    return first_vaccine


def calculate_age_at_surgery(first_surgery_df: pd.DataFrame,
                              cohort_df: pd.DataFrame) -> pd.DataFrame:
    """수술 시 나이 계산"""
    # 생년월일 변환
    cohort = cohort_df.copy()
    cohort['생년월일'] = pd.to_datetime(cohort['생년월'], format='%Y%m%d', errors='coerce')

    # 수술 데이터와 조인
    merged = first_surgery_df.merge(cohort[['연구번호', '생년월일', '사망여부', '사망일자', '최종추적일자']],
                                     on='연구번호', how='left')

    # 수술 시 나이 계산 (년 단위)
    merged['수술시나이'] = (merged['첫수술일자'] - merged['생년월일']).dt.days / 365.25

    return merged


def identify_vaccinated_group(first_surgery_with_age: pd.DataFrame,
                               vaccine_df: pd.DataFrame) -> pd.DataFrame:
    """접종군 식별"""
    # 수술 후 백신 접종자 추출
    vaccinated = get_first_vaccine_after_surgery(first_surgery_with_age, vaccine_df)

    # 접종군 표시
    vaccinated['접종여부'] = True
    vaccinated['index_date'] = vaccinated['접종일자']

    return vaccinated


def find_matching_candidates(vaccinated_patient: pd.Series,
                              unvaccinated_pool: pd.DataFrame,
                              surgery_year_tolerance: int = 1,
                              age_tolerance: int = 5) -> pd.DataFrame:
    """
    접종군 환자에 대한 매칭 후보 찾기

    매칭 기준:
    - 수술시점 (calendar year): ±1년 이내
    - 수술 시 나이: ±5세 이내
    - 수술 방법: 정확히 일치 (원추절제술/자궁절제술)
    """
    candidates = unvaccinated_pool.copy()

    # 수술 방법 일치
    candidates = candidates[candidates['수술방법'] == vaccinated_patient['수술방법']]

    # 수술 연도 ±1년
    candidates = candidates[
        abs(candidates['수술연도'] - vaccinated_patient['수술연도']) <= surgery_year_tolerance
    ]

    # 수술 시 나이 ±5세
    candidates = candidates[
        abs(candidates['수술시나이'] - vaccinated_patient['수술시나이']) <= age_tolerance
    ]

    return candidates


def perform_matching(vaccinated_df: pd.DataFrame,
                     unvaccinated_pool: pd.DataFrame,
                     matching_ratio: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    1:N 매칭 수행

    Args:
        vaccinated_df: 접종군 데이터
        unvaccinated_pool: 비접종군 풀
        matching_ratio: 매칭 비율 (1:N)

    Returns:
        matched_vaccinated: 매칭된 접종군
        matched_unvaccinated: 매칭된 비접종군
    """
    matched_vaccinated = []
    matched_unvaccinated = []
    used_unvaccinated = set()

    # 접종군 순회
    for idx, vax_patient in vaccinated_df.iterrows():
        # 매칭 후보 찾기
        candidates = find_matching_candidates(vax_patient, unvaccinated_pool)

        # 이미 매칭된 환자 제외
        candidates = candidates[~candidates['연구번호'].isin(used_unvaccinated)]

        if len(candidates) == 0:
            continue

        # 매칭 비율만큼 선택 (거리 기반 정렬 후 선택)
        # 나이 차이와 연도 차이를 기준으로 가장 가까운 환자 선택
        candidates = candidates.copy()
        candidates['age_diff'] = abs(candidates['수술시나이'] - vax_patient['수술시나이'])
        candidates['year_diff'] = abs(candidates['수술연도'] - vax_patient['수술연도'])
        candidates['total_diff'] = candidates['age_diff'] + candidates['year_diff']
        # 재현성을 위해 연구번호로 2차 정렬 (tie-breaking)
        candidates = candidates.sort_values(['total_diff', '연구번호'])

        selected = candidates.head(matching_ratio)

        if len(selected) > 0:
            # 접종군 추가
            vax_patient_copy = vax_patient.copy()
            vax_patient_copy['matched'] = True
            vax_patient_copy['match_id'] = idx
            matched_vaccinated.append(vax_patient_copy)

            # 비접종군 추가 (index date 계산)
            for _, unvax_patient in selected.iterrows():
                unvax_copy = unvax_patient.copy()
                unvax_copy['matched'] = True
                unvax_copy['match_id'] = idx
                unvax_copy['접종여부'] = False

                # Index date = 수술일 + 매칭된 접종군의 수술-접종 간격
                interval_days = vax_patient['수술_접종_간격일']
                unvax_copy['index_date'] = unvax_copy['첫수술일자'] + pd.Timedelta(days=interval_days)
                unvax_copy['수술_접종_간격일'] = interval_days  # 매칭된 접종군의 간격 적용

                matched_unvaccinated.append(unvax_copy)
                used_unvaccinated.add(unvax_patient['연구번호'])

    # DataFrame으로 변환
    matched_vax_df = pd.DataFrame(matched_vaccinated)
    matched_unvax_df = pd.DataFrame(matched_unvaccinated)

    return matched_vax_df, matched_unvax_df


def build_cohort(data_dir: Path, matching_ratio: int = 5) -> pd.DataFrame:
    """
    전체 코호트 구축 파이프라인

    Args:
        data_dir: 데이터 디렉토리 경로
        matching_ratio: 매칭 비율 (1:N)

    Returns:
        final_cohort: 최종 매칭 코호트
    """
    print("=" * 60)
    print("HPV 백신 효과 연구 - 매칭 코호트 구축")
    print("=" * 60)

    # 1. 데이터 로드
    print("\n1. 데이터 로드 중...")
    surgery_df, prescription_df, cohort_df = load_data(data_dir)
    print(f"   - 수술 데이터: {len(surgery_df):,}건")
    print(f"   - 처방 데이터: {len(prescription_df):,}건")
    print(f"   - 코호트 데이터: {len(cohort_df):,}명")

    # 2. 수술 데이터 전처리
    print("\n2. 수술 데이터 전처리 중...")
    surgery_df = preprocess_surgery_data(surgery_df)
    first_surgery = get_first_surgery(surgery_df)
    print(f"   - 수술 환자 수: {len(first_surgery):,}명")
    print(f"   - 수술 방법 분포:")
    print(f"     * 원추절제술: {(first_surgery['수술방법'] == '원추절제술').sum():,}명")
    print(f"     * 자궁절제술: {(first_surgery['수술방법'] == '자궁절제술').sum():,}명")

    # 3. 수술 시 나이 계산
    print("\n3. 수술 시 나이 계산 중...")
    first_surgery_with_age = calculate_age_at_surgery(first_surgery, cohort_df)
    print(f"   - 평균 수술 나이: {first_surgery_with_age['수술시나이'].mean():.1f}세")

    # 4. HPV 백신 추출
    print("\n4. HPV 백신 접종 데이터 추출 중...")
    vaccine_df = extract_hpv_vaccines(prescription_df)
    print(f"   - 총 백신 접종 기록: {len(vaccine_df):,}건")
    print(f"   - 백신 접종자 수: {vaccine_df['연구번호'].nunique():,}명")
    print(f"   - 백신 종류별 분포:")
    for vtype, count in vaccine_df['백신종류'].value_counts().items():
        print(f"     * {vtype}: {count:,}건")

    # 5. 접종군 식별 (수술 후 백신 접종자)
    print("\n5. 접종군 식별 중 (수술 후 백신 접종자)...")
    vaccinated = identify_vaccinated_group(first_surgery_with_age, vaccine_df)
    print(f"   - 수술 후 백신 접종자: {len(vaccinated):,}명")
    print(f"   - 평균 수술-접종 간격: {vaccinated['수술_접종_간격일'].mean():.0f}일 ({vaccinated['수술_접종_간격일'].mean()/30:.1f}개월)")

    # 6. 비접종군 풀 생성
    print("\n6. 비접종군 풀 생성 중...")
    vaccinated_ids = set(vaccinated['연구번호'])
    unvaccinated_pool = first_surgery_with_age[~first_surgery_with_age['연구번호'].isin(vaccinated_ids)].copy()
    print(f"   - 비접종군 풀: {len(unvaccinated_pool):,}명")

    # 7. 매칭 수행
    print(f"\n7. 1:{matching_ratio} 매칭 수행 중...")
    print("   - 매칭 기준:")
    print("     * 수술시점 (calendar year): ±1년")
    print("     * 수술 시 나이: ±5세")
    print("     * 수술 방법: Exact matching")

    matched_vax, matched_unvax = perform_matching(vaccinated, unvaccinated_pool, matching_ratio)

    print(f"   - 매칭된 접종군: {len(matched_vax):,}명")
    print(f"   - 매칭된 비접종군: {len(matched_unvax):,}명")

    if len(matched_vax) == 0:
        print("\n⚠️ 매칭된 환자가 없습니다!")
        return pd.DataFrame()

    # 8. 최종 코호트 생성
    print("\n8. 최종 코호트 생성 중...")

    # 공통 컬럼 선택
    common_cols = ['연구번호', '첫수술일자', '수술연도', '수술방법', '수술시나이',
                   '생년월일', '사망여부', '사망일자', '최종추적일자',
                   '접종여부', 'index_date', '수술_접종_간격일', 'match_id']

    # 접종군 정리
    vax_cols = [c for c in common_cols if c in matched_vax.columns]
    matched_vax_clean = matched_vax[vax_cols].copy()
    if '백신종류' in matched_vax.columns:
        matched_vax_clean['백신종류'] = matched_vax['백신종류']
    else:
        matched_vax_clean['백신종류'] = None

    # 비접종군 정리
    unvax_cols = [c for c in common_cols if c in matched_unvax.columns]
    matched_unvax_clean = matched_unvax[unvax_cols].copy()
    matched_unvax_clean['백신종류'] = None

    # 합치기
    final_cohort = pd.concat([matched_vax_clean, matched_unvax_clean], ignore_index=True)

    # Index date 유효성 검증 (사망/자격상실 전인지 확인)
    final_cohort['최종추적일자'] = pd.to_datetime(final_cohort['최종추적일자'], format='%Y%m%d', errors='coerce')
    final_cohort['사망일자'] = pd.to_datetime(final_cohort['사망일자'], format='%Y%m%d', errors='coerce')

    # Index date가 최종추적일자 이전인 환자만 유지
    valid_index = (final_cohort['index_date'] <= final_cohort['최종추적일자']) | (final_cohort['최종추적일자'].isna())
    final_cohort = final_cohort[valid_index].copy()

    print(f"   - 최종 코호트 크기: {len(final_cohort):,}명")
    print(f"     * 접종군: {final_cohort['접종여부'].sum():,}명")
    print(f"     * 비접종군: {(~final_cohort['접종여부']).sum():,}명")

    return final_cohort


def main():
    """메인 실행 함수"""
    # 경로 설정
    base_path = Path(__file__).parent.parent
    data_dir = base_path / 'Data'
    output_path = data_dir / 'matched_cohort.csv'
    summary_path = data_dir / 'cohort_summary.csv'

    # 코호트 구축 (1:5 매칭)
    final_cohort = build_cohort(data_dir, matching_ratio=5)

    if len(final_cohort) == 0:
        print("\n코호트 구축 실패!")
        return

    # 결과 저장
    print("\n9. 결과 저장 중...")
    final_cohort.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"   - 매칭 코호트: {output_path}")

    # 요약 통계 생성
    summary = {
        '구분': ['접종군', '비접종군', '전체'],
        '환자수': [
            final_cohort['접종여부'].sum(),
            (~final_cohort['접종여부']).sum(),
            len(final_cohort)
        ],
        '평균_수술나이': [
            final_cohort[final_cohort['접종여부']]['수술시나이'].mean(),
            final_cohort[~final_cohort['접종여부']]['수술시나이'].mean(),
            final_cohort['수술시나이'].mean()
        ],
        '원추절제술_비율': [
            (final_cohort[final_cohort['접종여부']]['수술방법'] == '원추절제술').mean() * 100,
            (final_cohort[~final_cohort['접종여부']]['수술방법'] == '원추절제술').mean() * 100,
            (final_cohort['수술방법'] == '원추절제술').mean() * 100
        ],
        '평균_수술_index_간격일': [
            final_cohort[final_cohort['접종여부']]['수술_접종_간격일'].mean(),
            final_cohort[~final_cohort['접종여부']]['수술_접종_간격일'].mean(),
            final_cohort['수술_접종_간격일'].mean()
        ]
    }
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
    print(f"   - 요약 통계: {summary_path}")

    print("\n" + "=" * 60)
    print("코호트 구축 완료!")
    print("=" * 60)

    # 요약 출력
    print("\n[코호트 요약]")
    print(summary_df.to_string(index=False))

    return final_cohort


if __name__ == "__main__":
    cohort = main()
