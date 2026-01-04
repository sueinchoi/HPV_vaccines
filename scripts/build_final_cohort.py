"""
코호트 세부 필터링 및 Fine Matching을 통한 최종 분석 데이터 생성

필터링 조건:
1. Index date ≤ 2020년 12월 31일 (5년 이상 추적 관찰 확보)
2. Biopsy 또는 HPV 추적 관찰 2회 이상

Fine Matching 변수 (Step 3):
- Index date 기준 나이
- Index date와 가장 가까운 BMI
- 수술 년도
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
from datetime import datetime
from typing import Optional


# 고위험 HPV 유형 정의
HIGH_RISK_HPV_TYPES = [16, 18, 31, 33, 45, 52, 58, 35, 39, 51, 56, 59, 66, 68]


def load_data(data_dir: Path) -> tuple:
    """데이터 로드"""
    # 매칭 코호트
    cohort = pd.read_csv(data_dir / 'matched_cohort.csv', encoding='utf-8-sig')
    cohort['index_date'] = pd.to_datetime(cohort['index_date'])
    cohort['첫수술일자'] = pd.to_datetime(cohort['첫수술일자'])
    cohort['최종추적일자'] = pd.to_datetime(cohort['최종추적일자'])

    # 병리 데이터
    pathology_file = data_dir / 'pathology.csv'
    if not pathology_file.exists():
        pathology_file = data_dir / 'pathology_sample.csv'
    pathology = pd.read_csv(pathology_file, encoding='utf-8')
    pathology['실시일자'] = pd.to_datetime(pathology['실시일자'], format='%Y%m%d', errors='coerce')

    # 기초임상정보 (BMI)
    clinical = pd.read_csv(
        data_dir / '한국 HPV 코호트 자료를 이용한 자_기초임상정보.csv',
        encoding='cp949',
        low_memory=False
    )
    clinical['기록일자'] = pd.to_datetime(clinical['기록일자'], format='%Y%m%d', errors='coerce')

    # 코호트 데이터 (생년월일)
    cohort_info = pd.read_csv(
        data_dir / '한국 HPV 코호트 자료를 이용한 자_코호트.csv',
        encoding='cp949'
    )
    cohort_info['생년월일'] = pd.to_datetime(cohort_info['생년월'], format='%Y%m%d', errors='coerce')

    return cohort, pathology, clinical, cohort_info


def calculate_bmi(height_cm: float, weight_kg: float) -> Optional[float]:
    """BMI 계산 (kg/m²)"""
    if pd.isna(height_cm) or pd.isna(weight_kg):
        return None
    if height_cm <= 0 or weight_kg <= 0:
        return None
    height_m = height_cm / 100
    return weight_kg / (height_m ** 2)


def get_closest_bmi(clinical_df: pd.DataFrame,
                     patient_id: str,
                     target_date: pd.Timestamp) -> Optional[float]:
    """
    Index date와 가장 가까운 날짜의 BMI 반환
    """
    patient_data = clinical_df[clinical_df['연구번호'] == patient_id].copy()

    # BMI 계산 가능한 레코드만
    patient_data = patient_data[
        patient_data['키'].notna() & patient_data['몸무게'].notna()
    ].copy()

    if len(patient_data) == 0:
        return None

    # BMI 계산
    patient_data['BMI'] = patient_data.apply(
        lambda x: calculate_bmi(x['키'], x['몸무게']), axis=1
    )

    # 날짜 차이 계산
    patient_data['date_diff'] = abs((patient_data['기록일자'] - target_date).dt.days)

    # 가장 가까운 날짜의 BMI 반환
    closest_idx = patient_data['date_diff'].idxmin()
    return patient_data.loc[closest_idx, 'BMI']


def calculate_age_at_index(birth_date: pd.Timestamp,
                            index_date: pd.Timestamp) -> Optional[float]:
    """Index date 기준 나이 계산"""
    if pd.isna(birth_date) or pd.isna(index_date):
        return None
    return (index_date - birth_date).days / 365.25


def add_matching_variables(cohort: pd.DataFrame,
                            clinical: pd.DataFrame,
                            cohort_info: pd.DataFrame) -> pd.DataFrame:
    """
    Fine matching을 위한 변수 추가:
    - index_age: Index date 기준 나이
    - closest_bmi: Index date와 가장 가까운 BMI
    - surgery_year: 수술 년도 (이미 있음)
    """
    # 기존 생년월일 컬럼 제거 (있는 경우)
    if '생년월일' in cohort.columns:
        cohort = cohort.drop(columns=['생년월일'])

    # 생년월일 정보 추가
    cohort_info_subset = cohort_info[['연구번호', '생년월일']].drop_duplicates(subset='연구번호')
    cohort = cohort.merge(
        cohort_info_subset,
        on='연구번호',
        how='left'
    )

    # Index date 기준 나이 계산
    cohort['index_age'] = cohort.apply(
        lambda x: calculate_age_at_index(x['생년월일'], x['index_date']),
        axis=1
    )

    # Index date와 가장 가까운 BMI
    print("    - BMI 추출 중...")
    bmi_values = []
    for _, row in cohort.iterrows():
        bmi = get_closest_bmi(clinical, row['연구번호'], row['index_date'])
        bmi_values.append(bmi)
    cohort['closest_bmi'] = bmi_values

    return cohort


def filter_by_index_date(cohort: pd.DataFrame,
                          cutoff_date: str = '2020-12-31') -> pd.DataFrame:
    """Step 1: Index date 기준 필터링"""
    cutoff = pd.Timestamp(cutoff_date)
    return cohort[cohort['index_date'] <= cutoff].copy()


def count_follow_up_visits(pathology: pd.DataFrame,
                            patient_id: str,
                            index_date: pd.Timestamp) -> dict:
    """환자의 index date 이후 추적 관찰 횟수 계산"""
    patient_data = pathology[pathology['연구번호'] == patient_id].copy()
    after_index = patient_data[patient_data['실시일자'] > index_date]

    biopsy_count = len(after_index[after_index['병리검사구분'] == '조직병리'])
    hpv_count = len(after_index[after_index['병리검사구분'] == '분자병리'])

    return {
        'biopsy_count': biopsy_count,
        'hpv_count': hpv_count,
        'total_follow_up': biopsy_count + hpv_count
    }


def filter_by_follow_up(cohort: pd.DataFrame,
                         pathology: pd.DataFrame,
                         min_visits: int = 2) -> tuple:
    """Step 2: 추적 관찰 횟수 기준 필터링"""
    follow_up_info = []

    for _, patient in cohort.iterrows():
        counts = count_follow_up_visits(
            pathology, patient['연구번호'], patient['index_date']
        )
        follow_up_info.append({'연구번호': patient['연구번호'], **counts})

    follow_up_df = pd.DataFrame(follow_up_info)
    cohort_with_fu = cohort.merge(follow_up_df, on='연구번호', how='left')
    cohort_with_fu['total_follow_up'] = cohort_with_fu['total_follow_up'].fillna(0)

    filtered = cohort_with_fu[cohort_with_fu['total_follow_up'] >= min_visits].copy()
    return filtered, cohort_with_fu


def perform_fine_matching(vaccinated: pd.DataFrame,
                           unvaccinated: pd.DataFrame,
                           age_tolerance: float = 5.0,
                           bmi_tolerance: float = 3.0,
                           year_tolerance: int = 1,
                           matching_ratio: int = 1) -> tuple:
    """
    Step 3: Fine Matching 수행

    매칭 변수:
    - index_age: Index date 기준 나이 (±5세)
    - closest_bmi: BMI (±3 kg/m²)
    - surgery_year: 수술 년도 (±1년)
    """
    matched_vax = []
    matched_unvax = []
    used_unvax_ids = set()

    for idx, vax in vaccinated.iterrows():
        # 매칭 후보 찾기
        candidates = unvaccinated[~unvaccinated['연구번호'].isin(used_unvax_ids)].copy()

        # 수술 년도 매칭 (±1년)
        if pd.notna(vax['수술연도']):
            candidates = candidates[
                abs(candidates['수술연도'] - vax['수술연도']) <= year_tolerance
            ]

        # Index 나이 매칭 (±5세)
        if pd.notna(vax['index_age']):
            candidates = candidates[
                candidates['index_age'].notna() &
                (abs(candidates['index_age'] - vax['index_age']) <= age_tolerance)
            ]

        # BMI 매칭 (±3 kg/m², 둘 다 있는 경우만)
        if pd.notna(vax['closest_bmi']):
            # BMI가 있는 후보만 고려, 없으면 BMI 조건 무시
            candidates_with_bmi = candidates[
                candidates['closest_bmi'].notna() &
                (abs(candidates['closest_bmi'] - vax['closest_bmi']) <= bmi_tolerance)
            ]
            # BMI 매칭 후보가 있으면 사용, 없으면 BMI 무시
            if len(candidates_with_bmi) > 0:
                candidates = candidates_with_bmi

        if len(candidates) == 0:
            continue

        # 거리 기반 최적 매칭 선택
        candidates = candidates.copy()

        # 정규화된 거리 계산
        candidates['age_diff'] = 0.0
        candidates['bmi_diff'] = 0.0
        candidates['year_diff'] = 0.0

        if pd.notna(vax['index_age']):
            candidates['age_diff'] = abs(candidates['index_age'] - vax['index_age']) / age_tolerance

        if pd.notna(vax['closest_bmi']) and candidates['closest_bmi'].notna().any():
            candidates.loc[candidates['closest_bmi'].notna(), 'bmi_diff'] = \
                abs(candidates.loc[candidates['closest_bmi'].notna(), 'closest_bmi'] - vax['closest_bmi']) / bmi_tolerance

        if pd.notna(vax['수술연도']):
            candidates['year_diff'] = abs(candidates['수술연도'] - vax['수술연도']) / year_tolerance

        candidates['total_distance'] = candidates['age_diff'] + candidates['bmi_diff'] + candidates['year_diff']
        candidates = candidates.sort_values('total_distance')

        # 매칭 비율만큼 선택
        selected = candidates.head(matching_ratio)

        # 매칭 결과 저장
        vax_copy = vax.copy()
        vax_copy['fine_match_id'] = idx
        matched_vax.append(vax_copy)

        for _, unvax in selected.iterrows():
            unvax_copy = unvax.copy()
            unvax_copy['fine_match_id'] = idx
            matched_unvax.append(unvax_copy)
            used_unvax_ids.add(unvax['연구번호'])

    matched_vax_df = pd.DataFrame(matched_vax) if matched_vax else pd.DataFrame()
    matched_unvax_df = pd.DataFrame(matched_unvax) if matched_unvax else pd.DataFrame()

    return matched_vax_df, matched_unvax_df


def detect_hsil_cin3_recurrence(result_text: str) -> dict:
    """HSIL/CIN3 이상 병변 검출"""
    if pd.isna(result_text):
        return {'is_positive': False, 'lesion': None, 'severity': None}

    result_text = str(result_text).upper()
    patterns = {
        'Carcinoma': [r'SQUAMOUS\s*CELL\s*CARCINOMA', r'ADENOCARCINOMA',
                      r'CERVICAL\s*CANCER', r'CARCINOMA\s*IN\s*SITU'],
        'CIN3': [r'CIN\s*3\b', r'CIN\s*III\b', r'CINIII\b'],
        'HSIL': [r'\bHSIL\b', r'HIGH[\s-]*GRADE\s*SQUAMOUS\s*INTRAEPITHELIAL'],
        'CIN2': [r'CIN\s*2\b', r'CIN\s*II\b', r'CINII\b', r'CIN\s*2/3\b'],
    }

    for severity, pattern_list in patterns.items():
        for pattern in pattern_list:
            match = re.search(pattern, result_text)
            if match:
                return {'is_positive': True, 'lesion': match.group(), 'severity': severity}

    return {'is_positive': False, 'lesion': None, 'severity': None}


def detect_high_risk_hpv(result_text: str) -> dict:
    """고위험 HPV 감염 검출"""
    if pd.isna(result_text):
        return {'is_positive': False, 'types': [], 'detail': None}

    result_text_upper = str(result_text).upper()
    detected_types = []

    for hpv_type in HIGH_RISK_HPV_TYPES:
        if re.search(rf'POSITIVE\s*\(\s*{hpv_type}\s*\)', result_text_upper):
            detected_types.append(hpv_type)

    if re.search(r'POSITIVE\s*\(\s*OTHER', result_text_upper):
        other_types = [t for t in HIGH_RISK_HPV_TYPES if t not in [16, 18]]
        detected_types.extend([t for t in other_types if t not in detected_types])

    pools = {'P1': [33, 58], 'P2': [56, 59, 66], 'P3': [35, 39, 68]}
    for pool, types in pools.items():
        if re.search(rf'POSITIVE\s*\(\s*{pool}\s*\)', result_text_upper):
            detected_types.extend([t for t in types if t not in detected_types])

    if re.search(r'HIGH[\s-]*RISK.*POSITIVE', result_text_upper) and not detected_types:
        return {'is_positive': True, 'types': ['unspecified'], 'detail': 'High-risk positive'}

    if re.search(r'HPV.*:\s*NEGATIVE', result_text_upper) and not detected_types:
        return {'is_positive': False, 'types': [], 'detail': 'Negative'}

    return {
        'is_positive': len(detected_types) > 0,
        'types': sorted(set([t for t in detected_types if isinstance(t, int)])),
        'detail': None
    }


def extract_outcomes(cohort: pd.DataFrame, pathology: pd.DataFrame) -> pd.DataFrame:
    """최종 코호트에서 결과 변수 추출"""
    results = []

    for _, patient in cohort.iterrows():
        patient_id = patient['연구번호']
        index_date = patient['index_date']

        pt_pathology = pathology[
            (pathology['연구번호'] == patient_id) &
            (pathology['실시일자'] > index_date)
        ].sort_values('실시일자')

        result = {
            '연구번호': patient_id,
            'index_date': index_date,
            '접종여부': patient['접종여부'],
            '수술방법': patient['수술방법'],
            'index_age': patient.get('index_age'),
            'closest_bmi': patient.get('closest_bmi'),
            '수술연도': patient.get('수술연도'),
            '최종추적일자': patient.get('최종추적일자'),
            'fine_match_id': patient.get('fine_match_id'),
            'biopsy_count': patient.get('biopsy_count', 0),
            'hpv_count': patient.get('hpv_count', 0),
            'has_recurrence': False,
            'recurrence_date': None,
            'recurrence_severity': None,
            'days_to_recurrence': None,
            'has_hpv_infection': False,
            'hpv_infection_date': None,
            'hpv_types': None,
            'days_to_hpv': None,
            'follow_up_days': None,
        }

        if pd.notna(patient.get('최종추적일자')):
            result['follow_up_days'] = (patient['최종추적일자'] - index_date).days

        # 병변 재발
        tissue = pt_pathology[pt_pathology['병리검사구분'] == '조직병리']
        for _, record in tissue.iterrows():
            lesion = detect_hsil_cin3_recurrence(record['판독결과'])
            if lesion['is_positive']:
                result['has_recurrence'] = True
                result['recurrence_date'] = record['실시일자']
                result['recurrence_severity'] = lesion['severity']
                result['days_to_recurrence'] = (record['실시일자'] - index_date).days
                break

        # HPV 감염
        molecular = pt_pathology[pt_pathology['병리검사구분'] == '분자병리']
        for _, record in molecular.iterrows():
            hpv = detect_high_risk_hpv(record['판독결과'])
            if hpv['is_positive']:
                result['has_hpv_infection'] = True
                result['hpv_infection_date'] = record['실시일자']
                result['hpv_types'] = str(hpv['types'])
                result['days_to_hpv'] = (record['실시일자'] - index_date).days
                break

        results.append(result)

    return pd.DataFrame(results)


def print_cohort_flow(step_name: str, cohort: pd.DataFrame, excluded: int = 0):
    """코호트 현황 출력"""
    total = len(cohort)
    vaccinated = cohort['접종여부'].sum() if '접종여부' in cohort.columns else 0
    unvaccinated = total - vaccinated

    print(f"\n{step_name}")
    print(f"  - 전체: {total:,}명")
    print(f"    * 접종군: {vaccinated:,}명")
    print(f"    * 비접종군: {unvaccinated:,}명")
    if excluded > 0:
        print(f"  - 제외: {excluded:,}명")


def print_matching_balance(vaccinated: pd.DataFrame, unvaccinated: pd.DataFrame):
    """매칭 균형 확인"""
    print("\n  [매칭 변수 균형 확인]")
    print(f"  {'변수':<20} {'접종군':>15} {'비접종군':>15} {'차이':>10}")
    print("  " + "-" * 60)

    # Index Age
    vax_age = vaccinated['index_age'].mean()
    unvax_age = unvaccinated['index_age'].mean()
    print(f"  {'Index 나이 (세)':<20} {vax_age:>15.1f} {unvax_age:>15.1f} {abs(vax_age-unvax_age):>10.1f}")

    # BMI
    vax_bmi = vaccinated['closest_bmi'].mean()
    unvax_bmi = unvaccinated['closest_bmi'].mean()
    if pd.notna(vax_bmi) and pd.notna(unvax_bmi):
        print(f"  {'BMI (kg/m²)':<20} {vax_bmi:>15.1f} {unvax_bmi:>15.1f} {abs(vax_bmi-unvax_bmi):>10.1f}")
    else:
        print(f"  {'BMI (kg/m²)':<20} {'N/A':>15} {'N/A':>15} {'N/A':>10}")

    # Surgery Year
    vax_year = vaccinated['수술연도'].mean()
    unvax_year = unvaccinated['수술연도'].mean()
    print(f"  {'수술 년도':<20} {vax_year:>15.1f} {unvax_year:>15.1f} {abs(vax_year-unvax_year):>10.1f}")


def main():
    """메인 실행 함수"""
    base_path = Path(__file__).parent.parent
    data_dir = base_path / 'Data'

    print("=" * 70)
    print("코호트 필터링 및 Fine Matching을 통한 최종 분석 데이터 생성")
    print("=" * 70)

    # 데이터 로드
    print("\n[데이터 로드]")
    cohort, pathology, clinical, cohort_info = load_data(data_dir)
    print(f"  - 매칭 코호트: {len(cohort):,}명")
    print(f"  - 병리 데이터: {len(pathology):,}건")
    print(f"  - 기초임상정보: {len(clinical):,}건")

    # 코호트 플로우
    print("\n" + "=" * 70)
    print("코호트 선정 흐름도 (Cohort Flow)")
    print("=" * 70)

    # Step 0: 초기 코호트
    step0_n = len(cohort)
    print_cohort_flow("[Step 0] 초기 매칭 코호트", cohort)

    # Step 1: Index date 필터링
    cohort_step1 = filter_by_index_date(cohort, '2023-12-31')
    print_cohort_flow("[Step 1] Index date ≤ 2023-12-31 (2년 추적 확보)",
                      cohort_step1, step0_n - len(cohort_step1))

    # Step 2: 추적 관찰 횟수 필터링
    cohort_step2, cohort_with_fu = filter_by_follow_up(cohort_step1, pathology, min_visits=2)

    # 추적 관찰 데이터가 없는 경우 (샘플 데이터 등) Step 1 코호트 사용
    if len(cohort_step2) == 0 and len(cohort_step1) > 0:
        print("\n  ⚠️ 병리 데이터에 매칭되는 환자가 없음 - 추적 관찰 필터 생략")
        print("     (샘플 데이터 사용 시 발생할 수 있음)")
        cohort_step2 = cohort_step1.copy()
        cohort_step2['biopsy_count'] = 0
        cohort_step2['hpv_count'] = 0
        cohort_step2['total_follow_up'] = 0
        print_cohort_flow("[Step 2] 추적 관찰 필터 생략 (데이터 미매칭)",
                          cohort_step2, 0)
    else:
        print_cohort_flow("[Step 2] 추적 관찰 ≥ 2회",
                          cohort_step2, len(cohort_step1) - len(cohort_step2))

        print(f"\n  [추적 관찰 현황]")
        print(f"    - 추적 0회: {(cohort_with_fu['total_follow_up'] == 0).sum():,}명")
        print(f"    - 추적 1회: {(cohort_with_fu['total_follow_up'] == 1).sum():,}명")
        print(f"    - 추적 2회 이상: {(cohort_with_fu['total_follow_up'] >= 2).sum():,}명")

    # Step 3: Fine Matching 변수 추가 및 매칭
    if len(cohort_step2) > 0:
        print("\n[Step 3] Fine Matching 변수 추가")
        cohort_step2 = add_matching_variables(cohort_step2, clinical, cohort_info)

        # BMI 현황
        bmi_available = cohort_step2['closest_bmi'].notna().sum()
        print(f"    - BMI 보유 환자: {bmi_available:,}명 ({bmi_available/len(cohort_step2)*100:.1f}%)")

        # Fine Matching 수행
        print("\n[Step 3] Fine Matching 수행")
        print("  - 매칭 변수:")
        print("    * Index 나이: ±5세")
        print("    * BMI: ±3 kg/m²")
        print("    * 수술 년도: ±1년")

        vaccinated = cohort_step2[cohort_step2['접종여부'] == True]
        unvaccinated = cohort_step2[cohort_step2['접종여부'] == False]

        matched_vax, matched_unvax = perform_fine_matching(
            vaccinated, unvaccinated,
            age_tolerance=5.0,
            bmi_tolerance=3.0,
            year_tolerance=1,
            matching_ratio=1
        )

        print_cohort_flow("[Step 3] Fine Matching 완료",
                          pd.concat([matched_vax, matched_unvax]) if len(matched_vax) > 0 else pd.DataFrame())

        if len(matched_vax) > 0 and len(matched_unvax) > 0:
            print_matching_balance(matched_vax, matched_unvax)

        # 최종 코호트
        final_cohort = pd.concat([matched_vax, matched_unvax], ignore_index=True)
    else:
        print("\n[Step 2 이후 대상자 없음 - Fine Matching 생략]")
        final_cohort = cohort_step2

    # 결과 변수 추출
    print("\n" + "=" * 70)
    print("결과 변수 추출")
    print("=" * 70)

    if len(final_cohort) > 0:
        outcomes = extract_outcomes(final_cohort, pathology)

        # 결과 요약
        print("\n[결과 요약]")
        for group_name, is_vax in [('접종군', True), ('비접종군', False)]:
            group = outcomes[outcomes['접종여부'] == is_vax]
            n = len(group)
            if n == 0:
                continue

            recurrence = group['has_recurrence'].sum()
            hpv = group['has_hpv_infection'].sum()

            print(f"\n  {group_name} (n={n})")
            print(f"    - 병변 재발: {recurrence}건 ({recurrence/n*100:.1f}%)")
            print(f"    - HPV 감염: {hpv}건 ({hpv/n*100:.1f}%)")

        # 저장
        print("\n[결과 저장]")

        final_cohort.to_csv(data_dir / 'final_matched_cohort.csv', index=False, encoding='utf-8-sig')
        print(f"  - 최종 코호트: final_matched_cohort.csv")

        outcomes.to_csv(data_dir / 'final_matched_outcomes.csv', index=False, encoding='utf-8-sig')
        print(f"  - 결과 데이터: final_matched_outcomes.csv")

        # 요약 통계
        summary_data = []
        for group_name, is_vax in [('접종군', True), ('비접종군', False), ('전체', None)]:
            group = outcomes if is_vax is None else outcomes[outcomes['접종여부'] == is_vax]
            n = len(group)
            if n == 0:
                continue

            summary_data.append({
                '구분': group_name,
                '환자수': n,
                '평균_Index나이': group['index_age'].mean(),
                '평균_BMI': group['closest_bmi'].mean(),
                '평균_수술년도': group['수술연도'].mean(),
                '평균_추적기간_일': group['follow_up_days'].mean(),
                '병변재발_건수': group['has_recurrence'].sum(),
                '병변재발률_%': group['has_recurrence'].mean() * 100,
                'HPV감염_건수': group['has_hpv_infection'].sum(),
                'HPV감염률_%': group['has_hpv_infection'].mean() * 100,
            })

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(data_dir / 'final_matched_summary.csv', index=False, encoding='utf-8-sig')
        print(f"  - 요약 통계: final_matched_summary.csv")
    else:
        print("\n⚠️ 최종 코호트에 포함된 환자가 없습니다.")
        outcomes = pd.DataFrame()

    print("\n" + "=" * 70)
    print("완료!")
    print("=" * 70)

    return final_cohort, outcomes


if __name__ == "__main__":
    final_cohort, outcomes = main()
