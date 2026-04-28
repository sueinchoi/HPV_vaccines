"""
매칭 코호트의 Index Date 이후 결과 변수 추출

연구 계획서 7.1 주요 결과변수 (Primary Outcomes):
1. 병변 재발: Index date 이후 조직검사로 확인된 HSIL/CIN3 이상 병변 재발
2. 새로운 고위험 HPV 감염: Index date 이후 HPV 양성 전환
   - 고위험 HPV 유형: 16, 18, 31, 33, 45, 52, 58, 35, 39, 51, 56, 59, 66, 68
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
from datetime import datetime


# 고위험 HPV 유형 정의
HIGH_RISK_HPV_TYPES = [16, 18, 31, 33, 45, 52, 58, 35, 39, 51, 56, 59, 66, 68]


def load_cohort_data(file_path: str) -> pd.DataFrame:
    """매칭 코호트 데이터 로드"""
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    df['index_date'] = pd.to_datetime(df['index_date'])
    df['첫수술일자'] = pd.to_datetime(df['첫수술일자'])
    df['최종추적일자'] = pd.to_datetime(df['최종추적일자'])
    return df


def load_pathology_data(file_path: str) -> pd.DataFrame:
    """병리 검사 데이터 로드"""
    df = pd.read_csv(file_path, encoding='utf-8')
    df['실시일자'] = pd.to_datetime(df['실시일자'], format='%Y%m%d', errors='coerce')
    df['판독일자'] = pd.to_datetime(df['판독일자'], format='%Y%m%d', errors='coerce')
    return df


def detect_hsil_cin3_recurrence(result_text: str) -> dict:
    """
    판독결과에서 HSIL/CIN3 이상 병변 재발 여부 확인
    """
    if pd.isna(result_text):
        return {
            'is_hsil_cin3_or_higher': False,
            'detected_lesion': None,
            'severity_level': None
        }

    result_text = str(result_text).upper()

    # 1. Carcinoma (가장 심각)
    carcinoma_patterns = [
        r'SQUAMOUS\s*CELL\s*CARCINOMA',
        r'ADENOCARCINOMA',
        r'CERVICAL\s*CANCER',
        r'INVASIVE\s*CARCINOMA',
        r'CARCINOMA\s*IN\s*SITU',
        r'\bCIS\b',
    ]
    for pattern in carcinoma_patterns:
        match = re.search(pattern, result_text)
        if match:
            return {
                'is_hsil_cin3_or_higher': True,
                'detected_lesion': match.group(),
                'severity_level': 'Carcinoma'
            }

    # 2. CIN3 패턴
    cin3_patterns = [
        r'CIN\s*3\b', r'CIN\s*III\b', r'CINIII\b', r'CIN-3\b',
        r'CERVICAL\s*INTRAEPITHELIAL\s*NEOPLASIA\s*3',
    ]
    for pattern in cin3_patterns:
        match = re.search(pattern, result_text)
        if match:
            return {
                'is_hsil_cin3_or_higher': True,
                'detected_lesion': match.group(),
                'severity_level': 'CIN3'
            }

    # 3. HSIL
    hsil_patterns = [
        r'\bHSIL\b', r'\bH-SIL\b',
        r'HIGH[\s-]*GRADE\s*SQUAMOUS\s*INTRAEPITHELIAL\s*LESION',
    ]
    for pattern in hsil_patterns:
        match = re.search(pattern, result_text)
        if match:
            return {
                'is_hsil_cin3_or_higher': True,
                'detected_lesion': match.group(),
                'severity_level': 'HSIL'
            }

    # 4. CIN2
    cin2_patterns = [
        r'CIN\s*2\b', r'CIN\s*II\b', r'CINII\b', r'CIN-2\b',
        r'CIN\s*2/3\b', r'CIN\s*II/III\b',
    ]
    for pattern in cin2_patterns:
        match = re.search(pattern, result_text)
        if match:
            return {
                'is_hsil_cin3_or_higher': True,
                'detected_lesion': match.group(),
                'severity_level': 'CIN2'
            }

    return {
        'is_hsil_cin3_or_higher': False,
        'detected_lesion': None,
        'severity_level': None
    }


def detect_high_risk_hpv(result_text: str) -> dict:
    """
    판독결과에서 고위험 HPV 감염 여부 확인
    """
    if pd.isna(result_text):
        return {
            'is_high_risk_hpv_positive': False,
            'detected_hpv_types': [],
            'hpv_result_detail': None
        }

    result_text_upper = str(result_text).upper()
    detected_types = []
    result_detail = None

    # 1. 개별 HPV 유형 양성 확인
    for hpv_type in HIGH_RISK_HPV_TYPES:
        patterns = [
            rf'POSITIVE\s*\(\s*{hpv_type}\s*\)',
            rf'HPV\s*{hpv_type}\s*:\s*POSITIVE',
        ]
        for pattern in patterns:
            if re.search(pattern, result_text_upper):
                if hpv_type not in detected_types:
                    detected_types.append(hpv_type)

    # 2. "Positive(other)" - 16, 18 외 고위험 유형
    if re.search(r'POSITIVE\s*\(\s*OTHER', result_text_upper):
        result_detail = "Other high-risk types detected"
        other_types = [t for t in HIGH_RISK_HPV_TYPES if t not in [16, 18]]
        for t in other_types:
            if t not in detected_types:
                detected_types.append(t)

    # 3. Pool 그룹 (P1, P2, P3)
    pool_types = {'P1': [33, 58], 'P2': [56, 59, 66], 'P3': [35, 39, 68]}
    for pool, types in pool_types.items():
        if re.search(rf'POSITIVE\s*\(\s*{pool}\s*\)', result_text_upper):
            for t in types:
                if t not in detected_types:
                    detected_types.append(t)

    # 4. 일반 고위험 HPV 양성
    general_patterns = [
        r'HPV\s*:\s*POSITIVE\s*\(\s*HIGH[\s-]*RISK',
        r'HIGH[\s-]*RISK\s*HPV\s*:\s*POSITIVE',
    ]
    for pattern in general_patterns:
        if re.search(pattern, result_text_upper):
            if not detected_types:
                result_detail = "High-risk HPV positive (unspecified)"
                detected_types = ['high_risk_unspecified']

    # 5. Negative 확인
    if re.search(r'HPV.*:\s*NEGATIVE', result_text_upper) and not detected_types:
        return {
            'is_high_risk_hpv_positive': False,
            'detected_hpv_types': [],
            'hpv_result_detail': 'Negative'
        }

    is_positive = len(detected_types) > 0
    numeric_types = sorted([t for t in detected_types if isinstance(t, int)])

    return {
        'is_high_risk_hpv_positive': is_positive,
        'detected_hpv_types': numeric_types if numeric_types else detected_types,
        'hpv_result_detail': result_detail
    }


def extract_outcomes_after_index_date(cohort_df: pd.DataFrame,
                                       pathology_df: pd.DataFrame) -> pd.DataFrame:
    """
    각 환자의 index date 이후 결과 변수 추출
    """
    results = []

    for _, patient in cohort_df.iterrows():
        patient_id = patient['연구번호']
        index_date = patient['index_date']
        follow_up_end = patient['최종추적일자']

        # 해당 환자의 병리 데이터 필터링
        patient_pathology = pathology_df[pathology_df['연구번호'] == patient_id].copy()

        # Index date 이후 데이터만 필터링
        patient_pathology = patient_pathology[
            patient_pathology['실시일자'] > index_date
        ].sort_values('실시일자')

        # 결과 변수 초기화
        result = {
            '연구번호': patient_id,
            'index_date': index_date,
            '접종여부': patient['접종여부'],
            '수술방법': patient['수술방법'],
            '수술시나이': patient['수술시나이'],
            '최종추적일자': follow_up_end,
            # 병변 재발
            'has_recurrence': False,
            'recurrence_date': None,
            'recurrence_lesion': None,
            'recurrence_severity': None,
            'days_to_recurrence': None,
            # 고위험 HPV 감염
            'has_hpv_infection': False,
            'hpv_infection_date': None,
            'hpv_types_detected': None,
            'days_to_hpv_infection': None,
            # 추적 기간
            'follow_up_days': None,
        }

        # 추적 기간 계산
        if pd.notna(follow_up_end) and pd.notna(index_date):
            result['follow_up_days'] = (follow_up_end - index_date).days

        if len(patient_pathology) == 0:
            results.append(result)
            continue

        # 병변 재발 검사 (조직병리)
        tissue_pathology = patient_pathology[
            patient_pathology['병리검사구분'] == '조직병리'
        ]

        for _, record in tissue_pathology.iterrows():
            lesion_result = detect_hsil_cin3_recurrence(record['판독결과'])
            if lesion_result['is_hsil_cin3_or_higher']:
                result['has_recurrence'] = True
                result['recurrence_date'] = record['실시일자']
                result['recurrence_lesion'] = lesion_result['detected_lesion']
                result['recurrence_severity'] = lesion_result['severity_level']
                result['days_to_recurrence'] = (record['실시일자'] - index_date).days
                break  # 첫 번째 재발만 기록

        # 고위험 HPV 감염 검사 (분자병리)
        molecular_pathology = patient_pathology[
            patient_pathology['병리검사구분'] == '분자병리'
        ]

        for _, record in molecular_pathology.iterrows():
            hpv_result = detect_high_risk_hpv(record['판독결과'])
            if hpv_result['is_high_risk_hpv_positive']:
                result['has_hpv_infection'] = True
                result['hpv_infection_date'] = record['실시일자']
                result['hpv_types_detected'] = str(hpv_result['detected_hpv_types'])
                result['days_to_hpv_infection'] = (record['실시일자'] - index_date).days
                break  # 첫 번째 감염만 기록

        results.append(result)

    return pd.DataFrame(results)


def calculate_summary_statistics(outcomes_df: pd.DataFrame) -> pd.DataFrame:
    """
    접종군 vs 비접종군 결과 비교 요약
    """
    summary_data = []

    for group_name, is_vaccinated in [('접종군', True), ('비접종군', False)]:
        group = outcomes_df[outcomes_df['접종여부'] == is_vaccinated]

        n_total = len(group)
        if n_total == 0:
            continue

        # 병변 재발
        n_recurrence = group['has_recurrence'].sum()
        recurrence_rate = (n_recurrence / n_total) * 100

        # 재발까지 평균 기간
        recurrence_days = group[group['has_recurrence']]['days_to_recurrence']
        mean_days_to_recurrence = recurrence_days.mean() if len(recurrence_days) > 0 else None

        # 고위험 HPV 감염
        n_hpv = group['has_hpv_infection'].sum()
        hpv_rate = (n_hpv / n_total) * 100

        # HPV 감염까지 평균 기간
        hpv_days = group[group['has_hpv_infection']]['days_to_hpv_infection']
        mean_days_to_hpv = hpv_days.mean() if len(hpv_days) > 0 else None

        # 평균 추적 기간
        mean_follow_up = group['follow_up_days'].mean()

        summary_data.append({
            '구분': group_name,
            '환자수': n_total,
            '평균_추적기간_일': round(mean_follow_up, 1) if pd.notna(mean_follow_up) else None,
            '병변재발_건수': n_recurrence,
            '병변재발률_%': round(recurrence_rate, 2),
            '재발까지_평균일수': round(mean_days_to_recurrence, 1) if pd.notna(mean_days_to_recurrence) else None,
            'HPV감염_건수': n_hpv,
            'HPV감염률_%': round(hpv_rate, 2),
            'HPV감염까지_평균일수': round(mean_days_to_hpv, 1) if pd.notna(mean_days_to_hpv) else None,
        })

    return pd.DataFrame(summary_data)


def main():
    """메인 실행 함수"""
    # 경로 설정
    base_path = Path(__file__).parent.parent
    data_dir = base_path / 'Data'

    cohort_path = data_dir / 'matched_cohort.csv'
    pathology_path = data_dir / 'pathology.csv'
    outcomes_path = data_dir / 'cohort_outcomes.csv'
    summary_path = data_dir / 'outcomes_summary.csv'

    print("=" * 60)
    print("Index Date 이후 결과 변수 추출")
    print("=" * 60)

    # 1. 데이터 로드
    print("\n1. 데이터 로드 중...")
    cohort_df = load_cohort_data(cohort_path)
    pathology_df = load_pathology_data(pathology_path)

    print(f"   - 코호트 환자 수: {len(cohort_df):,}명")
    print(f"     * 접종군: {cohort_df['접종여부'].sum():,}명")
    print(f"     * 비접종군: {(~cohort_df['접종여부']).sum():,}명")
    print(f"   - 병리 검사 기록: {len(pathology_df):,}건")

    # 2. 코호트 환자 중 병리 데이터 있는 환자 확인
    cohort_ids = set(cohort_df['연구번호'])
    pathology_ids = set(pathology_df['연구번호'])
    overlap_ids = cohort_ids & pathology_ids

    print(f"\n2. 데이터 연결 확인...")
    print(f"   - 코호트 환자 중 병리 데이터 보유: {len(overlap_ids):,}명")

    # 3. Index date 이후 결과 변수 추출
    print("\n3. Index date 이후 결과 변수 추출 중...")
    outcomes_df = extract_outcomes_after_index_date(cohort_df, pathology_df)

    # 4. 결과 요약
    print("\n4. 결과 요약:")
    print("-" * 50)

    # 병변 재발
    recurrence_count = outcomes_df['has_recurrence'].sum()
    recurrence_vax = outcomes_df[outcomes_df['접종여부']]['has_recurrence'].sum()
    recurrence_unvax = outcomes_df[~outcomes_df['접종여부']]['has_recurrence'].sum()

    print(f"   [병변 재발 (HSIL/CIN3+)]")
    print(f"   - 전체: {recurrence_count}건")
    print(f"   - 접종군: {recurrence_vax}건 ({recurrence_vax/outcomes_df['접종여부'].sum()*100:.1f}%)")
    print(f"   - 비접종군: {recurrence_unvax}건 ({recurrence_unvax/(~outcomes_df['접종여부']).sum()*100:.1f}%)")

    # 고위험 HPV 감염
    hpv_count = outcomes_df['has_hpv_infection'].sum()
    hpv_vax = outcomes_df[outcomes_df['접종여부']]['has_hpv_infection'].sum()
    hpv_unvax = outcomes_df[~outcomes_df['접종여부']]['has_hpv_infection'].sum()

    print(f"\n   [고위험 HPV 감염]")
    print(f"   - 전체: {hpv_count}건")
    print(f"   - 접종군: {hpv_vax}건 ({hpv_vax/outcomes_df['접종여부'].sum()*100:.1f}%)")
    print(f"   - 비접종군: {hpv_unvax}건 ({hpv_unvax/(~outcomes_df['접종여부']).sum()*100:.1f}%)")

    # 5. 요약 통계 생성
    print("\n5. 요약 통계 생성 중...")
    summary_df = calculate_summary_statistics(outcomes_df)

    # 6. 결과 저장
    print("\n6. 결과 저장:")
    outcomes_df.to_csv(outcomes_path, index=False, encoding='utf-8-sig')
    print(f"   - 개인별 결과: {outcomes_path}")

    summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
    print(f"   - 요약 통계: {summary_path}")

    print("\n" + "=" * 60)
    print("결과 변수 추출 완료!")
    print("=" * 60)

    # 요약 테이블 출력
    print("\n[접종군 vs 비접종군 결과 비교]")
    print(summary_df.to_string(index=False))

    return outcomes_df, summary_df


if __name__ == "__main__":
    outcomes_df, summary_df = main()
