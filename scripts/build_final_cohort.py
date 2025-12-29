"""
코호트 세부 필터링 및 최종 분석 데이터 생성

필터링 조건:
1. Index date ≤ 2020년 12월 31일 (5년 이상 추적 관찰 확보)
2. Biopsy 또는 HPV 추적 관찰 2회 이상
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
from datetime import datetime


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
    pathology = pd.read_csv(data_dir / 'pathology_sample.csv', encoding='utf-8')
    pathology['실시일자'] = pd.to_datetime(pathology['실시일자'], format='%Y%m%d', errors='coerce')

    return cohort, pathology


def filter_by_index_date(cohort: pd.DataFrame,
                          cutoff_date: str = '2020-12-31') -> pd.DataFrame:
    """
    Step 1: Index date 기준 필터링
    5년 이상 추적 관찰을 위해 cutoff_date 이전 index date만 포함
    """
    cutoff = pd.Timestamp(cutoff_date)
    filtered = cohort[cohort['index_date'] <= cutoff].copy()
    return filtered


def count_follow_up_visits(pathology: pd.DataFrame,
                            patient_id: str,
                            index_date: pd.Timestamp) -> dict:
    """
    환자의 index date 이후 추적 관찰 횟수 계산
    """
    patient_data = pathology[pathology['연구번호'] == patient_id].copy()

    # Index date 이후 데이터만
    after_index = patient_data[patient_data['실시일자'] > index_date]

    # 조직병리 (Biopsy) 횟수
    biopsy_count = len(after_index[after_index['병리검사구분'] == '조직병리'])

    # 분자병리 (HPV) 횟수
    hpv_count = len(after_index[after_index['병리검사구분'] == '분자병리'])

    # 전체 추적 횟수
    total_count = biopsy_count + hpv_count

    return {
        'biopsy_count': biopsy_count,
        'hpv_count': hpv_count,
        'total_follow_up': total_count
    }


def filter_by_follow_up(cohort: pd.DataFrame,
                         pathology: pd.DataFrame,
                         min_visits: int = 2) -> pd.DataFrame:
    """
    Step 2: 추적 관찰 횟수 기준 필터링
    Biopsy 또는 HPV 검사가 min_visits회 이상인 환자만 포함
    """
    follow_up_info = []

    for _, patient in cohort.iterrows():
        counts = count_follow_up_visits(
            pathology,
            patient['연구번호'],
            patient['index_date']
        )
        follow_up_info.append({
            '연구번호': patient['연구번호'],
            **counts
        })

    follow_up_df = pd.DataFrame(follow_up_info)

    # 코호트에 추적 정보 추가
    cohort_with_fu = cohort.merge(follow_up_df, on='연구번호', how='left')
    cohort_with_fu['total_follow_up'] = cohort_with_fu['total_follow_up'].fillna(0)

    # 2회 이상 추적 관찰 필터링
    filtered = cohort_with_fu[cohort_with_fu['total_follow_up'] >= min_visits].copy()

    return filtered, cohort_with_fu


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

    # 개별 유형 검출
    for hpv_type in HIGH_RISK_HPV_TYPES:
        if re.search(rf'POSITIVE\s*\(\s*{hpv_type}\s*\)', result_text_upper):
            detected_types.append(hpv_type)

    # Other 유형
    if re.search(r'POSITIVE\s*\(\s*OTHER', result_text_upper):
        other_types = [t for t in HIGH_RISK_HPV_TYPES if t not in [16, 18]]
        detected_types.extend([t for t in other_types if t not in detected_types])

    # Pool 그룹
    pools = {'P1': [33, 58], 'P2': [56, 59, 66], 'P3': [35, 39, 68]}
    for pool, types in pools.items():
        if re.search(rf'POSITIVE\s*\(\s*{pool}\s*\)', result_text_upper):
            detected_types.extend([t for t in types if t not in detected_types])

    # 일반 고위험 양성
    if re.search(r'HIGH[\s-]*RISK.*POSITIVE', result_text_upper) and not detected_types:
        return {'is_positive': True, 'types': ['unspecified'], 'detail': 'High-risk positive'}

    # Negative 확인
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

        # 해당 환자의 index date 이후 병리 데이터
        pt_pathology = pathology[
            (pathology['연구번호'] == patient_id) &
            (pathology['실시일자'] > index_date)
        ].sort_values('실시일자')

        result = {
            '연구번호': patient_id,
            'index_date': index_date,
            '접종여부': patient['접종여부'],
            '수술방법': patient['수술방법'],
            '수술시나이': patient['수술시나이'],
            '최종추적일자': patient['최종추적일자'],
            'biopsy_count': patient.get('biopsy_count', 0),
            'hpv_count': patient.get('hpv_count', 0),
            'total_follow_up': patient.get('total_follow_up', 0),
            # 병변 재발
            'has_recurrence': False,
            'recurrence_date': None,
            'recurrence_severity': None,
            'days_to_recurrence': None,
            # HPV 감염
            'has_hpv_infection': False,
            'hpv_infection_date': None,
            'hpv_types': None,
            'days_to_hpv': None,
            # 추적 기간
            'follow_up_days': None,
        }

        # 추적 기간
        if pd.notna(patient['최종추적일자']):
            result['follow_up_days'] = (patient['최종추적일자'] - index_date).days

        # 병변 재발 (조직병리)
        tissue = pt_pathology[pt_pathology['병리검사구분'] == '조직병리']
        for _, record in tissue.iterrows():
            lesion = detect_hsil_cin3_recurrence(record['판독결과'])
            if lesion['is_positive']:
                result['has_recurrence'] = True
                result['recurrence_date'] = record['실시일자']
                result['recurrence_severity'] = lesion['severity']
                result['days_to_recurrence'] = (record['실시일자'] - index_date).days
                break

        # HPV 감염 (분자병리)
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
    vaccinated = cohort['접종여부'].sum()
    unvaccinated = total - vaccinated

    print(f"\n{step_name}")
    print(f"  - 전체: {total:,}명")
    print(f"    * 접종군: {vaccinated:,}명")
    print(f"    * 비접종군: {unvaccinated:,}명")
    if excluded > 0:
        print(f"  - 제외: {excluded:,}명")


def main():
    """메인 실행 함수"""
    # 경로 설정
    base_path = Path(__file__).parent.parent
    data_dir = base_path / 'Data'

    print("=" * 70)
    print("코호트 필터링 및 최종 분석 데이터 생성")
    print("=" * 70)

    # 1. 데이터 로드
    print("\n[데이터 로드]")
    cohort, pathology = load_data(data_dir)
    print(f"  - 매칭 코호트: {len(cohort):,}명")
    print(f"  - 병리 데이터: {len(pathology):,}건 ({pathology['연구번호'].nunique():,}명)")

    # 코호트 플로우 시작
    print("\n" + "=" * 70)
    print("코호트 선정 흐름도 (Cohort Flow)")
    print("=" * 70)

    # Step 0: 초기 코호트
    step0_n = len(cohort)
    print_cohort_flow("[Step 0] 초기 매칭 코호트", cohort)

    # Step 1: Index date 필터링
    cohort_step1 = filter_by_index_date(cohort, '2020-12-31')
    step1_excluded = step0_n - len(cohort_step1)
    print_cohort_flow("[Step 1] Index date ≤ 2020-12-31 (5년 추적 확보)",
                      cohort_step1, step1_excluded)

    # Step 2: 추적 관찰 횟수 필터링
    cohort_step2, cohort_with_fu = filter_by_follow_up(cohort_step1, pathology, min_visits=2)
    step2_excluded = len(cohort_step1) - len(cohort_step2)
    print_cohort_flow("[Step 2] 추적 관찰 ≥ 2회 (Biopsy 또는 HPV)",
                      cohort_step2, step2_excluded)

    # 추적 관찰 현황 상세
    print("\n  [추적 관찰 현황]")
    fu_stats = cohort_with_fu[['total_follow_up', 'biopsy_count', 'hpv_count']].describe()
    print(f"    - 평균 추적 횟수: {cohort_with_fu['total_follow_up'].mean():.1f}회")
    print(f"    - 추적 0회: {(cohort_with_fu['total_follow_up'] == 0).sum():,}명")
    print(f"    - 추적 1회: {(cohort_with_fu['total_follow_up'] == 1).sum():,}명")
    print(f"    - 추적 2회 이상: {(cohort_with_fu['total_follow_up'] >= 2).sum():,}명")

    # 최종 코호트
    final_cohort = cohort_step2.copy()
    print("\n" + "-" * 70)
    print_cohort_flow("[최종 분석 코호트]", final_cohort)

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
            if n > 0 and group['follow_up_days'].notna().any():
                print(f"    - 평균 추적기간: {group['follow_up_days'].mean():.0f}일")

        # 저장
        print("\n[결과 저장]")

        # 최종 코호트 저장
        final_cohort_path = data_dir / 'final_analysis_cohort.csv'
        final_cohort.to_csv(final_cohort_path, index=False, encoding='utf-8-sig')
        print(f"  - 최종 코호트: {final_cohort_path}")

        # 결과 데이터 저장
        outcomes_path = data_dir / 'final_outcomes.csv'
        outcomes.to_csv(outcomes_path, index=False, encoding='utf-8-sig')
        print(f"  - 결과 데이터: {outcomes_path}")

        # 요약 통계 저장
        summary_data = []
        for group_name, is_vax in [('접종군', True), ('비접종군', False), ('전체', None)]:
            if is_vax is None:
                group = outcomes
            else:
                group = outcomes[outcomes['접종여부'] == is_vax]

            n = len(group)
            if n == 0:
                continue

            summary_data.append({
                '구분': group_name,
                '환자수': n,
                '평균_추적기간_일': group['follow_up_days'].mean(),
                '평균_추적횟수': group['total_follow_up'].mean(),
                '병변재발_건수': group['has_recurrence'].sum(),
                '병변재발률_%': group['has_recurrence'].mean() * 100,
                'HPV감염_건수': group['has_hpv_infection'].sum(),
                'HPV감염률_%': group['has_hpv_infection'].mean() * 100,
            })

        summary_df = pd.DataFrame(summary_data)
        summary_path = data_dir / 'final_summary.csv'
        summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
        print(f"  - 요약 통계: {summary_path}")

    else:
        print("\n⚠️ 최종 코호트에 포함된 환자가 없습니다.")
        outcomes = pd.DataFrame()

    print("\n" + "=" * 70)
    print("완료!")
    print("=" * 70)

    return final_cohort, outcomes


if __name__ == "__main__":
    final_cohort, outcomes = main()
