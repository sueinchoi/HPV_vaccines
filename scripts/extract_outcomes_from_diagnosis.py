"""
진단 데이터에서 결과 변수 추출

- 병변 재발: Index date 이후 CIN2+/CIN3+/HSIL 진단
- HPV 관련: 진단 코드 기반 (Lab에 HPV 검사 없음)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


def load_data(data_dir: Path):
    """데이터 로드"""
    # 최종 매칭 코호트
    cohort = pd.read_csv(data_dir / 'final_matched_cohort.csv', encoding='utf-8-sig')
    cohort['index_date'] = pd.to_datetime(cohort['index_date'])

    # 진단 정보
    diag = pd.read_csv(data_dir / '한국 HPV 코호트 자료를 이용한 자_진단정보.csv',
                       encoding='cp949', low_memory=False)
    # 날짜 형식: YYYYMMDD 정수
    diag['진단일자'] = pd.to_datetime(diag['진단일자'].astype(str), format='%Y%m%d', errors='coerce')

    return cohort, diag


def extract_lesion_recurrence(cohort: pd.DataFrame, diag: pd.DataFrame) -> pd.DataFrame:
    """
    Index date 이후 병변 재발 추출

    재발 기준:
    - CIN2/CIN3/HSIL/CIS (D06, N87.1, N87.2 등)
    """
    # 재발 관련 진단 코드/명 패턴 (CIN2+ 만 해당, 일반 CIN 제외)
    recurrence_patterns = {
        'codes': ['D06', 'N871', 'N872'],  # CIS, CIN2, CIN3
        'names': ['CIN II', 'CIN III', 'CIN2', 'CIN3', 'HSIL',
                  'Severe dysplasia', 'Moderate dysplasia',
                  'CIN 2', 'CIN 3', '고도 이형성', '상피내암'],
        # 제외 패턴 (일반 CIN 또는 CIN1)
        'exclude': ['CIN1', 'CIN I', 'CIN 1', 'LSIL', 'Low grade', 'Mild dysplasia']
    }

    # 재발 판정 최소 기간 (수술 후 90일 이후부터 재발로 인정)
    MIN_DAYS_FOR_RECURRENCE = 90

    results = []

    for _, patient in cohort.iterrows():
        patient_id = patient['연구번호']
        index_date = patient['index_date']
        last_fu_date = pd.to_datetime(patient['최종추적일자'])

        # 환자의 진단 기록
        patient_diag = diag[diag['연구번호'] == patient_id].copy()

        # Index date + 90일 이후 진단만 (수술 직후 치료 관련 진단 제외)
        min_recurrence_date = index_date + pd.Timedelta(days=MIN_DAYS_FOR_RECURRENCE)
        patient_diag = patient_diag[patient_diag['진단일자'] >= min_recurrence_date]

        # 재발 진단 찾기
        recurrence_found = False
        recurrence_date = None
        recurrence_diagnosis = None

        for _, dx in patient_diag.iterrows():
            dx_code = str(dx['진단코드']) if pd.notna(dx['진단코드']) else ''
            dx_name = str(dx['진단명']) if pd.notna(dx['진단명']) else ''

            # 제외 패턴 확인
            is_excluded = any(excl.lower() in dx_name.lower()
                            for excl in recurrence_patterns['exclude'])
            if is_excluded:
                continue

            # 진단 코드 확인 (D06=CIS, N871=CIN2, N872=CIN3)
            code_match = any(code in dx_code for code in recurrence_patterns['codes'])

            # 진단명 확인 - 특정 고등급 병변만
            name_match = any(pattern.lower() in dx_name.lower()
                           for pattern in recurrence_patterns['names'])

            # 일반 "CIN" 또는 "CIS of Cx" 만 있는 경우는 숫자가 있어야 함
            # "CIN (CIN : Cervical...)" 같은 일반 코드 제외
            if 'CIN' in dx_name and not any(x in dx_name for x in ['CIN2', 'CIN3', 'CIN II', 'CIN III', 'CIN 2', 'CIN 3']):
                if 'CIS' not in dx_name:  # CIS는 CIN3 급이므로 포함
                    name_match = False

            if code_match or name_match:
                recurrence_found = True
                recurrence_date = dx['진단일자']
                recurrence_diagnosis = dx_name
                break  # 첫 재발만

        # 추적 기간 계산
        # follow_up_days: 전체 추적 기간 (최종 추적일까지)
        # days_to_recurrence: 재발까지의 시간 (재발 없으면 NaN)
        follow_up_days = (last_fu_date - index_date).days if pd.notna(last_fu_date) else 0

        if recurrence_found and pd.notna(recurrence_date):
            days_to_recurrence = (recurrence_date - index_date).days
        else:
            days_to_recurrence = None  # 재발 없으면 NaN (censored)

        results.append({
            '연구번호': patient_id,
            'index_date': index_date,
            '접종여부': patient['접종여부'],
            'index_age': patient.get('index_age'),
            'closest_bmi': patient.get('closest_bmi'),
            '수술연도': patient.get('수술연도'),
            '수술방법': patient.get('수술방법'),
            'fine_match_id': patient.get('fine_match_id'),  # 매칭 ID 추가
            'has_recurrence': 1 if recurrence_found else 0,
            'days_to_recurrence': days_to_recurrence,  # 재발까지 시간 (재발 없으면 NaN)
            'recurrence_date': recurrence_date,
            'recurrence_diagnosis': recurrence_diagnosis,
            'follow_up_days': max(follow_up_days, 1)  # 전체 추적기간 (최소 1일)
        })

    return pd.DataFrame(results)


def main():
    base_path = Path(__file__).parent.parent
    data_dir = base_path / 'Data'

    print("=" * 70)
    print("진단 데이터에서 결과 변수 추출")
    print("=" * 70)

    # 데이터 로드
    print("\n[1. 데이터 로드]")
    cohort, diag = load_data(data_dir)
    print(f"  - 최종 매칭 코호트: {len(cohort)}명")
    print(f"  - 진단 데이터: {len(diag):,}건")

    # 코호트 환자의 진단 기록
    cohort_ids = set(cohort['연구번호'].unique())
    diag_cohort = diag[diag['연구번호'].isin(cohort_ids)]
    print(f"  - 코호트 환자 진단: {len(diag_cohort):,}건")

    # 병변 재발 추출
    print("\n[2. 병변 재발 추출]")
    print("  - 재발 기준: CIN2+, CIN3+, HSIL, CIS (Index date 이후)")

    outcomes = extract_lesion_recurrence(cohort, diag)

    # 결과 요약
    print("\n[3. 결과 요약]")
    for group_name, is_vax in [('접종군', True), ('비접종군', False)]:
        group = outcomes[outcomes['접종여부'] == is_vax]
        n = len(group)
        recurrence = group['has_recurrence'].sum()
        mean_fu = group['follow_up_days'].mean()
        recurred = group[group['has_recurrence'] == 1]
        mean_days_to_recur = recurred['days_to_recurrence'].mean() if len(recurred) > 0 else 0

        print(f"\n  {group_name} (n={n})")
        print(f"    - 병변 재발: {recurrence}건 ({recurrence/n*100:.1f}%)")
        print(f"    - 평균 추적기간: {mean_fu:.0f}일 ({mean_fu/365:.1f}년)")
        if recurrence > 0:
            print(f"    - 재발까지 평균 기간: {mean_days_to_recur:.0f}일 ({mean_days_to_recur/365:.1f}년)")

    # 저장
    print("\n[4. 결과 저장]")
    outcomes.to_csv(data_dir / 'final_matched_outcomes.csv', index=False, encoding='utf-8-sig')
    print(f"  - 저장: final_matched_outcomes.csv")

    print("\n" + "=" * 70)
    print("완료!")
    print("=" * 70)

    return outcomes


if __name__ == "__main__":
    outcomes = main()
