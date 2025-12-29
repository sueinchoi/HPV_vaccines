"""
병리 검사 데이터에서 주요 결과 변수 추출

연구 계획서 7.1 주요 결과변수 (Primary Outcomes):
1. 병변 재발: 조직검사로 확인된 HSIL/CIN3 이상 병변 재발
2. 새로운 고위험 HPV 감염: Index date 이후 HPV 양성 전환
   - 고위험 HPV 유형: 16, 18, 31, 33, 45, 52, 58, 35, 39, 51, 56, 59, 66, 68
"""

import pandas as pd
import re
from pathlib import Path


# 고위험 HPV 유형 정의
HIGH_RISK_HPV_TYPES = [16, 18, 31, 33, 45, 52, 58, 35, 39, 51, 56, 59, 66, 68]


def load_pathology_data(file_path: str) -> pd.DataFrame:
    """병리 검사 데이터 로드"""
    df = pd.read_csv(file_path, encoding='utf-8')

    # 날짜 컬럼 변환
    date_columns = ['처방일자', '실시일자', '판독일자']
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format='%Y%m%d', errors='coerce')

    return df


def detect_hsil_cin3_recurrence(result_text: str) -> dict:
    """
    판독결과에서 HSIL/CIN3 이상 병변 재발 여부 확인

    Returns:
        dict: {
            'is_hsil_cin3_or_higher': bool,
            'detected_lesion': str or None,
            'severity_level': str or None  # 'CIN2', 'CIN3', 'HSIL', 'Carcinoma'
        }
    """
    if pd.isna(result_text):
        return {
            'is_hsil_cin3_or_higher': False,
            'detected_lesion': None,
            'severity_level': None
        }

    result_text = str(result_text).upper()

    # 결과 저장용 변수
    detected_lesion = None
    severity_level = None

    # 1. Carcinoma (가장 심각) - 자궁경부암
    carcinoma_patterns = [
        r'SQUAMOUS\s*CELL\s*CARCINOMA',
        r'ADENOCARCINOMA',
        r'CERVICAL\s*CANCER',
        r'INVASIVE\s*CARCINOMA',
        r'CARCINOMA\s*IN\s*SITU',
        r'CIS\b',  # Carcinoma in situ
    ]

    for pattern in carcinoma_patterns:
        match = re.search(pattern, result_text)
        if match:
            detected_lesion = match.group()
            severity_level = 'Carcinoma'
            return {
                'is_hsil_cin3_or_higher': True,
                'detected_lesion': detected_lesion,
                'severity_level': severity_level
            }

    # 2. CIN3 / CIN III / CINIII 패턴
    cin3_patterns = [
        r'CIN\s*3\b',
        r'CIN\s*III\b',
        r'CINIII\b',
        r'CIN-3\b',
        r'CERVICAL\s*INTRAEPITHELIAL\s*NEOPLASIA\s*3',
        r'CERVICAL\s*INTRAEPITHELIAL\s*NEOPLASM[A]?\s*\(?CIN\)?\s*3',
    ]

    for pattern in cin3_patterns:
        match = re.search(pattern, result_text)
        if match:
            detected_lesion = match.group()
            severity_level = 'CIN3'
            return {
                'is_hsil_cin3_or_higher': True,
                'detected_lesion': detected_lesion,
                'severity_level': severity_level
            }

    # 3. HSIL (High-grade Squamous Intraepithelial Lesion)
    hsil_patterns = [
        r'HSIL\b',
        r'H-SIL\b',
        r'HIGH[\s-]*GRADE\s*SQUAMOUS\s*INTRAEPITHELIAL\s*LESION',
        r'HIGH[\s-]*GRADE\s*SIL\b',
    ]

    for pattern in hsil_patterns:
        match = re.search(pattern, result_text)
        if match:
            detected_lesion = match.group()
            severity_level = 'HSIL'
            return {
                'is_hsil_cin3_or_higher': True,
                'detected_lesion': detected_lesion,
                'severity_level': severity_level
            }

    # 4. CIN2 / CIN II / CINII (연구에 따라 HSIL에 포함)
    cin2_patterns = [
        r'CIN\s*2\b',
        r'CIN\s*II\b',
        r'CINII\b',
        r'CIN-2\b',
        r'CERVICAL\s*INTRAEPITHELIAL\s*NEOPLASIA\s*2',
        r'CIN\s*2/3\b',  # CIN 2/3 복합
        r'CIN\s*II/III\b',
    ]

    for pattern in cin2_patterns:
        match = re.search(pattern, result_text)
        if match:
            detected_lesion = match.group()
            severity_level = 'CIN2'
            return {
                'is_hsil_cin3_or_higher': True,
                'detected_lesion': detected_lesion,
                'severity_level': severity_level
            }

    return {
        'is_hsil_cin3_or_higher': False,
        'detected_lesion': None,
        'severity_level': None
    }


def detect_high_risk_hpv(result_text: str) -> dict:
    """
    판독결과에서 고위험 HPV 감염 여부 확인

    고위험 HPV 유형: 16, 18, 31, 33, 45, 52, 58, 35, 39, 51, 56, 59, 66, 68

    Returns:
        dict: {
            'is_high_risk_hpv_positive': bool,
            'detected_hpv_types': list,
            'hpv_result_detail': str or None
        }
    """
    if pd.isna(result_text):
        return {
            'is_high_risk_hpv_positive': False,
            'detected_hpv_types': [],
            'hpv_result_detail': None
        }

    result_text_upper = str(result_text).upper()
    result_text_orig = str(result_text)

    detected_types = []
    result_detail = None

    # 1. 개별 HPV 유형 양성 확인 (예: Positive(16), Positive(18))
    for hpv_type in HIGH_RISK_HPV_TYPES:
        patterns = [
            rf'POSITIVE\s*\(\s*{hpv_type}\s*\)',
            rf'HPV\s*{hpv_type}\s*:\s*POSITIVE',
            rf'HPV\s*{hpv_type}\s*양성',
            rf'TYPE\s*{hpv_type}\s*:\s*POSITIVE',
        ]
        for pattern in patterns:
            if re.search(pattern, result_text_upper):
                if hpv_type not in detected_types:
                    detected_types.append(hpv_type)

    # 2. "Positive(other)" - 16, 18 외 고위험 유형 (31,33,35,39,45,51,52,56,58,59,66,68)
    other_patterns = [
        r'POSITIVE\s*\(\s*OTHER\s*(?:TYPE)?\s*\)',
        r'POSITIVE\s*\(\s*OTHER\s*\)',
    ]
    for pattern in other_patterns:
        if re.search(pattern, result_text_upper):
            result_detail = "Other high-risk types detected"
            # 'other'는 16, 18 외 고위험 HPV를 의미
            other_types = [t for t in HIGH_RISK_HPV_TYPES if t not in [16, 18]]
            for t in other_types:
                if t not in detected_types:
                    detected_types.append(t)

    # 3. Pool 그룹 양성 확인 (P1, P2, P3)
    # P1: 33, 58 / P2: 56, 59, 66 / P3: 35, 39, 68
    pool_patterns = {
        'P1': [33, 58],
        'P2': [56, 59, 66],
        'P3': [35, 39, 68],
    }

    for pool, types in pool_patterns.items():
        pattern = rf'POSITIVE\s*\(\s*{pool}\s*\)'
        if re.search(pattern, result_text_upper):
            for t in types:
                if t not in detected_types:
                    detected_types.append(t)

    # 4. 일반적인 고위험 HPV 양성 표시
    general_positive_patterns = [
        r'HPV\s*:\s*POSITIVE\s*\(\s*HIGH[\s-]*RISK\s*\)',
        r'HIGH[\s-]*RISK\s*HPV\s*:\s*POSITIVE',
        r'HPV\s*HIGH[\s-]*RISK\s*:\s*POSITIVE',
        r'HIGH[\s-]*RISK\s*:\s*POSITIVE',
    ]

    for pattern in general_positive_patterns:
        if re.search(pattern, result_text_upper):
            if not detected_types:  # 구체적 유형이 없으면 일반 양성으로 표시
                result_detail = "High-risk HPV positive (unspecified type)"
                detected_types = ['high_risk_unspecified']

    # 5. HPV DNA Chip 결과에서 유형 추출
    chip_pattern = r'HPV\s*DNA\s*CHIP.*?POSITIVE'
    if re.search(chip_pattern, result_text_upper):
        # 개별 유형 번호 추출 시도
        type_matches = re.findall(r'\b(' + '|'.join(map(str, HIGH_RISK_HPV_TYPES)) + r')\b', result_text_orig)
        # 문맥상 양성 결과에서 나온 경우만 추가
        if 'POSITIVE' in result_text_upper:
            for t in type_matches:
                t_int = int(t)
                if t_int in HIGH_RISK_HPV_TYPES and t_int not in detected_types:
                    detected_types.append(t_int)

    # 6. Negative 결과 명시적 확인
    negative_patterns = [
        r'HPV\s*GENOTYPING\s*REAL[\s-]*TIME\s*PCR\s*:\s*NEGATIVE',
        r'HPV\s*:\s*NEGATIVE',
        r'\[\[RESULT\]\]\s*NEGATIVE',
        r'HIGH[\s-]*RISK\s*HPV\s*:\s*NEGATIVE',
    ]

    is_explicitly_negative = any(re.search(p, result_text_upper) for p in negative_patterns)

    # Negative가 명시된 경우 결과 초기화
    if is_explicitly_negative and not detected_types:
        return {
            'is_high_risk_hpv_positive': False,
            'detected_hpv_types': [],
            'hpv_result_detail': 'Negative'
        }

    # 결과 정리
    is_positive = len(detected_types) > 0

    # 숫자 유형만 필터링 (unspecified 제외하고 정렬)
    numeric_types = sorted([t for t in detected_types if isinstance(t, int)])

    return {
        'is_high_risk_hpv_positive': is_positive,
        'detected_hpv_types': numeric_types if numeric_types else detected_types,
        'hpv_result_detail': result_detail
    }


def extract_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """
    전체 병리 데이터에서 결과 변수 추출

    Returns:
        DataFrame with outcome columns added
    """
    # 병변 재발 (HSIL/CIN3+) 추출
    lesion_results = df['판독결과'].apply(detect_hsil_cin3_recurrence)
    df['is_hsil_cin3_or_higher'] = lesion_results.apply(lambda x: x['is_hsil_cin3_or_higher'])
    df['detected_lesion'] = lesion_results.apply(lambda x: x['detected_lesion'])
    df['severity_level'] = lesion_results.apply(lambda x: x['severity_level'])

    # 고위험 HPV 감염 추출
    hpv_results = df['판독결과'].apply(detect_high_risk_hpv)
    df['is_high_risk_hpv_positive'] = hpv_results.apply(lambda x: x['is_high_risk_hpv_positive'])
    df['detected_hpv_types'] = hpv_results.apply(lambda x: x['detected_hpv_types'])
    df['hpv_result_detail'] = hpv_results.apply(lambda x: x['hpv_result_detail'])

    return df


def get_patient_outcomes_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    환자별 결과 변수 요약

    각 환자(연구번호)별로:
    - 첫 번째 HSIL/CIN3+ 발생 일자
    - 첫 번째 고위험 HPV 양성 일자
    - 검출된 HPV 유형 목록
    """
    summary_list = []

    for patient_id in df['연구번호'].unique():
        patient_df = df[df['연구번호'] == patient_id].copy()
        patient_df = patient_df.sort_values('실시일자')

        # HSIL/CIN3+ 첫 발생
        hsil_positive = patient_df[patient_df['is_hsil_cin3_or_higher'] == True]
        first_hsil_date = hsil_positive['실시일자'].min() if len(hsil_positive) > 0 else None
        first_hsil_lesion = hsil_positive['detected_lesion'].iloc[0] if len(hsil_positive) > 0 else None
        first_hsil_severity = hsil_positive['severity_level'].iloc[0] if len(hsil_positive) > 0 else None

        # 고위험 HPV 양성 첫 발생
        hpv_positive = patient_df[patient_df['is_high_risk_hpv_positive'] == True]
        first_hpv_date = hpv_positive['실시일자'].min() if len(hpv_positive) > 0 else None

        # 모든 검출된 HPV 유형
        all_hpv_types = set()
        for types_list in hpv_positive['detected_hpv_types']:
            if isinstance(types_list, list):
                all_hpv_types.update([t for t in types_list if isinstance(t, int)])

        summary_list.append({
            '연구번호': patient_id,
            # 병변 재발 관련
            'has_hsil_cin3_recurrence': len(hsil_positive) > 0,
            'first_hsil_cin3_date': first_hsil_date,
            'first_detected_lesion': first_hsil_lesion,
            'first_severity_level': first_hsil_severity,
            'total_hsil_cin3_events': len(hsil_positive),
            # 고위험 HPV 관련
            'has_high_risk_hpv': len(hpv_positive) > 0,
            'first_high_risk_hpv_date': first_hpv_date,
            'detected_hpv_types': list(sorted(all_hpv_types)) if all_hpv_types else [],
            'total_hpv_positive_tests': len(hpv_positive),
        })

    return pd.DataFrame(summary_list)


def main():
    """메인 실행 함수"""
    # 데이터 경로 설정
    base_path = Path(__file__).parent.parent
    data_path = base_path / 'Data' / 'pathology_sample.csv'
    output_path = base_path / 'Data' / 'pathology_outcomes.csv'
    summary_path = base_path / 'Data' / 'patient_outcomes_summary.csv'

    print("=" * 60)
    print("병리 검사 데이터 결과 변수 추출")
    print("=" * 60)

    # 데이터 로드
    print(f"\n1. 데이터 로드: {data_path}")
    df = load_pathology_data(data_path)
    print(f"   - 전체 레코드 수: {len(df):,}")
    print(f"   - 환자 수: {df['연구번호'].nunique():,}")

    # 결과 변수 추출
    print("\n2. 결과 변수 추출 중...")
    df = extract_outcomes(df)

    # 결과 요약 출력
    print("\n3. 추출 결과 요약:")
    print("-" * 40)

    # 병변 재발 (HSIL/CIN3+)
    hsil_count = df['is_hsil_cin3_or_higher'].sum()
    hsil_patients = df[df['is_hsil_cin3_or_higher']]['연구번호'].nunique()
    print(f"   [병변 재발 (HSIL/CIN3+)]")
    print(f"   - 양성 레코드 수: {hsil_count:,}")
    print(f"   - 양성 환자 수: {hsil_patients:,}")

    if hsil_count > 0:
        severity_counts = df[df['is_hsil_cin3_or_higher']]['severity_level'].value_counts()
        print(f"   - 중증도별 분포:")
        for level, count in severity_counts.items():
            print(f"     * {level}: {count}")

    # 고위험 HPV 감염
    hpv_count = df['is_high_risk_hpv_positive'].sum()
    hpv_patients = df[df['is_high_risk_hpv_positive']]['연구번호'].nunique()
    print(f"\n   [고위험 HPV 감염]")
    print(f"   - 양성 레코드 수: {hpv_count:,}")
    print(f"   - 양성 환자 수: {hpv_patients:,}")

    # 환자별 요약 생성
    print("\n4. 환자별 결과 요약 생성 중...")
    summary_df = get_patient_outcomes_summary(df)

    # 결과 저장
    print("\n5. 결과 저장:")

    # 상세 결과 저장
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"   - 상세 결과: {output_path}")

    # 환자별 요약 저장
    summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
    print(f"   - 환자별 요약: {summary_path}")

    print("\n" + "=" * 60)
    print("추출 완료!")
    print("=" * 60)

    return df, summary_df


if __name__ == "__main__":
    df, summary_df = main()
