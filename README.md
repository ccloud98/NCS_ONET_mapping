# NCS–O\*NET 직무 매핑 프로젝트

한국 국가직무능력표준(NCS) **사업관리** 분야의 능력단위를 미국 직업정보망(O\*NET) 직업과 의미적으로 매핑하는 자동화 파이프라인입니다.

---

## 프로젝트 개요

NCS와 O\*NET은 직무 역량을 기술하는 구조와 언어가 달라 단순 번역이나 키워드 매칭으로는 정확한 대응이 어렵습니다. 본 프로젝트는 Sentence-BERT 임베딩, TF-IDF 앙상블, 차원별 비교 분석을 결합하여 두 체계 간 최적 매핑 후보를 자동으로 생성합니다.

### 분석 대상

| 구분 | 설명 |
|---|---|
| NCS 도메인 | 사업관리 (5개 세분류, 51개 능력단위) |
| 세분류 | 프로젝트관리 / 산학협력관리 / 공적개발원조사업관리 / 해외법인설립관리 / 해외취업관리 |
| O\*NET 직업 | 전체 1,016개 → SOC 필터링 후 272개 (관련 SOC 11·13·15·17·19) |

### 분석 단계

```
Step 1. 데이터 전처리     — NCS 원본 텍스트 영어 번역, O*NET 구조화
Step 2. 유사도 매핑       — Sentence-BERT + TF-IDF 앙상블로 Top-20 후보 생성
Step 3. 매핑 기준 설계    — 4개 비교 차원 정의 (업무활동 / 수행과업 / 지식 / 기술)
Step 4. 공통점·차이점 분석 — 차원별 정량 점수 + 정성 해석
```

---

## 디렉토리 구조

```
ncs_onet_mapping/
├── data/
│   ├── raw/
│   │   ├── NCS_DB.xlsx              # NCS 사업관리 원본 (190 MB, git 제외)
│   │   └── db_30_2_excel/           # O*NET 30.2 전체 데이터셋 (git 제외)
│   └── processed/                   # 전처리 결과 (git 제외, 스크립트로 재생성)
│       ├── ncs_units_translated.csv # 능력단위 영어 번역 텍스트 (51행)
│       ├── onet_processed.csv       # O*NET 기본 전처리 결과
│       ├── onet_enhanced.csv        # O*NET 텍스트 보강 결과
│       └── embeddings/              # Sentence-BERT 임베딩 캐시 (.npy)
│
├── notebooks/
│   ├── 01_data_preprocessing.ipynb  # Step 1: 전처리 파이프라인
│   └── 02_similarity_mapping.ipynb  # Step 2: 유사도 매핑 파이프라인
│
├── src/
│   ├── improved_mapping.py          # Step 2 개선 파이프라인 (SOC 필터 + 앙상블)
│   └── dimensional_analysis.py      # Step 3+4 차원별 비교 분석
│
├── outputs/                         # 생성된 매핑 결과 (git 제외, 재생성 가능)
│   ├── improved/                    # 개선 파이프라인 출력물
│   │   ├── imp_candidates_unit.csv  # 능력단위별 Top-20 후보
│   │   ├── imp_candidates_detail.csv
│   │   ├── imp_candidates_sub.csv
│   │   ├── imp_mapping_candidates.xlsx
│   │   └── before_after_comparison.csv
│   └── dimensional/                 # 차원 분석 출력물
│       ├── dimensional_analysis.txt
│       └── dimensional_analysis.json
│
├── data_preprocessing_report.txt    # Step 1 실행 결과 보고서
├── similarity_mapping_report.txt    # Step 2 유사도 매핑 보고서
├── dimensional_mapping_report.txt   # Step 3+4 차원 분석 보고서
├── requirements.txt
└── .gitignore
```

---

## 핵심 방법론

### Sentence-BERT 임베딩

- 모델: `sentence-transformers/all-MiniLM-L6-v2` (384차원)
- NCS 텍스트는 900자 청크로 분할 후 각 청크를 개별 임베딩 → 평균 풀링 (256 토큰 한계 우회)
- O\*NET 텍스트는 Knowledge / Skills / Work Activities (IM ≥ 4.0/4.5) 항목으로 보강

### TF-IDF 앙상블

```
최종 유사도 = 임베딩 코사인 유사도 × 0.65 + TF-IDF 코사인 유사도 × 0.35
```

TF-IDF: bigram, sublinear_tf=True, stop_words=english

### SOC 필터링

비즈니스·IT·경영 관련 SOC 대분류만 유지하여 의미 없는 직업군 제거:

| SOC 코드 | 직업군 |
|---|---|
| 11 | Management Occupations |
| 13 | Business & Financial Operations |
| 15 | Computer & Mathematical |
| 17 | Architecture & Engineering |
| 19 | Life, Physical, and Social Science |

### 4개 비교 차원

| 차원 | NCS 구성 요소 | O\*NET 구성 요소 |
|---|---|---|
| [A] 업무활동 | 능력단위요소명 | Work Activities (IM ≥ 3.5) |
| [T] 수행과업 | 수행준거 | Task Statements |
| [K] 지식 | 지식 KSA | Knowledge (IM ≥ 3.5) |
| [S] 기술 | 기술 KSA | Skills (IM ≥ 3.5) |

---

## 환경 설정 및 실행

### 요구 사항

- Python 3.9 이상
- 원본 데이터 파일 (`data/raw/` 하위) 별도 준비 필요

### 패키지 설치

```bash
pip install -r requirements.txt
```

### 실행 순서

#### Step 1: 데이터 전처리

```bash
jupyter notebook notebooks/01_data_preprocessing.ipynb
```

실행 결과:
- `data/processed/ncs_units_translated.csv`
- `data/processed/onet_processed.csv`

#### Step 2: 유사도 매핑 (기본)

```bash
jupyter notebook notebooks/02_similarity_mapping.ipynb
```

#### Step 2: 유사도 매핑 (개선 파이프라인)

SOC 필터링, 텍스트 보강, 청크 임베딩, TF-IDF 앙상블을 모두 적용한 개선 버전:

```bash
python src/improved_mapping.py
```

실행 결과: `outputs/improved/` 하위 CSV 및 XLSX 파일

#### Step 3+4: 차원별 비교 분석

```bash
python src/dimensional_analysis.py
```

실행 결과: `outputs/dimensional/dimensional_analysis.txt`, `dimensional_analysis.json`

### 임베딩 캐시

`data/processed/embeddings/` 에 `.npy` 파일이 있으면 재사용합니다. 삭제하면 처음부터 재계산합니다 (Sentence-BERT 추론 시간: CPU 기준 약 5~10분).

---

## 주요 결과 요약

| 항목 | 기본 파이프라인 | 개선 파이프라인 |
|---|---|---|
| O\*NET 후보 수 | 1,016개 | 272개 (SOC 필터) |
| Top-1 평균 유사도 | 0.5767 | 0.4071 |
| 유사도 범위 | 0.36 ~ 0.73 | -0.09 ~ 0.52 |
| 오매핑 제거 | - | Real Estate Brokers 등 3건 제거 확인 |

> Top-1 평균 유사도 수치가 낮아진 것은 관련 없는 직업군을 제거한 후 진짜 유사한 직업만 남아 분포가 좁아졌기 때문입니다.

---

## 보고서 파일

| 파일 | 내용 |
|---|---|
| `data_preprocessing_report.txt` | NCS·O\*NET 데이터 구조, 전처리 과정 상세 |
| `similarity_mapping_report.txt` | 유사도 매핑 방법론, 세분류별 Top-5 결과, 개선 전후 비교 |
| `dimensional_mapping_report.txt` | 4개 차원 정의 근거, 15개 NCS-O\*NET 쌍 차원별 분석, 종합 결과 |

---

## 데이터 출처

- **NCS**: 국가직무능력표준 (https://www.ncs.go.kr)
- **O\*NET**: O\*NET Resource Center, Release 30.2 (https://www.onetonline.org)
