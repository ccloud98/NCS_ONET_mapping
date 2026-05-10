"""
Step 3+4: 매핑 기준 설계 & 차원별 공통점·차이점 분석

비교 차원 (영어 텍스트 기반):
  [A] 업무활동  : NCS "Tasks" (능력단위요소) ↔ O*NET Work Activities (IM>=3.5)
  [T] 수행과업  : NCS "Performance criteria" ↔ O*NET Task Statements
  [K] 지식      : NCS "Knowledge"            ↔ O*NET Knowledge (IM>=3.5)
  [S] 기술      : NCS "Skills"               ↔ O*NET Skills (IM>=3.5)
"""

import re, warnings
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
warnings.filterwarnings("ignore")

BASE     = Path(__file__).resolve().parents[1]
RAW      = BASE / "data" / "raw"
ONET_DIR = RAW / "db_30_2_excel"
PROC     = BASE / "data" / "processed"
OUT      = BASE / "outputs" / "dimensional"
OUT.mkdir(parents=True, exist_ok=True)

# 세분류 → Top-3 SOC (개선 파이프라인 결과)
TOP3_MAP = {
    "공적개발원조사업관리": [
        ("11-9041.00", "Architectural and Engineering Managers",   0.3938),
        ("13-1082.00", "Project Management Specialists",           0.3864),
        ("15-1299.09", "Information Technology Project Managers",  0.3842),
    ],
    "산학협력관리": [
        ("15-1299.09", "Information Technology Project Managers",  0.3060),
        ("11-9041.00", "Architectural and Engineering Managers",   0.2963),
        ("11-2021.00", "Marketing Managers",                       0.2950),
    ],
    "프로젝트관리": [
        ("13-1082.00", "Project Management Specialists",           0.4488),
        ("15-1299.09", "Information Technology Project Managers",  0.4423),
        ("11-9041.00", "Architectural and Engineering Managers",   0.4016),
    ],
    "해외법인설립관리": [
        ("11-3031.01", "Treasurers and Controllers",               0.3534),
        ("11-3031.03", "Investment Fund Managers",                 0.3324),
        ("11-3031.00", "Financial Managers",                       0.3289),
    ],
    "해외취업관리": [
        ("13-1071.00", "Human Resources Specialists",              0.3286),
        ("13-1111.00", "Management Analysts",                      0.3187),
        ("13-1151.00", "Training and Development Specialists",     0.3059),
    ],
}

DIM_LABEL = {
    "A": "업무활동[A]",
    "T": "수행과업[T]",
    "K": "지식    [K]",
    "S": "기술    [S]",
}

# ══════════════════════════════════════════════════════════════════
# 1. 데이터 로드
# ══════════════════════════════════════════════════════════════════
print("[1/4] 데이터 로드...")

ncs_t = pd.read_csv(PROC / "ncs_units_translated.csv", encoding="utf-8-sig")

kn_raw = pd.read_excel(ONET_DIR / "Knowledge.xlsx",       engine="openpyxl")
sk_raw = pd.read_excel(ONET_DIR / "Skills.xlsx",          engine="openpyxl")
wa_raw = pd.read_excel(ONET_DIR / "Work Activities.xlsx", engine="openpyxl")
ts_raw = pd.read_excel(ONET_DIR / "Task Statements.xlsx", engine="openpyxl")

# ══════════════════════════════════════════════════════════════════
# 2. NCS 차원별 텍스트 파싱 (영어 번역 텍스트 섹션 분할)
# ══════════════════════════════════════════════════════════════════
print("[2/4] NCS 차원별 텍스트 파싱...")

SECTION_KEYS = {
    "A": r"Tasks\s*:",               # 업무활동 ← 능력단위요소(Tasks 헤더)
    "T": r"Performance criteria\s*:", # 수행과업 ← 수행준거
    "K": r"Knowledge\s*:",            # 지식
    "S": r"Skills\s*:",               # 기술
    "att": r"Attitudes?\s*:",         # 태도 (파싱용, 분석에서 제외)
}

ORDER = ["A", "T", "K", "S", "att"]

def parse_sections(text: str) -> dict:
    """clean_text_en을 섹션별로 분리."""
    pattern = "|".join(f"({v})" for v in SECTION_KEYS.values())
    parts = re.split(pattern, text)

    result = {k: "" for k in ORDER}
    # 섹션 시작 위치 탐지
    anchors = [(m.group(), m.start()) for m in
               re.finditer("|".join(SECTION_KEYS.values()), text)]

    for i, (hdr, start) in enumerate(anchors):
        end = anchors[i+1][1] if i+1 < len(anchors) else len(text)
        content = text[start + len(hdr): end].strip().rstrip(".")
        for dim_key, pat in SECTION_KEYS.items():
            if re.match(pat, hdr, re.I):
                result[dim_key] = content
                break
    return result

def ncs_dim_texts_for_det(det_name: str) -> dict:
    """세분류의 모든 능력단위 텍스트를 파싱 후 차원별로 합산."""
    rows = ncs_t[ncs_t["세분류코드명"] == det_name]
    agg = {k: [] for k in ORDER}
    for _, row in rows.iterrows():
        secs = parse_sections(str(row["clean_text_en"]))
        for k in ORDER:
            if secs[k]:
                agg[k].append(secs[k])
    return {k: " ".join(set(agg[k])) for k in ORDER}

# ══════════════════════════════════════════════════════════════════
# 3. O*NET 차원별 텍스트 추출
# ══════════════════════════════════════════════════════════════════
IM_THR = 3.5

def onet_dim(soc: str) -> dict:
    def elem_text(df, threshold=IM_THR):
        sub = df[(df["O*NET-SOC Code"]==soc) & (df["Scale ID"]=="IM") &
                 (df["Data Value"]>=threshold)]
        return ". ".join(sub["Element Name"].tolist())
    tasks = ts_raw[ts_raw["O*NET-SOC Code"]==soc]["Task"].tolist()
    return {
        "A": elem_text(wa_raw),
        "T": ". ".join(tasks),
        "K": elem_text(kn_raw),
        "S": elem_text(sk_raw),
    }

# ══════════════════════════════════════════════════════════════════
# 4. 차원별 TF-IDF 유사도 + 키워드 분석
# ══════════════════════════════════════════════════════════════════
print("[3/4] 차원별 유사도 계산...")

def tfidf_sim(text_a: str, text_b: str) -> float:
    if not text_a.strip() or not text_b.strip():
        return 0.0
    try:
        v = TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True, min_df=1)
        m = v.fit_transform([text_a, text_b])
        return float(cosine_similarity(m[0], m[1])[0, 0])
    except Exception:
        return 0.0

def top_kw(text: str, n: int = 10) -> list:
    if not text.strip():
        return []
    try:
        v = TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True,
                            max_features=200, min_df=1,
                            stop_words="english")
        m = v.fit_transform([text])
        names = v.get_feature_names_out()
        idx = np.argsort(m.toarray()[0])[::-1][:n]
        return [names[i] for i in idx if m.toarray()[0][i] > 0]
    except Exception:
        return []

results = {}

for det, top3 in TOP3_MAP.items():
    ncs_dims = ncs_dim_texts_for_det(det)
    det_res = []

    for rank, (soc, title, total_score) in enumerate(top3, 1):
        on_dims = onet_dim(soc)
        dim_scores = {d: tfidf_sim(ncs_dims[d], on_dims[d])
                      for d in ("A", "T", "K", "S")}
        dim_kw = {}
        for d in ("A", "T", "K", "S"):
            nkw = set(top_kw(ncs_dims[d]))
            okw = set(top_kw(on_dims[d]))
            dim_kw[d] = {
                "ncs":    sorted(nkw)[:8],
                "onet":   sorted(okw)[:8],
                "common": sorted(nkw & okw),
                "ncs_only":  sorted(nkw - okw)[:5],
                "onet_only": sorted(okw - nkw)[:5],
            }
        det_res.append({
            "rank": rank, "soc": soc, "title": title,
            "total_score": total_score,
            "dim_scores": dim_scores,
            "dim_kw": dim_kw,
        })
    results[det] = det_res

# ══════════════════════════════════════════════════════════════════
# 5. 텍스트 보고서 생성
# ══════════════════════════════════════════════════════════════════
print("[4/4] 보고서 생성...")

SEP  = "=" * 78
SEP2 = "-" * 70

L = []

L.append(SEP)
L.append("  Step 3+4: Dimensional Mapping Analysis (NCS-O*NET)")
L.append(SEP)
L.append("")

# ── Step 3 ──────────────────────────────────────────────────────
L.append("[ Step 3 ] Comparison Dimension Framework")
L.append(SEP2)
dim_defs = [
    ("업무활동 [A]", "NCS: Tasks (능력단위요소)",
     "O*NET: Work Activities (IM>=3.5)",
     "상위 행동 범주. 직무를 수행하기 위해 반복적으로 사용되는 고수준 활동. "
     "NCS는 능력단위를 달성하기 위한 핵심 요소 행동, "
     "O*NET은 'Getting Information', 'Making Decisions' 등의 범주."),
    ("수행과업 [T]", "NCS: Performance criteria (수행준거)",
     "O*NET: Task Statements",
     "실제로 수행하는 구체적 과업. NCS는 '~할 수 있다' 형태의 검증 가능한 준거, "
     "O*NET은 '~한다' 형태의 직무 기술. 가장 직접적인 비교 차원."),
    ("지식     [K]", "NCS: Knowledge (지식 KSA)",
     "O*NET: Knowledge (IM>=3.5)",
     "직무 수행에 필요한 이론·원리·분야 지식. 학문 영역 키워드로 비교. "
     "두 체계 모두 같은 개념 사용."),
    ("기술     [S]", "NCS: Skills (기술 KSA)",
     "O*NET: Skills (IM>=3.5)",
     "직무 수행 방법론·도구 활용 역량. NCS는 도구·SW·절차 중심, "
     "O*NET은 인지·사회·기술적 역량으로 구분. 가장 큰 구조적 차이 발생 차원."),
]
for name, ncs_src, onet_src, note in dim_defs:
    L.append(f"  {name}")
    L.append(f"    NCS  : {ncs_src}")
    L.append(f"    O*NET: {onet_src}")
    L.append(f"    의미  : {note}")
    L.append("")

L.append("  [유사도 방법] TF-IDF bigram, sublinear_tf, stop_words=english -> cosine similarity")
L.append("  [키워드 방법] 각 차원 텍스트에서 TF-IDF 상위 10개 추출 후 교집합/차집합 분석")
L.append("")

# ── Step 4 ──────────────────────────────────────────────────────
L.append("[ Step 4 ] Dimensional Analysis by 세분류 (Top-3 O*NET)")
L.append(SEP)

for det, det_res in results.items():
    L.append("")
    L.append(f"[{det}]")
    L.append(SEP2)

    # 세분류 수준 NCS 차원 키워드 (한번만 출력)
    ncs_dims = ncs_dim_texts_for_det(det)
    L.append("  NCS 차원별 핵심 키워드")
    for d in ("A", "T", "K", "S"):
        kw_list = top_kw(ncs_dims[d], 8)
        L.append(f"    {DIM_LABEL[d]}: {', '.join(kw_list) or '(없음)'}")
    L.append("")

    for r in det_res:
        L.append(f"  >> Rank {r['rank']}: [{r['soc']}] {r['title']}")
        L.append(f"     종합 유사도(임베딩+TF-IDF앙상블): {r['total_score']:.4f}")
        L.append("")

        # 차원별 점수 바차트
        scores = r["dim_scores"]
        L.append(f"  {'차원':<18} {'유사도':>7}  {'시각화(0~1)':}")
        L.append("  " + "-" * 60)
        for d in ("A", "T", "K", "S"):
            sc = scores[d]
            bar_len = int(sc * 30)
            bar = "#" * bar_len + "." * (30 - bar_len)
            L.append(f"  {DIM_LABEL[d]:<18} {sc:>7.4f}  |{bar}|")
        avg4 = np.mean([scores[d] for d in "ATKS"])
        L.append(f"  {'4차원 평균':<18} {avg4:>7.4f}")
        L.append("")

        # 차원별 공통·차이 분석
        L.append("  [차원별 공통점·차이점]")
        for d in ("A", "T", "K", "S"):
            kw = r["dim_kw"][d]
            common    = ", ".join(kw["common"][:6])    or "(없음)"
            ncs_only  = ", ".join(kw["ncs_only"][:5])  or "(없음)"
            onet_only = ", ".join(kw["onet_only"][:5]) or "(없음)"
            L.append(f"  {DIM_LABEL[d]}")
            L.append(f"    공통      : {common}")
            L.append(f"    NCS 전용  : {ncs_only}")
            L.append(f"    O*NET 전용: {onet_only}")

        # 정성 요약
        best  = max(scores, key=scores.get)
        worst = min(scores, key=scores.get)
        L.append("")
        L.append("  [정성 요약]")
        L.append(f"    가장 유사한 차원  : {DIM_LABEL[best]} ({scores[best]:.4f})")
        L.append(f"    가장 차이 큰 차원 : {DIM_LABEL[worst]} ({scores[worst]:.4f})")

        # 공통점·차이점 서술
        common_total = sum(len(r["dim_kw"][d]["common"]) for d in "ATKS")
        L.append(f"    4차원 공통 키워드 총 {common_total}개")

        # 차원별 해석
        interp_map = {
            "A": "업무 행동 패턴",
            "T": "수행 과업 기술",
            "K": "요구 지식 영역",
            "S": "필요 기술 역량",
        }
        if scores[best] > 0.15:
            L.append(f"    -> {interp_map[best]} 측면에서 강한 유사성 확인")
        if scores[worst] < 0.05:
            L.append(f"    -> {interp_map[worst]} 측면에서 구조적 차이 존재")
        L.append("")
        L.append("  " + SEP2)

    L.append("")
    L.append(SEP)

# ── 종합 히트맵 ─────────────────────────────────────────────────
L.append("")
L.append("[ 종합 ] 세분류 x 차원별 평균 유사도 매트릭스")
L.append(SEP2)
hdr = f"  {'세분류':<20} {'[A]업무활동':>12} {'[T]수행과업':>12} {'[K]지식':>10} {'[S]기술':>10}  {'종합':>8}"
L.append(hdr)
L.append("  " + "-" * 76)
for det, det_res in results.items():
    row = {d: np.mean([r["dim_scores"][d] for r in det_res]) for d in "ATKS"}
    avg = np.mean([r["total_score"] for r in det_res])
    L.append(
        f"  {det[:20]:<20} {row['A']:>12.4f} {row['T']:>12.4f} "
        f"{row['K']:>10.4f} {row['S']:>10.4f}  {avg:>8.4f}"
    )

L.append("")
L.append("  [해석 기준]  > 0.30 강한유사  0.15~0.30 중간  0.05~0.15 약한  < 0.05 불일치")
L.append("")
L.append(SEP)
L.append("  END OF DIMENSIONAL ANALYSIS")
L.append(SEP)

report = "\n".join(L)
report_path = OUT / "dimensional_analysis.txt"
with open(report_path, "w", encoding="utf-8-sig") as f:
    f.write(report)

print(f"\n저장 완료: {report_path}")
# 터미널 출력 (ASCII 안전)
print(report.encode("ascii", errors="replace").decode("ascii"))
