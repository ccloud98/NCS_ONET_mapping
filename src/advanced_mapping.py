"""
NCS-O*NET 고도화 매핑 파이프라인 v3

[v2 대비 변경사항 -- 방법론적 취약점 직접 해결]

  취약점 1. SOC 필터링 순환성
    v2: 관련 SOC(11·13·15·17·19)를 사람이 사전 주입 → 그 가정을 확인하는 구조
    v3: 전체 O*NET 1,016개 대상 유사도 계산 → 결과 기반 분포 분석
        (SOC 구성은 사후 확인용으로만 사용)

  취약점 2. 유사도 점수 절대적 해석 문제 (Top-1 avg 0.4071)
    v2: all-MiniLM-L6-v2 (영어전용, 384d)
    v3: paraphrase-multilingual-mpnet-base-v2 (50+언어, 768d)
        → 더 강력한 의미 표현으로 절대 점수 향상

  취약점 3. 번역 의존성 비대칭
    v2: 한국어 NCS → 영어 번역 → 임베딩 (번역 오류·편향 미통제)
    v3: 한국어 NCS 원문 직접 임베딩 (번역 단계 제거)
        → 다국어 모델이 한국어·영어 동일 공간에서 비교

  추가 개선. 차원별 분리 임베딩 + 가중 합산
    [T] 수행과업 0.40 · [A] 업무활동 0.30 · [K] 지식 0.20 · [S] 기술 0.10
    (차원 분석에서 [T]가 구조적으로 가장 높은 정렬도 보임 → 최고 가중)
"""

import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
warnings.filterwarnings("ignore")

BASE      = Path(__file__).resolve().parents[1]
PROCESSED = BASE / "data" / "processed"
EMBED_DIR = PROCESSED / "embeddings"
RAW_DIR   = BASE / "data" / "raw"
ONET_DIR  = RAW_DIR / "db_30_2_excel"
ADV_OUT   = BASE / "outputs" / "advanced"
ADV_OUT.mkdir(parents=True, exist_ok=True)

MODEL_NAME  = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
TOP_K       = 20
DIM_WEIGHTS = {"A": 0.30, "T": 0.40, "K": 0.20, "S": 0.10}
IM_THR      = 3.5   # O*NET 중요도(IM) 임계값

SOC_LABELS = {
    "11": "Management",           "13": "Business & Financial",
    "15": "Computer & Math",      "17": "Architecture & Engineering",
    "19": "Life/Physical/Social", "21": "Community & Social Svc",
    "23": "Legal",                "25": "Education",
    "27": "Arts & Media",         "41": "Sales",
    "43": "Office & Admin",       "53": "Transportation",
}

print("=" * 70)
print("  NCS-O*NET 고도화 매핑 파이프라인 v3")
print("=" * 70)
print(f"  모델    : {MODEL_NAME}")
print(f"  가중치  : [T]{DIM_WEIGHTS['T']} [A]{DIM_WEIGHTS['A']} "
      f"[K]{DIM_WEIGHTS['K']} [S]{DIM_WEIGHTS['S']}")
print(f"  O*NET   : SOC 필터 없음, 전체 1,016개 대상")
print()


# ══════════════════════════════════════════════════════════════════
# 1. NCS 원문 로드 & 한국어 차원별 섹션 추출
# ══════════════════════════════════════════════════════════════════
print("[1/5] NCS 한국어 원문 로드 및 차원별 섹션 추출...")

ncs_raw = pd.read_excel(RAW_DIR / "NCS_DB.xlsx", engine="openpyxl")
ncs_raw = ncs_raw.fillna("")
print(f"  NCS_DB 로드 완료: {ncs_raw.shape}")

# 컬럼 자동 탐지 (전처리 리포트 기준 컬럼명 우선, 없으면 키워드 탐색)
def find_col(df, exact=None, keyword=None, exclude_kw=None):
    if exact and exact in df.columns:
        return exact
    if keyword:
        excl = exclude_kw or []
        cands = [c for c in df.columns
                 if keyword in c and all(e not in c for e in excl)]
        return cands[0] if cands else None
    return None

col_sub    = find_col(ncs_raw, exact="소분류코드명",   keyword="소분류", exclude_kw=["번호", "코드$"])
col_det    = find_col(ncs_raw, exact="세분류코드명",   keyword="세분류", exclude_kw=["번호", "코드$"])
col_unit   = find_col(ncs_raw, exact="능력단위명칭",   keyword="능력단위명칭")
col_elem   = find_col(ncs_raw, exact="능력단위요소명", keyword="능력단위요소명")
col_perf   = find_col(ncs_raw, exact="수행준거",       keyword="수행준거", exclude_kw=["번호"])
col_ksa_nm = find_col(ncs_raw, exact="지식기술태도코드명", keyword="지식기술태도코드명")
col_ksa_tx = find_col(ncs_raw, exact="지식기술태도의의",   keyword="지식기술태도의의")

required = {
    "소분류코드명": col_sub, "세분류코드명": col_det, "능력단위명칭": col_unit,
    "능력단위요소명": col_elem, "수행준거": col_perf,
    "KSA코드명": col_ksa_nm, "KSA내용": col_ksa_tx,
}
for name, col in required.items():
    status = col if col else "[탐지 실패]"
    print(f"    {name:<14} -> {status}")

missing = [n for n, c in required.items() if not c]
if missing:
    raise ValueError(f"필수 컬럼 탐지 실패: {missing}\n컬럼 목록: {ncs_raw.columns.tolist()}")

# 능력단위별 4개 차원 한국어 텍스트 추출
def unique_join(series):
    vals = [str(v).strip() for v in series if str(v).strip()]
    return " ".join(sorted(set(vals)))

# ncs_meta 기준 순서 유지 (세분류·능력단위 순서 일관성)
ncs_meta = pd.read_csv(PROCESSED / "ncs_units_translated.csv", encoding="utf-8-sig")
ordered_units = ncs_meta["능력단위명칭"].tolist()

records = []
ncs_grp = ncs_raw.groupby(col_unit, sort=False)

for unit_name in ordered_units:
    if unit_name not in ncs_grp.groups:
        print(f"  [경고] 능력단위 없음: {unit_name}")
        continue
    grp = ncs_grp.get_group(unit_name)
    mr  = grp.iloc[0]

    ksa = grp[[col_ksa_nm, col_ksa_tx]]
    ksa = ksa[ksa[col_ksa_tx].astype(str).str.strip() != ""]

    K_mask = ksa[col_ksa_nm].astype(str).str.contains("지식", na=False)
    S_mask = ksa[col_ksa_nm].astype(str).str.contains("기술", na=False)

    records.append({
        "소분류코드명": str(mr[col_sub]),
        "세분류코드명": str(mr[col_det]),
        "능력단위명칭": unit_name,
        "A": unique_join(grp[col_elem]),
        "T": unique_join(grp[col_perf]),
        "K": unique_join(ksa.loc[K_mask, col_ksa_tx]),
        "S": unique_join(ksa.loc[S_mask, col_ksa_tx]),
    })

ncs_sections = pd.DataFrame(records).reset_index(drop=True)
print(f"\n  능력단위 추출 완료: {len(ncs_sections)}개")
print(f"  {'차원':<6} {'빈항목':>6}  {'평균길이':>8}")
for d in "ATKS":
    empty = (ncs_sections[d].str.strip() == "").sum()
    avg   = ncs_sections[d].str.len().mean()
    print(f"  [{d}]    {empty:>5}개  {avg:>8.0f}자")


# ══════════════════════════════════════════════════════════════════
# 2. O*NET 차원별 섹션 추출 (전체 1,016개)
# ══════════════════════════════════════════════════════════════════
print("\n[2/5] O*NET 차원별 섹션 추출 (전체, SOC 필터 없음)...")

onet_meta    = pd.read_csv(PROCESSED / "onet_processed.csv", encoding="utf-8-sig")
ONET_SEC_CSV = PROCESSED / "onet_sections.csv"

if ONET_SEC_CSV.exists():
    onet_sections = pd.read_csv(ONET_SEC_CSV, encoding="utf-8-sig").fillna("")
    print(f"  캐시 로드: {ONET_SEC_CSV.name}  ({len(onet_sections)}개 직업)")
else:
    print("  O*NET XLSX 로드 중 (최초 1회)...")
    wa_raw = pd.read_excel(ONET_DIR / "Work Activities.xlsx", engine="openpyxl")
    ts_raw = pd.read_excel(ONET_DIR / "Task Statements.xlsx", engine="openpyxl")
    kn_raw = pd.read_excel(ONET_DIR / "Knowledge.xlsx",       engine="openpyxl")
    sk_raw = pd.read_excel(ONET_DIR / "Skills.xlsx",          engine="openpyxl")

    def im_group(df, thr=IM_THR, text_col="Element Name"):
        sub = df[(df["Scale ID"] == "IM") & (df["Data Value"] >= thr)]
        return (sub.groupby("O*NET-SOC Code")[text_col]
                .apply(lambda x: ". ".join(x.dropna()))
                .reset_index())

    wa_g = im_group(wa_raw).rename(columns={"Element Name": "A"})
    ts_g = (ts_raw.groupby("O*NET-SOC Code")["Task"]
            .apply(lambda x: ". ".join(x.dropna()))
            .reset_index().rename(columns={"Task": "T"}))
    kn_g = im_group(kn_raw).rename(columns={"Element Name": "K"})
    sk_g = im_group(sk_raw).rename(columns={"Element Name": "S"})

    onet_sections = (
        onet_meta[["O*NET-SOC Code", "Title"]]
        .merge(wa_g, on="O*NET-SOC Code", how="left")
        .merge(ts_g, on="O*NET-SOC Code", how="left")
        .merge(kn_g, on="O*NET-SOC Code", how="left")
        .merge(sk_g, on="O*NET-SOC Code", how="left")
        .fillna("")
    )
    onet_sections.to_csv(ONET_SEC_CSV, index=False, encoding="utf-8-sig")
    print(f"  저장: {ONET_SEC_CSV.name}")

print(f"  O*NET 직업 수: {len(onet_sections)}개 (전체)")


# ══════════════════════════════════════════════════════════════════
# 3. 다국어 임베딩 생성 -- 한국어 NCS & 영어 O*NET
# ══════════════════════════════════════════════════════════════════
print(f"\n[3/5] 다국어 임베딩 생성...")
print(f"  NCS  : 한국어 원문 직접 임베딩 (번역 없음)")
print(f"  O*NET: 영어 원문 임베딩")

model = SentenceTransformer(MODEL_NAME)
dim   = model.get_sentence_embedding_dimension()
print(f"  모델 로드  |  dim={dim}  max_seq={model.max_seq_length}")

def embed_section(texts, cache_path):
    if cache_path.exists():
        arr = np.load(cache_path)
        print(f"  캐시 {cache_path.name}  {arr.shape}")
        return arr
    safe = [t if str(t).strip() else " " for t in texts]
    arr  = model.encode(safe, batch_size=32, show_progress_bar=True,
                        normalize_embeddings=True)
    np.save(cache_path, arr)
    print(f"  저장 {cache_path.name}  {arr.shape}")
    return arr

print("\n  -- NCS 차원별 임베딩 (한국어) --")
ncs_emb = {d: embed_section(ncs_sections[d].tolist(),
                             EMBED_DIR / f"ncs_v3_{d}.npy") for d in "ATKS"}

print("\n  -- O*NET 차원별 임베딩 (영어) --")
onet_emb = {d: embed_section(onet_sections[d].tolist(),
                              EMBED_DIR / f"onet_v3_{d}.npy") for d in "ATKS"}


# ══════════════════════════════════════════════════════════════════
# 4. 차원별 가중 유사도 행렬
# ══════════════════════════════════════════════════════════════════
print("\n[4/5] 차원별 가중 유사도 계산...")

dim_sim = {}
for d in "ATKS":
    dim_sim[d] = cosine_similarity(ncs_emb[d], onet_emb[d])  # (51, 1016)
    mn, mx, av = dim_sim[d].min(), dim_sim[d].max(), dim_sim[d].mean()
    print(f"  [{d}] {dim_sim[d].shape}  범위 {mn:.4f}~{mx:.4f}  평균 {av:.4f}")

final_sim = sum(DIM_WEIGHTS[d] * dim_sim[d] for d in "ATKS")  # (51, 1016)

top1 = final_sim.max(axis=1)
print(f"\n  최종 가중 유사도: {final_sim.shape}")
print(f"  범위     : {final_sim.min():.4f} ~ {final_sim.max():.4f}")
print(f"  Top-1 평균: {top1.mean():.4f}  중앙값: {np.median(top1):.4f}  최솟값: {top1.min():.4f}")

np.save(EMBED_DIR / "sim_matrix_v3.npy", final_sim)

# SOC 분포 확인 (사전 주입 없이 결과에서 분포 관찰)
top1_soc = []
for i in range(len(ncs_sections)):
    best_oi = int(np.argmax(final_sim[i]))
    soc = onet_sections.iloc[best_oi]["O*NET-SOC Code"][:2]
    top1_soc.append(soc)

from collections import Counter
soc_dist = Counter(top1_soc)
print(f"\n  [Top-1 SOC 분포 -- 필터 없이 모델이 선택한 결과]")
print(f"  {'SOC':>5}  {'직군':<30}  {'건수':>5}")
for code, cnt in sorted(soc_dist.items(), key=lambda x: -x[1]):
    label = SOC_LABELS.get(code, f"SOC-{code}")
    bar = "#" * cnt
    print(f"  {code}-xx   {label:<30}  {cnt:>3}건  {bar}")


# ══════════════════════════════════════════════════════════════════
# 5. Top-K 추출 및 저장
# ══════════════════════════════════════════════════════════════════
print("\n[5/5] Top-K 추출 및 저장...")

unit_rows = []
for i, nrow in ncs_sections.iterrows():
    scores   = final_sim[i]
    top_idxs = np.argsort(scores)[::-1][:TOP_K]
    for rank, oi in enumerate(top_idxs, 1):
        orow = onet_sections.iloc[oi]
        d_sc = {f"sim_{d}": round(float(dim_sim[d][i, oi]), 4) for d in "ATKS"}
        unit_rows.append({
            "소분류코드명":       nrow["소분류코드명"],
            "세분류코드명":       nrow["세분류코드명"],
            "능력단위명칭":       nrow["능력단위명칭"],
            "rank":              rank,
            "O*NET-SOC Code":    orow["O*NET-SOC Code"],
            "O*NET Title":       orow["Title"],
            "cosine_similarity": round(float(scores[oi]), 4),
            **d_sc,
        })

topk_unit = pd.DataFrame(unit_rows)

def build_topk_grp(sim_mat, dim_sims, ncs_df, onet_df, k, col):
    rows = []
    for grp_name, grp in ncs_df.groupby(col, sort=False):
        idx = grp.index.tolist()
        agg = sim_mat[idx].mean(axis=0)
        for rank, oi in enumerate(np.argsort(agg)[::-1][:k], 1):
            orow = onet_df.iloc[oi]
            d_sc = {f"sim_{d}": round(float(dim_sims[d][idx].mean(axis=0)[oi]), 4)
                    for d in "ATKS"}
            rows.append({
                col:                 grp_name,
                "rank":              rank,
                "O*NET-SOC Code":    orow["O*NET-SOC Code"],
                "O*NET Title":       orow["Title"],
                "cosine_similarity": round(float(agg[oi]), 4),
                **d_sc,
            })
    return pd.DataFrame(rows)

topk_det = build_topk_grp(final_sim, dim_sim, ncs_sections, onet_sections, TOP_K, "세분류코드명")
topk_sub = build_topk_grp(final_sim, dim_sim, ncs_sections, onet_sections, TOP_K, "소분류코드명")

topk_unit.to_csv(ADV_OUT / "adv_candidates_unit.csv",   index=False, encoding="utf-8-sig")
topk_det.to_csv( ADV_OUT / "adv_candidates_detail.csv", index=False, encoding="utf-8-sig")
topk_sub.to_csv( ADV_OUT / "adv_candidates_sub.csv",    index=False, encoding="utf-8-sig")

with pd.ExcelWriter(ADV_OUT / "adv_mapping_candidates.xlsx", engine="openpyxl") as xw:
    topk_sub.to_excel(xw,  sheet_name="소분류_Top20",        index=False)
    topk_det.to_excel(xw,  sheet_name="세분류_Top20",        index=False)
    topk_unit.to_excel(xw, sheet_name="능력단위_전체_Top20",  index=False)
    for det, grp in topk_unit.groupby("세분류코드명"):
        grp.to_excel(xw, sheet_name=str(det)[:25], index=False)


# ══════════════════════════════════════════════════════════════════
# 결과 출력
# ══════════════════════════════════════════════════════════════════
SEP = "=" * 70

print("\n" + SEP)
print("  세분류별 Top-5 결과 (v3 -- 다국어 모델 + 차원 가중)")
print(SEP)
for det in ncs_sections["세분류코드명"].unique():
    grp = topk_det[topk_det["세분류코드명"] == det].head(5)
    print(f"\n  [{det}]")
    print(f"  {'순위':<4} {'SOC Code':<14} {'O*NET Title':<42} {'종합':>6}  [T]   [A]   [K]   [S]")
    print("  " + "-" * 88)
    for _, r in grp.iterrows():
        print(f"  {int(r['rank']):<4} {r['O*NET-SOC Code']:<14} {str(r['O*NET Title'])[:40]:<42} "
              f"{r['cosine_similarity']:>6.4f}  "
              f"{r['sim_T']:.3f} {r['sim_A']:.3f} {r['sim_K']:.3f} {r['sim_S']:.3f}")

# v2 vs v3 비교
print("\n" + SEP)
print("  v2 (SOC필터 + 영어전용) vs v3 (전체 + 다국어 + 차원가중) 비교")
print(SEP)
try:
    v2_det = pd.read_csv(BASE / "outputs" / "improved" / "imp_candidates_detail.csv",
                         encoding="utf-8-sig")
    print(f"  {'세분류':<22}  {'v2 Top-1':<38}  {'점수':>6}  |  {'v3 Top-1':<38}  {'점수':>6}")
    print("  " + "-" * 120)
    for det in ncs_sections["세분류코드명"].unique():
        v2r = v2_det[v2_det["세분류코드명"] == det].head(1)
        v3r = topk_det[topk_det["세분류코드명"] == det].head(1)
        if v2r.empty or v3r.empty:
            continue
        v2r, v3r = v2r.iloc[0], v3r.iloc[0]
        same = "(동일)" if v2r["O*NET-SOC Code"] == v3r["O*NET-SOC Code"] else "      "
        print(f"  {det:<22}  {str(v2r['O*NET Title'])[:36]:<38}  {v2r['cosine_similarity']:>6.4f}"
              f"  |  {str(v3r['O*NET Title'])[:36]:<38}  {v3r['cosine_similarity']:>6.4f}  {same}")
except FileNotFoundError:
    print("  v2 결과 파일 없음 -- 비교 생략")

# Top-1 유사도 통계 비교
print("\n" + SEP)
print("  점수 분포 비교 (v2 vs v3)")
print(SEP)
v3_top1 = topk_unit[topk_unit["rank"] == 1]["cosine_similarity"]
try:
    v2_unit = pd.read_csv(BASE / "outputs" / "improved" / "imp_candidates_unit.csv",
                          encoding="utf-8-sig")
    v2_top1 = v2_unit[v2_unit["rank"] == 1]["cosine_similarity"]
    print(f"  {'지표':<12}  {'v2':>8}  {'v3':>8}")
    print("  " + "-" * 32)
    print(f"  {'Top-1 평균':<12}  {v2_top1.mean():>8.4f}  {v3_top1.mean():>8.4f}")
    print(f"  {'Top-1 중앙값':<12}  {v2_top1.median():>8.4f}  {v3_top1.median():>8.4f}")
    print(f"  {'Top-1 최솟값':<12}  {v2_top1.min():>8.4f}  {v3_top1.min():>8.4f}")
    print(f"  {'Top-1 최댓값':<12}  {v2_top1.max():>8.4f}  {v3_top1.max():>8.4f}")
    n_unique_v2 = v2_unit[v2_unit["rank"] == 1]["O*NET Title"].nunique()
    n_unique_v3 = topk_unit[topk_unit["rank"] == 1]["O*NET Title"].nunique()
    print(f"  {'Top-1 다양성':<12}  {n_unique_v2:>8}개  {n_unique_v3:>8}개  (고유 직업 수 / 51)")
except FileNotFoundError:
    print(f"  v3 Top-1 평균: {v3_top1.mean():.4f}  중앙값: {v3_top1.median():.4f}")

print(f"\n  저장 완료: {ADV_OUT}")
print(SEP)
print("  v3 파이프라인 완료")
print(SEP)
