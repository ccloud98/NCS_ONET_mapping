"""
NCS-O*NET 매핑 정확도 개선 파이프라인

개선 항목:
  1. SOC 필터링          : 관련 SOC 대분류(11·13·15·17·19)로 O*NET 후보 제한
  2. O*NET 텍스트 보강   : Knowledge / Skills / Work Activities 고중요도 항목 추가
  3. NCS 청크 임베딩     : 긴 텍스트를 900자 청크로 분할 → 각각 임베딩 → 평균 풀링
  4. TF-IDF 앙상블       : 임베딩 유사도(65%) + 키워드 유사도(35%) 결합
"""

import sys, warnings
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
warnings.filterwarnings("ignore")

# ── 경로 설정 ──────────────────────────────────────────────────────────────
BASE      = Path(__file__).resolve().parents[1]
PROCESSED = BASE / "data" / "processed"
EMBED_DIR = PROCESSED / "embeddings"
RAW_DIR   = BASE / "data" / "raw"
ONET_DIR  = RAW_DIR / "db_30_2_excel"
OUTPUT    = BASE / "outputs"
IMPROVED  = OUTPUT / "improved"
IMPROVED.mkdir(parents=True, exist_ok=True)

MODEL_NAME  = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_CHARS = 900      # ≈ 225 tokens (256 max 기준 안전 마진)
CHUNK_OVERLAP = 100
EMBED_WEIGHT  = 0.65   # 앙상블 가중치
TFIDF_WEIGHT  = 0.35
TOP_K         = 20

# 관련 SOC 대분류
RELEVANT_SOC = {"11", "13", "15", "17", "19"}

print("=" * 65)
print("  NCS-O*NET 정확도 개선 파이프라인")
print("=" * 65)


# ══════════════════════════════════════════════════════════════════════════
# Step 1. 데이터 로드
# ══════════════════════════════════════════════════════════════════════════
print("\n[1/6] 데이터 로드 중...")

ncs_units = pd.read_csv(PROCESSED / "ncs_units_translated.csv", encoding="utf-8-sig")
onet      = pd.read_csv(PROCESSED / "onet_processed.csv",       encoding="utf-8-sig")
onet["final_text"] = onet["final_text"].fillna("").astype(str)

print(f"  NCS 능력단위 : {len(ncs_units)}개")
print(f"  O*NET 직업   : {len(onet)}개 (필터링 전)")


# ══════════════════════════════════════════════════════════════════════════
# Step 2. 개선 1 — SOC 필터링
# ══════════════════════════════════════════════════════════════════════════
print("\n[2/6] 개선 1: SOC 필터링...")

onet["soc_major"] = onet["O*NET-SOC Code"].str[:2]
onet_filtered = onet[onet["soc_major"].isin(RELEVANT_SOC)].copy().reset_index(drop=True)

soc_dist = onet_filtered["soc_major"].value_counts().sort_index()
print(f"  필터링 후 O*NET 직업 수: {len(onet_filtered)}개 / 1,016개")
for code, cnt in soc_dist.items():
    labels = {"11":"Management","13":"Business & Financial",
              "15":"Computer & Math","17":"Architecture & Eng",
              "19":"Life/Physical/Social Science"}
    print(f"    {code}-xxxx ({labels.get(code,'기타'):<30}): {cnt}개")


# ══════════════════════════════════════════════════════════════════════════
# Step 3. 개선 2 — O*NET 텍스트 보강
# ══════════════════════════════════════════════════════════════════════════
print("\n[3/6] 개선 2: O*NET 텍스트 보강 (Knowledge / Skills / Work Activities)...")

ENHANCED_PATH = PROCESSED / "onet_enhanced.csv"

if ENHANCED_PATH.exists():
    onet_enhanced = pd.read_csv(ENHANCED_PATH, encoding="utf-8-sig")
    onet_enhanced = onet_enhanced[onet_enhanced["O*NET-SOC Code"].isin(
        onet_filtered["O*NET-SOC Code"]
    )].reset_index(drop=True)
    print(f"  캐시 로드: {ENHANCED_PATH}")
else:
    def top_elements(filepath, scale="IM", threshold=4.0):
        """중요도(IM) 임계값 이상인 Element Name을 SOC별로 리스트로 반환."""
        df = pd.read_excel(filepath, engine="openpyxl")
        df.columns = df.columns.str.strip()
        top = (df[(df["Scale ID"] == scale) & (df["Data Value"] >= threshold)]
               .groupby("O*NET-SOC Code")["Element Name"]
               .apply(list).reset_index())
        return top

    print("  Knowledge.xlsx 로드...")
    kn_top = top_elements(ONET_DIR / "Knowledge.xlsx",       threshold=4.0)
    print("  Skills.xlsx 로드...")
    sk_top = top_elements(ONET_DIR / "Skills.xlsx",          threshold=4.0)
    print("  Work Activities.xlsx 로드...")
    wa_top = top_elements(ONET_DIR / "Work Activities.xlsx", threshold=4.5)

    onet_aug = (onet_filtered
                .merge(kn_top.rename(columns={"Element Name":"top_kn"}), on="O*NET-SOC Code", how="left")
                .merge(sk_top.rename(columns={"Element Name":"top_sk"}), on="O*NET-SOC Code", how="left")
                .merge(wa_top.rename(columns={"Element Name":"top_wa"}), on="O*NET-SOC Code", how="left"))

    def build_enhanced_text(row):
        parts = [str(row["final_text"])]
        if isinstance(row.get("top_wa"), list) and row["top_wa"]:
            parts.append("Key work activities: " + ". ".join(row["top_wa"][:10]) + ".")
        if isinstance(row.get("top_kn"), list) and row["top_kn"]:
            parts.append("Required knowledge: " + ", ".join(row["top_kn"][:8]) + ".")
        if isinstance(row.get("top_sk"), list) and row["top_sk"]:
            parts.append("Required skills: " + ", ".join(row["top_sk"][:8]) + ".")
        return " ".join(parts)

    onet_aug["enhanced_text"] = onet_aug.apply(build_enhanced_text, axis=1)
    onet_enhanced = onet_aug[["O*NET-SOC Code", "Title", "Description",
                               "final_text", "enhanced_text"]].copy()
    onet_enhanced.to_csv(ENHANCED_PATH, index=False, encoding="utf-8-sig")
    print(f"  저장: {ENHANCED_PATH}")

txt_lens = onet_enhanced["enhanced_text"].str.len()
print(f"  enhanced_text 평균 길이: {txt_lens.mean():.0f}자 (원본 final_text 평균 2,045자)")


# ══════════════════════════════════════════════════════════════════════════
# Step 4. 개선 3 — NCS 청크 임베딩
# ══════════════════════════════════════════════════════════════════════════
print("\n[4/6] 개선 3: NCS 청크 임베딩 (truncation 해결)...")

model = SentenceTransformer(MODEL_NAME)
print(f"  모델: {MODEL_NAME}  |  max_seq={model.max_seq_length} tokens")

NCS_CHUNK_EMBED_PATH = EMBED_DIR / "ncs_chunk_embeddings.npy"

def chunk_embed(text: str, mdl, chunk_chars=CHUNK_CHARS, overlap=CHUNK_OVERLAP) -> np.ndarray:
    """텍스트를 청크로 분할 후 각각 임베딩, 평균 반환."""
    chunks, i = [], 0
    while i < len(text):
        chunks.append(text[i : i + chunk_chars])
        i += chunk_chars - overlap
    if not chunks:
        chunks = [text[:chunk_chars]]
    vecs = mdl.encode(chunks, normalize_embeddings=False, show_progress_bar=False)
    mean_vec = vecs.mean(axis=0)
    # L2 재정규화
    norm = np.linalg.norm(mean_vec)
    return mean_vec / norm if norm > 0 else mean_vec

if NCS_CHUNK_EMBED_PATH.exists():
    ncs_chunk_emb = np.load(NCS_CHUNK_EMBED_PATH)
    print(f"  캐시 로드: {NCS_CHUNK_EMBED_PATH}")
else:
    ncs_texts = ncs_units["clean_text_en"].tolist()
    avg_chunks = int(np.mean([max(1, len(t) // CHUNK_CHARS) for t in ncs_texts]))
    print(f"  능력단위당 평균 청크 수 예상: {avg_chunks}개")
    ncs_chunk_emb = np.vstack([chunk_embed(t, model) for t in ncs_texts])
    np.save(NCS_CHUNK_EMBED_PATH, ncs_chunk_emb)
    print(f"  저장: {NCS_CHUNK_EMBED_PATH}")

print(f"  NCS 청크 임베딩 shape: {ncs_chunk_emb.shape}")


# ══════════════════════════════════════════════════════════════════════════
# Step 5. O*NET 보강 텍스트 임베딩 + 유사도 행렬
# ══════════════════════════════════════════════════════════════════════════
print("\n[5/6] O*NET 보강 임베딩 + 유사도 행렬 계산...")

ONET_ENH_EMBED_PATH = EMBED_DIR / "onet_enhanced_embeddings.npy"

if ONET_ENH_EMBED_PATH.exists():
    onet_enh_emb = np.load(ONET_ENH_EMBED_PATH)
    print(f"  캐시 로드: {ONET_ENH_EMBED_PATH}")
else:
    onet_texts_enh = onet_enhanced["enhanced_text"].tolist()
    print(f"  O*NET 보강 임베딩 생성 중 ({len(onet_texts_enh)}개)...")
    onet_enh_emb = model.encode(
        onet_texts_enh,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    np.save(ONET_ENH_EMBED_PATH, onet_enh_emb)
    print(f"  저장: {ONET_ENH_EMBED_PATH}")

# 임베딩 코사인 유사도
embed_sim = cosine_similarity(ncs_chunk_emb, onet_enh_emb)   # (51, N_filtered)
print(f"  임베딩 유사도 행렬: {embed_sim.shape}")


# ══════════════════════════════════════════════════════════════════════════
# Step 6. 개선 4 — TF-IDF 앙상블
# ══════════════════════════════════════════════════════════════════════════
print("\n[6/6] 개선 4: TF-IDF 앙상블 적용...")

ncs_texts_en   = ncs_units["clean_text_en"].tolist()
onet_texts_enh = onet_enhanced["enhanced_text"].tolist()

tfidf = TfidfVectorizer(
    max_features=15000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=1,
)
tfidf.fit(ncs_texts_en + onet_texts_enh)
ncs_tfidf  = tfidf.transform(ncs_texts_en)
onet_tfidf = tfidf.transform(onet_texts_enh)
tfidf_sim  = cosine_similarity(ncs_tfidf, onet_tfidf)        # (51, N_filtered)

# 앙상블 최종 유사도
final_sim = EMBED_WEIGHT * embed_sim + TFIDF_WEIGHT * tfidf_sim
print(f"  최종 유사도 행렬: {final_sim.shape}")
print(f"  값 범위: {final_sim.min():.4f} ~ {final_sim.max():.4f}")
print(f"  전체 평균: {final_sim.mean():.4f}")

# 행렬 저장
SIM_IMPROVED_PATH = EMBED_DIR / "sim_matrix_improved.npy"
np.save(SIM_IMPROVED_PATH, final_sim)


# ══════════════════════════════════════════════════════════════════════════
# Top-K 추출
# ══════════════════════════════════════════════════════════════════════════

def build_topk(sim_rows, meta_df, onet_df, k, groupby_col):
    records = []
    groups = meta_df.groupby(groupby_col).apply(lambda x: x.index.tolist())
    for group_name, idx_list in groups.items():
        agg = sim_rows[idx_list].mean(axis=0)
        for rank, onet_idx in enumerate(np.argsort(agg)[::-1][:k], 1):
            row = onet_df.iloc[onet_idx]
            records.append({
                groupby_col: group_name, "rank": rank,
                "O*NET-SOC Code": row["O*NET-SOC Code"],
                "O*NET Title": row["Title"],
                "cosine_similarity": round(float(agg[onet_idx]), 4),
                "O*NET Description": str(row.get("Description", ""))[:200],
            })
    return pd.DataFrame(records)

# 능력단위별 Top-K
unit_rows = []
for i, row in ncs_units.iterrows():
    scores = final_sim[i]
    for rank, onet_idx in enumerate(np.argsort(scores)[::-1][:TOP_K], 1):
        orow = onet_enhanced.iloc[onet_idx]
        unit_rows.append({
            "소분류코드명": row["소분류코드명"], "세분류코드명": row["세분류코드명"],
            "능력단위명칭": row["능력단위명칭"], "rank": rank,
            "O*NET-SOC Code": orow["O*NET-SOC Code"], "O*NET Title": orow["Title"],
            "cosine_similarity": round(float(scores[onet_idx]), 4),
            "O*NET Description": str(orow.get("Description", ""))[:200],
        })
topk_unit  = pd.DataFrame(unit_rows)
topk_det   = build_topk(final_sim, ncs_units, onet_enhanced, TOP_K, "세분류코드명")
topk_sub   = build_topk(final_sim, ncs_units, onet_enhanced, TOP_K, "소분류코드명")

# 저장
topk_unit.to_csv(IMPROVED / "imp_candidates_unit.csv",   index=False, encoding="utf-8-sig")
topk_det.to_csv( IMPROVED / "imp_candidates_detail.csv", index=False, encoding="utf-8-sig")
topk_sub.to_csv( IMPROVED / "imp_candidates_sub.csv",    index=False, encoding="utf-8-sig")

with pd.ExcelWriter(IMPROVED / "imp_mapping_candidates.xlsx", engine="openpyxl") as xw:
    topk_sub.to_excel(xw,  sheet_name="소분류_Top20",       index=False)
    topk_det.to_excel(xw,  sheet_name="세분류_Top20",       index=False)
    topk_unit.to_excel(xw, sheet_name="능력단위_전체_Top20", index=False)
    for det, grp in topk_unit.groupby("세분류코드명"):
        grp.to_excel(xw, sheet_name=det[:25], index=False)


# ══════════════════════════════════════════════════════════════════════════
# Before / After 비교 분석
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  Before / After 비교")
print("=" * 65)

orig_sim  = np.load(EMBED_DIR / "sim_matrix_unit_onet.npy")    # (51, 1016)
orig_unit = pd.read_csv(OUTPUT / "mapping_candidates_unit.csv", encoding="utf-8-sig")
orig_det  = pd.read_csv(OUTPUT / "mapping_candidates_detail.csv", encoding="utf-8-sig")

# Top-1 점수 비교
orig_top1_scores = orig_unit[orig_unit["rank"] == 1]["cosine_similarity"]
new_top1_scores  = topk_unit[topk_unit["rank"] == 1]["cosine_similarity"]

print(f"\n[Top-1 유사도 점수]")
print(f"  {'':30} {'개선 전':>8}  {'개선 후':>8}")
print(f"  {'평균':30} {orig_top1_scores.mean():>8.4f}  {new_top1_scores.mean():>8.4f}")
print(f"  {'중앙값':30} {orig_top1_scores.median():>8.4f}  {new_top1_scores.median():>8.4f}")
print(f"  {'최솟값':30} {orig_top1_scores.min():>8.4f}  {new_top1_scores.min():>8.4f}")

# 중복도 (Top-1 고유 직업 수)
orig_unique = orig_unit[orig_unit["rank"] == 1]["O*NET Title"].nunique()
new_unique  = topk_unit[topk_unit["rank"] == 1]["O*NET Title"].nunique()
print(f"\n[Top-1 고유 O*NET 직업 수 (다양성)]")
print(f"  개선 전: {orig_unique}개 / 51개 능력단위 → 중복률 {(51-orig_unique)/51*100:.1f}%")
print(f"  개선 후: {new_unique}개 / 51개 능력단위  → 중복률 {(51-new_unique)/51*100:.1f}%")

# SOC 분포 비교
SOC_LABELS = {"11":"Management","13":"Business & Financial","15":"Computer & Math",
              "17":"Architecture & Eng","19":"Life/Physical/Social Science",
              "21":"Community & Social Svc","23":"Legal","25":"Education",
              "27":"Arts & Media","33":"Protective Service","39":"Personal Care",
              "41":"Sales","43":"Office & Admin","47":"Construction",
              "51":"Production","53":"Transportation","31":"Healthcare Support"}

def top1_soc_dist(df):
    top1 = df[df["rank"] == 1].copy()
    top1["major"] = top1["O*NET-SOC Code"].str[:2]
    return top1["major"].value_counts()

orig_soc = top1_soc_dist(orig_unit)
new_soc  = top1_soc_dist(topk_unit)

print(f"\n[Top-1 SOC 대분류 분포 변화]")
all_majors = sorted(set(orig_soc.index) | set(new_soc.index))
print(f"  {'SOC':5} {'직군':<30} {'개선전':>5} → {'개선후':>5}")
print("  " + "-" * 50)
for m in all_majors:
    b = orig_soc.get(m, 0)
    a = new_soc.get(m, 0)
    diff = "(+)" if a > b else ("(-)" if a < b else "( )")
    label = SOC_LABELS.get(m, m)
    if b > 0 or a > 0:
        print(f"  {m}-xx  {label:<30} {b:>5}   {a:>5}  {diff}")

# 나쁜 매핑 제거 확인
BAD_MAPPINGS = {
    "현지 해외법인설립 절차 진행": "Real Estate Brokers",
    "해외취업 구직자 컨설팅": "Rehabilitation Counselors",
    "산학협력 지식재산권 운영관리": "Crushing, Grinding, and Polishing Machine Setters, Operators, and Tenders",
}
print(f"\n[알려진 오매핑 제거 확인]")
for unit_name, bad_title in BAD_MAPPINGS.items():
    # 개선 전
    orig_row = orig_unit[(orig_unit["능력단위명칭"] == unit_name) &
                         (orig_unit["rank"] <= 5)]
    was_present = bad_title in orig_row["O*NET Title"].values

    # 개선 후
    new_row = topk_unit[(topk_unit["능력단위명칭"] == unit_name) &
                        (topk_unit["rank"] <= 5)]
    still_present = bad_title in new_row["O*NET Title"].values

    status = "[OK] 제거됨" if (was_present and not still_present) else \
             ("[WARN] 여전히 존재" if still_present else "[--] 원래 없음")
    print(f"  [{unit_name}]")
    print(f"    {bad_title[:50]} -> {status}")

# Top-1 능력단위별 전체 비교 저장
comparison_rows = []
for i, row in ncs_units.iterrows():
    unit = row["능력단위명칭"]
    o1 = orig_unit[(orig_unit["능력단위명칭"] == unit) & (orig_unit["rank"] == 1)].iloc[0]
    n1 = topk_unit[(topk_unit["능력단위명칭"] == unit) & (topk_unit["rank"] == 1)].iloc[0]
    comparison_rows.append({
        "세분류코드명": row["세분류코드명"],
        "능력단위명칭": unit,
        "before_title": o1["O*NET Title"],
        "before_score": o1["cosine_similarity"],
        "after_title":  n1["O*NET Title"],
        "after_score":  n1["cosine_similarity"],
        "changed": "Y" if o1["O*NET Title"] != n1["O*NET Title"] else "N",
        "score_diff": round(n1["cosine_similarity"] - o1["cosine_similarity"], 4),
    })

comp_df = pd.DataFrame(comparison_rows)
comp_df.to_csv(IMPROVED / "before_after_comparison.csv", index=False, encoding="utf-8-sig")

print(f"\n[Top-1 변경 통계]")
print(f"  Top-1 직업 변경된 능력단위: {(comp_df['changed']=='Y').sum()}개 / 51개")
print(f"  점수 향상된 능력단위     : {(comp_df['score_diff']>0).sum()}개")
print(f"  점수 평균 변화           : {comp_df['score_diff'].mean():+.4f}")

# 세분류별 Top-5 결과 출력
print(f"\n[세분류별 개선 후 Top-5]")
for det in topk_det["세분류코드명"].unique():
    grp = topk_det[topk_det["세분류코드명"] == det].head(5)
    print(f"\n  [{det}]")
    for _, r in grp.iterrows():
        print(f"    {r['rank']:>2}. [{r['O*NET-SOC Code']}] {r['O*NET Title']:<45} {r['cosine_similarity']:.4f}")

print(f"\n저장 완료: {IMPROVED}")
print("=" * 65)
print("  개선 파이프라인 완료")
print("=" * 65)
