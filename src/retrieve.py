#가장 최신 retrieve


import re
import pickle
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import argparse
from typing import Optional

# =================================================================
# 1. 리소스 로드 (HuggingFace 모델, 데이터, Faiss 인덱스)
# =================================================================
try:
    # 모델 및 데이터 경로 설정
    model_path = r"C:\ai_work\RAG\snow\snowflake-arctic-embed-l-v2.0-ko"
    law_pkl_path = r"C:\ai_work\RAG\rag_final\data\산업안전보건법.pkl"
    law_index_path = r"C:\ai_work\RAG\rag_final\data\law_index1.faiss"
    basis_pkl_path = r"C:\ai_work\RAG\rag_final\data\산업안전보건기준법.pkl"
    basis_index_path = r"C:\ai_work\RAG\rag_final\data\law_index2.faiss"
    serious_pkl_path = r"C:\ai_work\RAG\rag_final\data\중대재해처벌법.pkl"
    serious_index_path = r"C:\ai_work\RAG\rag_final\data\law_index3.faiss"
    harbor_pkl_path = r"C:\ai_work\RAG\rag_final\data\항만법.pkl"
    harbor_index_path = r"C:\ai_work\RAG\rag_final\data\law_index4.faiss"
    shipping_pkl_path = r"C:\ai_work\RAG\rag_final\data\해운법.pkl"
    shipping_index_path = r"C:\ai_work\RAG\rag_final\data\law_index5.faiss"
    enforce_pkl_path = r"C:\ai_work\RAG\rag_final\data\산업안전보건법_시행령.pkl"
    enforce_index_path = r"C:\ai_work\RAG\rag_final\data\law_index6.faiss"

    # Sentence Transformer 모델 로드
    embedder = SentenceTransformer(model_path)

    # 법률 데이터 로드
    with open(law_pkl_path, "rb") as f: law_data = pickle.load(f)
    law_chunks, law_idx = law_data.get("chunks", []), faiss.read_index(law_index_path)
    with open(basis_pkl_path, "rb") as f: basis_data = pickle.load(f)
    basis_chunks, basis_idx = basis_data.get("chunks", []), faiss.read_index(basis_index_path)
    with open(serious_pkl_path, "rb") as f: serious_data = pickle.load(f)
    serious_chunks, serious_idx = serious_data.get("chunks", []), faiss.read_index(serious_index_path)
    with open(harbor_pkl_path, "rb") as f: harbor_data = pickle.load(f)
    harbor_chunks, harbor_idx = harbor_data.get("chunks", []), faiss.read_index(harbor_index_path)
    with open(shipping_pkl_path, "rb") as f: shipping_data = pickle.load(f)
    shipping_chunks, shipping_idx = shipping_data.get("chunks", []), faiss.read_index(shipping_index_path)
    with open(enforce_pkl_path, "rb") as f: enforce_data = pickle.load(f)
    enforce_chunks, enforce_idx = enforce_data.get("chunks", []), faiss.read_index(enforce_index_path)

except FileNotFoundError as e:
    print(f"오류: 파일을 찾을 수 없습니다. 경로를 확인해주세요. \n{e}")
    pass
except Exception as e:
    print(f"리소스 로딩 중 오류가 발생했습니다: {e}")
    exit()

# =================================================================
# 2. 유틸리티 함수 (조문 검색 및 제목 추출)
# =================================================================
def find_clause_chunk(chunks: list, question: str) -> Optional[str]:
    """
    질문에서 '제X조 Y항 Z호' 같은 패턴을 찾아 해당하는 텍스트 청크를 반환합니다.
    (수정) 조, 항, 호가 모두 일치하는 경우에만 결과를 반환하도록 로직 강화
    """
    m_art = re.search(r"(?:제)?\s*(\d+)\s*조(?:의\s*(\d+))?", question)
    if not m_art:
        return None
        
    main, sub = m_art.group(1), m_art.group(2)
    header = f"제{main}조" + (f"의{sub}" if sub else "")
    
    m_par = re.search(r"(?:제)?\s*(\d+)\s*항", question)
    par_hdr = f"제{m_par.group(1)}항." if m_par else None
    
    m_item = re.search(r"(?:제)?\s*(\d+)\s*호", question)
    item_hdr = f"제{m_item.group(1)}호." if m_item else None
    
    # 해당 '조'의 전체 내용을 찾는 정규식
    blk_pat = re.compile(rf"({re.escape(header)}\([^)]*\).*?)(?=(?:제\d+조(?:의\s*\d+)?\([^)]*\))|$)", re.DOTALL)

    for txt in chunks:
        # 1. 먼저 해당 '조'가 포함된 텍스트 블록 전체를 찾습니다.
        article_match = blk_pat.search(txt)
        if not article_match:
            continue  # 이 청크에는 해당 조가 없으므로 다음 청크로 넘어갑니다.

        current_segment = article_match.group(1).strip()

        # 2. 만약 질문에 '항'이 있다면, 찾은 텍스트 블록 안에서 '항'을 반드시 찾아야 합니다.
        if par_hdr:
            par_match = re.search(rf"({re.escape(par_hdr)}.*?)(?=(?:제\d+항\.|$))", current_segment, re.DOTALL)
            if not par_match:
                # '조'는 찾았지만 '항'을 못 찾았으면, 이번 청크는 실패입니다.
                continue
            # '항'을 찾았다면, 검색 범위를 해당 '항'의 내용으로 좁힙니다.
            current_segment = par_match.group(1).strip()

        # 3. 만약 질문에 '호'가 있다면, 현재 범위 안에서 '호'를 반드시 찾아야 합니다.
        if item_hdr:
            item_match = re.search(rf"({re.escape(item_hdr)}.*?)(?=(?:제\d+호\.|$))", current_segment, re.DOTALL)
            if not item_match:
                # '조'/'항'은 찾았지만 '호'를 못 찾았으면, 이번 청크는 실패입니다.
                continue
            # '호'를 찾았다면, 검색 범위를 해당 '호'의 내용으로 좁힙니다.
            current_segment = item_match.group(1).strip()
            
        # 4. 모든 조건을 통과했다면, 완벽하게 일치하는 결과이므로 반환합니다.
        return current_segment

    # 모든 청크를 다 찾아봤지만 완벽한 결과를 찾지 못했다면, None을 반환합니다.
    return None

def extract_article_title(text: str) -> Optional[str]:
    """
    텍스트에서 '제 O조(제목)' 또는 '제 O조의O(제목)' 패턴을 찾아 반환합니다.
    """
    if not text: return None
    pattern = r"제\s*\d+(?:의\d+)?조\([^)]+\)"
    match = re.search(pattern, text)
    if match: return match.group(0)
    return None

# =================================================================
# 3. 청크 검색 메인 함수 (보너스 점수 방식으로 수정)
# =================================================================
def retrieve_chunks(question: str, top_k: int = 10) -> list:
    """
    질문에 따라 적절한 법률 소스를 결정하고, 
    정확 매칭을 우선 시도한 후 의미론적 검색으로 폴백합니다.
    """
    # 질문에서 띄어쓰기 제거
    question_no_spaces = question.replace(' ', '')

    # 1) 검색할 소스 결정 및 키워드 확인
    has_basis = '기준법' in question_no_spaces or '기준에관한규칙' in question_no_spaces
    has_law = '보건법' in question_no_spaces
    has_serious = '중대재해처벌' in question_no_spaces
    has_harbor = '항만법' in question_no_spaces
    has_shipping = '해운법' in question_no_spaces
    has_enforce = '시행령' in question_no_spaces
    
    any_keyword_present = any([has_basis, has_law, has_serious, has_harbor, has_shipping, has_enforce])

    sources = []
    # 키워드가 없으면 전체 법률을, 있으면 해당 법률만 검색 대상으로 지정
    if not any_keyword_present:
        if 'basis_chunks' in globals(): sources.append(("산업안전보건기준에 관한 규칙", basis_chunks, basis_idx))
        if 'law_chunks' in globals(): sources.append(("산업안전보건법", law_chunks, law_idx))
        if 'serious_chunks' in globals(): sources.append(("중대재해처벌법", serious_chunks, serious_idx))
        if 'harbor_chunks' in globals(): sources.append(("항만법", harbor_chunks, harbor_idx))
        if 'shipping_chunks' in globals(): sources.append(("해운법", shipping_chunks, shipping_idx))
        if 'enforce_chunks' in globals(): sources.append(("산업안전보건법 시행령", enforce_chunks, enforce_idx))
    else:
        if has_basis and 'basis_chunks' in globals(): sources.append(("산업안전보건기준에 관한 규칙", basis_chunks, basis_idx))
        if has_law and 'law_chunks' in globals(): sources.append(("산업안전보건법", law_chunks, law_idx))
        if has_serious and 'serious_chunks' in globals(): sources.append(("중대재해처벌법", serious_chunks, serious_idx))
        if has_harbor and 'harbor_chunks' in globals(): sources.append(("항만법", harbor_chunks, harbor_idx))
        if has_shipping and 'shipping_chunks' in globals(): sources.append(("해운법", shipping_chunks, shipping_idx))
        if has_enforce and 'enforce_chunks' in globals(): sources.append(("산업안전보건법 시행령", enforce_chunks, enforce_idx))

    final_results = []
    exact_match_titles = set()
    
    # <-- 수정된 부분: 보너스 점수 시스템으로 변경
    BASE_SCORE = 2.0
    BONUS_SCORE = 0.5

    # 2) 정확 매칭 우선 검색
    for label, chunks_src, _ in sources:
        lex = find_clause_chunk(chunks_src, question)
        if lex:
            current_score = BASE_SCORE # 기본 점수 할당
            
            # 키워드와 법률이 일치하면 보너스 점수 추가
            if (has_basis and label == "산업안전보건기준에 관한 규칙") or \
               (has_law and label == "산업안전보건법") or \
               (has_serious and label == "중대재해처벌법") or \
               (has_harbor and label == "항만법") or \
               (has_shipping and label == "해운법") or \
               (has_enforce and label == "산업안전보건법 시행령"):
                current_score += BONUS_SCORE

            title = extract_article_title(lex)
            if title:
                exact_match_titles.add(title)
            
            final_results.append((f"[{label}] {lex}", current_score))

    # 정확 매칭 결과를 점수 높은 순으로 정렬
    final_results.sort(key=lambda x: x[1], reverse=True)

    # 3) 의미론적 검색 수행
    semantic_results = {}
    # 의미론적 검색은 키워드와 상관없이 모든 법률을 대상으로 수행하여 관련 정보를 놓치지 않도록 함
    all_sources = [
        ("산업안전보건기준에 관한 규칙", basis_chunks, basis_idx),
        ("산업안전보건법", law_chunks, law_idx),
        ("중대재해처벌법", serious_chunks, serious_idx),
        ("항만법", harbor_chunks, harbor_idx),
        ("해운법", shipping_chunks, shipping_idx),
        ("산업안전보건법 시행령", enforce_chunks, enforce_idx)
    ]
    
    query_with_prefix = "query: " + question
    q_emb = embedder.encode([query_with_prefix], convert_to_numpy=True)
    if q_emb.ndim == 1: q_emb = np.expand_dims(q_emb, axis=0)
    faiss.normalize_L2(q_emb)

    for label, chunks_src, idx_src in all_sources:
        if not chunks_src: continue # 데이터가 로드되지 않은 경우 건너뛰기
        D, I = idx_src.search(q_emb, top_k)
        for score, idx in zip(D[0], I[0]):
            if idx != -1:
                chunk_text = chunks_src[idx]
                semantic_title = extract_article_title(chunk_text)
                
                if semantic_title in exact_match_titles:
                    continue
                
                semantic_results[f"[{label}] {chunk_text}"] = float(score)

    # 4) 최종 결과 정렬 및 반환
    sorted_semantic = sorted(semantic_results.items(), key=lambda x: x[1], reverse=True)
    final_results.extend(sorted_semantic)
    return final_results[:top_k]

# =================================================================
# 4. 스크립트 실행 부분
# =================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="법률 문서 검색 시스템")
    parser.add_argument("--top_k", type=int, default=5, help="의미론적 검색 시 반환할 청크 수")
    args = parser.parse_args()

    print("법률 검색 시스템입니다. 질문을 입력하세요. (종료하려면 'exit' 입력)")
    
    while True:
        q = input("질문: ")
        if q.lower() == 'exit': break
        hits = retrieve_chunks(q, top_k=args.top_k)
        if not hits:
            print("관련 정보를 찾지 못했습니다.")
        else:
            print("\n[검색 결과]")
            for i, (txt, score) in enumerate(hits, 1):
                print(f"[{i}] 유사도 {score:.4f}\n{txt}")
                print(f"{'-'*50}")
        print("\n")