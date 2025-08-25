"""
RAG 기반 법령 질문 & 위험성평가 챗봇의 메인 실행 파일입니다.
"""
import argparse
import re
import json
import time
import string
from typing import Tuple
from llama_cpp import Llama
from langdetect import detect_langs

# 다른 파이썬 파일에서 기능별 함수들을 가져옵니다.
from risk import handle_risk_assessment
from general import handle_general_question

# =====================================================================
# 설정값 및 시스템 프롬프트
# =====================================================================
MIN_QUERY_LENGTH = 3
REPETITIVE_CHAR_LIMIT = 3
FILTER_WORDS = ['테스트', 'test']
SYSTEM_INSTRUCTION = (
    """
당신은 한국 산업안전보건법·KRAS 3x3 위험성 평가 전문가입니다.
- 법령 관련 질문에는 정확한 조문을 인용해 설명합니다.
- 위험성평가 요청 시에는 아래 Task-level 지침을 준수합니다.
"""
)

# =====================================================================
# ★★★ 도우미 및 유틸리티 함수 그룹 ★★★
# =====================================================================
def load_model(model_path: str) -> Llama:
    """GGUF 모델을 로드합니다."""
    return Llama(
        model_path=model_path,
        n_ctx=16384,
        verbose=False,
        n_gpu_layers=-1
    )

def is_invalid_query(question: str) -> Tuple[bool, str]:
    """규칙 기반으로 질문의 유효성을 검증합니다."""
    clean_question = question.strip()
    law_pattern = re.compile(r'(?:제)?\s*\d+\s*조(?:\s*제?\s*\d+\s*항)?(?:\s*제?\s*\d+\s*호)?')
    if law_pattern.search(clean_question):
        return False, "유효한 법령 조항 질문입니다."

    if len(clean_question) < MIN_QUERY_LENGTH: return True, "질문이 너무 짧습니다."
    if re.search(r'(.)\1{' + str(REPETITIVE_CHAR_LIMIT - 1) + r',}', clean_question): return True, "반복적인 문자로만 이루어진 질문입니다."
    mixed_pattern = re.findall(r'\d+[a-zA-Z]+|\d+[ㄱ-ㅎㅏ-ㅣ가-힣]+', clean_question)
    if len(mixed_pattern) > len(clean_question.split()) * 0.5: return True, "입력된 텍스트가 유효하지 않은 패턴을 포함하고 있습니다."
    jamo_only = re.findall(r'[ㄱ-ㅎㅏ-ㅣ]{2,}', clean_question)
    if jamo_only and sum(len(j) for j in jamo_only) > len(clean_question) * 0.3: return True, "입력된 텍스트가 유효하지 않은 패턴을 포함하고 있습니다."
    alpha_chars = [c for c in clean_question if c.isalpha() and c in string.ascii_letters]
    if alpha_chars:
        lowercase_ratio = sum(1 for c in alpha_chars if c.islower()) / len(alpha_chars)
        if lowercase_ratio < 0.2 or lowercase_ratio > 0.95: return True, "입력된 텍스트가 유효하지 않은 패턴을 포함하고 있습니다."
    try:
        detected = detect_langs(clean_question)
        if all(lang.prob < 0.5 for lang in detected): return True, "입력된 텍스트가 유효하지 않은 패턴을 포함하고 있습니다."
    except: return True, "입력된 텍스트가 유효하지 않은 패턴을 포함하고 있습니다."
    if len(set(clean_question)) < len(clean_question) * 0.2: return True, "입력된 텍스트가 유효하지 않은 패턴을 포함하고 있습니다."
    if clean_question.lower() in FILTER_WORDS: return True, "테스트 질문은 처리하지 않습니다."
    if re.fullmatch(r'\d+', clean_question): return True, "숫자로만 이루어진 질문은 처리하지 않습니다."
    if re.search(r'[a-zA-Z]', clean_question) and not re.search(r'[가-힣]', clean_question): return True, "영어 질문입니다. 한글로 답변해주세요."
    if not re.search(r'[가-힣0-9]', clean_question): return True, "의미를 알 수 없는 질문입니다."
    return False, "유효한 질문입니다."

def is_invalid_by_llm(model: Llama, question: str) -> Tuple[bool, str]:
    """LLM을 사용하여 의미론적 유효성을 검증합니다."""
    validation_prompt = f"""
당신은 사용자 입력이 '유효한 검색어'인지 '무의미한 입력'인지를 판단하는 시스템입니다. 사용자의 의도를 파악하는 것이 중요합니다.
[판단 기준]
### 핵심 규칙:
# '무의미'로 판단해야 하는 경우:
- 키보드를 무작위로 누른 것으로 보이는 입력 (예: "djskal;fjdkl;")
- 의미 없는 자모음의 반복 (예: "아아아아", "ㅋㅋㅋㅎ")
-
# '유효'로 판단해야 하는 경우:
- 위의 '무의미' 기준에 해당하지 않는, **의도를 파악할 수 있는 모든 문장과 질문**. (문법이 틀리거나 짧아도 괜찮습니다)
오직 '유효' 또는 '무의미' 두 단어 중 하나로만 답변해야 합니다. 다른 부가 설명은 절대 금지합니다.
입력 텍스트: {question}
판단:
"""
    try:
        res = model(validation_prompt, max_tokens=10, temperature=0.0)
        answer = res['choices'][0]['text'].strip()
        if "무의미" in answer:
            return True, "의미를 알 수 없는 질문입니다."
        else:
            return False, "유효한 질문입니다."
    except Exception as e:
        print(f"  [경고] LLM 유효성 검증 중 오류 발생: {e}")
        return False, "LLM 검증 중 오류"

# =====================================================================
# ★★★ 메인 답변 생성 및 실행 함수 ★★★
# =====================================================================

def answer_question(model: Llama, question: str, general_k: int = 10):
    """질문의 유형에 따라 적절한 핸들러를 호출하여 답변을 생성합니다."""
    invalid, reason = is_invalid_query(question)
    if invalid:
        print(f"  [입력 오류] {reason}")
        return {"type": "error", "message": reason}, reason
    
    invalid, reason = is_invalid_by_llm(model, question)
    if invalid:
        print(f"  [입력 오류] LLM이 의미 없는 질문으로 판단했습니다.")
        return {"type": "error", "message": reason}, reason
    
    # 질문에 '위험성평가' 키워드가 포함되어 있는지 여부로 로직을 분기합니다.
    if "위험성평가" in question:
        return handle_risk_assessment(model, question, SYSTEM_INSTRUCTION)
    else:
        return handle_general_question(model, question, general_k, SYSTEM_INSTRUCTION)

def main():
    """메인 실행 함수: 모델 로딩, 사용자 입력 루프, 답변 생성 및 출력을 담당합니다."""
    parser = argparse.ArgumentParser(description="RAG 기반 법령 질문 & 위험성평가")
    parser.add_argument("--model_path", default=r"C:\ai_work\RAG\models\A.X-4.0-Light-Q4_K_M.gguf", help="GGUF 모델 경로")
    parser.add_argument("--top_k", type=int, default=20, help="일반 질문 시 검색할 법령 청크 수")
    args = parser.parse_args()
    
    print("모델을 로딩 중입니다...")
    model = load_model(args.model_path)
    print("모델 로딩 완료.")
    
    while True:
        question = input("\n질문 (종료 입력 시 '종료' 또는 'exit'): ")
        if question.strip().lower() in ("종료", "exit"):
            print("프로그램을 종료합니다.")
            break
        
        if not question.strip():
            continue
        
        start_time = time.time()
        answer_json, raw_text = answer_question(model, question, general_k=args.top_k)
        end_time = time.time()
        
        if isinstance(raw_text, str):
            print(f"\n=== 생성된 답변 (원본) ===\n{raw_text}\n")
        
        if isinstance(answer_json, dict):
            print(f"\n=== 생성된 답변 (JSON) ===\n{json.dumps(answer_json, ensure_ascii=False, indent=4)}\n")
        else:
            print(f"\n=== 생성된 답변 ===\n{answer_json}\n")
        
        print(f"⏱️ 답변 생성 시간: {end_time - start_time:.2f}초\n")

if __name__ == "__main__":
    main()