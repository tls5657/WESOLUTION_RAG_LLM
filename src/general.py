"""
위험성 평가가 아닌 일반 법령 질문 관련 함수들을 모아놓은 파일입니다.
"""
import re
from llama_cpp import Llama
from retrieve import retrieve_chunks

# =====================================================================
# ★★★ 일반 법령 질문 관련 LLM 프롬프트 및 호출 함수 그룹 ★★★
# =====================================================================

def classify_question_type(model: Llama, question: str) -> str:
    """질문의 유형을 'law_text' 또는 'law_interpretation'으로 분류합니다."""
    classification_prompt = f"""
당신은 법령 관련 질문을 두 가지 중 하나로 분류합니다.
[분류 기준]
- law_text: 질문자가 법령 조문이나 법률 원문 자체를 직접 보고 싶어하는 경우
  (법령을 확인하려는 의도인 경우)
- law_interpretation: 질문자가 법령을 근거로 한 설명, 절차, 방법, 요건, 적용 사례 등 **해석 또는 활용 방법**을 알고 싶어하는 경우
  (법령 원문을 보는 것만으로는 답이 충분하지 않고, 실행·적용을 위한 설명이 필요한 경우)
[판단 방법]
- 질문이 '법령을 확인하는 것'이 목적이라면 law_text
- 질문이 '법령이 아닌 다른 것'을 묻는 것이라면 law_interpretation
- 특정 단어나 표현 여부가 아니라, 질문의 전체 의미와 의도를 이해하여 판정
질문: {question}
출력: law_text 또는 law_interpretation
"""
    res = model(classification_prompt, max_tokens=10, temperature=0.2)
    answer = res['choices'][0]['text'].strip().lower()
    return 'law_text' if 'law_text' in answer else 'law_interpretation'

def make_general_prompt(model: Llama, hits: list, question: str, q_type: str, system_instruction: str) -> str:
    """분류된 질문 유형에 따라 적절한 LLM 프롬프트를 생성합니다."""
    context = "\n\n".join(f"[조문{i+1}] {txt}" for i, txt in enumerate(hits))
    if q_type == 'law_text':
        return f"""
{system_instruction}
아래 "법령 조문 목록"을 참고하여, "질문"에 대한 답변을 아래 "매우 중요한 지시사항"에 따라 완벽하게 생성하세요.
[법령 조문 목록]
{context}
[질문]
{question}
---
[매우 중요한 지시사항]
1. **선별 작업:** 가장 높은 점수로 검색된 법령을 최우선으로 답변에 포함하세요.
2. 질문이 법령 하나를 묻는건지 여러 개의 법령을 묻는건지 판단하세요.
2. **법령 나열:** `- 법령:` 이라는 제목 아래에, 법령들의 **원문 전체**를 이어서 나열하세요.  제공된 [법령 조문 목록] 외의 내용은 절대로 언급하지 말고 지어내거나 각색하지 마세요. 각 조항 사이에는 빈 줄을 넣어 구분하세요.
3. **종합 설명:** 모든 법령을 나열한 후, `- 설명:` 이라는 제목 아래에, 당신이 선별한 법령들의 **핵심 내용을 종합하여 하나의 문단으로 요약**하세요. **개별 법령에 대한 설명을 절대 반복하지 마세요.**
4. **형식 준수:** 다른 서론이나 결론 없이, 반드시 `- 법령:`과 `- 설명:` 두 가지 항목으로만 답변을 구성하세요. 절대로 '|' 문자나 표(테이블) 형식을 사용하지 마세요.
5. 존댓말로 답변해주세요.
---
답변:
"""
    else: # law_interpretation
        return f"""
{system_instruction}
아래 "법령 조문"을 참고하여, "질문"에 대해 "생성할 답변 형식"을 완벽하게 지켜서 답변하세요.
[법령 조문]
{context}
[질문]
{question}
---
[매우 중요한 지시사항]
1. 반드시 "- 답변:"과 "- 관련 법령:" 두 항목으로만 구성하여 답변을 시작하세요.
2. **'|' 문자나 표(테이블) 형식은 절대로, 절대로 사용하지 마세요.**
3. `- 답변:` 다음 줄부터 질문에 대한 최대한 상세한 설명을 작성하세요. **법령 조문에 없는 법령은 작성하지 마세요.**
4. `- 관련 법령:` 다음 줄에 답변의 근거가 된 법령 이름과 조항을 적으세요.
5. 다른 모든 말은 생략하고, 아래 "생성할 답변 형식"과 똑같이 답변을 시작하세요.
6. 존댓말로 답변해주세요.
- 답변:
[질문에 대한 답변을 작성하세요]
- 관련 법령:
[답변 내용과 관련된 법령명과 조문 번호를 여기에 작성하세요]
"""

def advanced_parse_llm_output(raw_text: str, q_type: str) -> dict:
    """LLM의 일반 법령 질문 답변 텍스트를 체계적인 JSON으로 파싱합니다."""
    def _process_laws_content(text: str) -> list:
        law_title_pattern = r'(\[.*?\]\s*제\s*\d+\s*조(?:의\s*\d+)?(?:\(.*?\))?)'
        matches = list(re.finditer(law_title_pattern, text, re.DOTALL))
        laws_list = []
        
        if not matches:
            if text:
                split_pos = text.find(')') + 1
                if split_pos > 0 and split_pos < len(text):
                    law_title = text[:split_pos].strip()
                    law_content = text[split_pos:].strip()
                    laws_list.append({"law_title": law_title, "law_content": law_content})
                else:
                    parts = text.split('\n', 1)
                    law_title = parts[0].strip()
                    law_content = parts[1].strip() if len(parts) > 1 else ""
                    if law_title:
                        laws_list.append({"law_title": law_title, "law_content": law_content})
            return laws_list

        for i, match in enumerate(matches):
            start_index = match.start()
            end_index = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            law_block = text[start_index:end_index].strip()
            split_pos = law_block.find(')') + 1
            law_title, law_content = "", ""
            
            if split_pos > 0 and split_pos < len(law_block):
                law_title = law_block[:split_pos].strip()
                law_content = law_block[split_pos:].strip()
            else:
                parts = law_block.split('\n', 1)
                law_title = parts[0].strip()
                law_content = parts[1].strip() if len(parts) > 1 else ""
            
            if law_title:
                laws_list.append({"law_title": law_title, "law_content": law_content})
        return laws_list
        
    try:
        markers = ["- 답변:", "- 법령:", "- 관련 법령:", "- 설명:"]
        marker_to_key = {"- 답변:": "answer", "- 법령:": "laws", "- 관련 법령:": "related_laws", "- 설명:": "description"}
        pattern = '|'.join(re.escape(m) for m in markers)
        parts = re.split(f'({pattern})', raw_text)
        ordered_data = []
        current_marker, temp_notes = None, []

        for part in parts:
            part = part.strip()
            if not part: continue
            
            if part in markers:
                if temp_notes:
                    ordered_data.append(('additional_notes', "\n".join(temp_notes)))
                    temp_notes = []
                current_marker = part
            elif current_marker:
                json_key = marker_to_key[current_marker]
                main_content, leftover_text = part, None
                
                if json_key in ['related_laws', 'description']:
                    content_parts = part.split('\n\n', 1)
                    main_content = content_parts[0].strip()
                    if len(content_parts) > 1:
                        leftover_text = content_parts[1].strip()
                
                if json_key == 'laws':
                    processed_content = _process_laws_content(part)
                    ordered_data.append((json_key, processed_content))
                else:
                    final_content = part if json_key == 'answer' else main_content
                    ordered_data.append((json_key, final_content))
                
                if leftover_text:
                    temp_notes.append(leftover_text)
                
                current_marker = None
            else:
                temp_notes.append(part)
        
        if temp_notes:
            ordered_data.append(('additional_notes', "\n".join(temp_notes)))

        final_data_dict = dict(ordered_data)
        return {"type": q_type, "data": final_data_dict}
    except Exception as e:
        print(f"  [오류] 고급 파싱 실패: {e}")
        return {"type": "error", "message": "LLM 답변을 파싱하는 데 실패했습니다.", "raw_output": raw_text}

def handle_general_question(model: Llama, question: str, general_k: int, system_instruction: str):
    """일반 법령 질문 전체 프로세스를 처리하고 최종 결과를 반환합니다."""
    print("\n답변 생성 중...")
    hits = retrieve_chunks(question, general_k)
    if not hits:
        reason = "관련 법령 조항을 찾을 수 없습니다."
        print("  [검색 결과 없음]")
        return { "type": "no_result", "message": reason }, reason
    
    limit = general_k
    match = re.search(r"(\d+)\s*(?:가지|개|가지만|개만)", question)
    if match:
        try:
            limit = int(match.group(1))
            print(f"  [요청 개수 감지] → {limit}개로 제한")
        except (ValueError, IndexError): pass
    
    q_type = classify_question_type(model, question)
    print(f"  [질문 분류 결과] → {q_type}")
    print("  [처리 방식] → LLM 답변 생성 후 JSON으로 변환")
    prompt_hits = [item for item, score in hits]
    prompt = make_general_prompt(model, prompt_hits[:limit], question, q_type, system_instruction)
    res = model(prompt, max_tokens=16000, temperature=0.0, top_p=1.0)
    final_answer_text = res['choices'][0]['text'].strip()
    
    json_output = advanced_parse_llm_output(final_answer_text, q_type)
    return json_output, final_answer_text