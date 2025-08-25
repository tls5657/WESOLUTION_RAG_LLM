"""
위험성 평가 관련 함수들을 모아놓은 파일입니다.
"""

import re
import json
from llama_cpp import Llama
from typing import Tuple, List, Dict, Union
from retrieve import retrieve_chunks

# =====================================================================
# ★★★ 위험성평가 관련 LLM 프롬프트 및 호출 함수 그룹 ★★★
# =====================================================================

def make_initial_risk_prompt(question: str, legal_context: str, system_instruction: str) -> str:
    """초기 유해위험요인 및 감소대책 도출을 위한 프롬프트를 생성합니다."""
    return f"""
{system_instruction}
## 위험성 평가 1단계: 유해위험요인 및 감소대책 도출 (전문가적 분석)

당신은 안전보건 전문가로서, 주어진 상황에 대한 위험 요소를 분석하고 **현실적인 해결책**을 제시해야 합니다.
아래 [참고 법령]은 당신의 지식이 법적 테두리를 벗어나지 않도록 **단순히 참고만 하세요.**

[참고 법령]
{legal_context}

[분석 지침]
1.  **독창성**: 당신의 전문가적 지식과 경험을 바탕으로 '유해위험요인'과 '위험성 감소대책'을 **스스로 생성하세요.**
2.  **구체성**: '유해위험요인'은 구체적인 사고 시나리오로, '위험성 감소대책'은 현장에서 바로 적용할 수 있는 실질적인 조치로 작성하세요.
3.  **(매우 중요) 복사 금지**: [참고 법령]의 문장을 **절대로 그대로 복사하여 감소대책으로 사용하지 마세요.** 법령은 아이디어를 얻는 용도로만 활용하세요.

[질문]
{question}

[분석 결과]
'유해위험요인: [내용]'과 '감소대책: [내용]' 형식으로 여러 쌍을 나열해주세요.
"""

def parse_hazards_and_measures(model: Llama, initial_analysis: str) -> List[Dict[str, str]]:
    """LLM을 사용하여 1차 분석 결과 텍스트에서 '유해위험요인'과 '감소대책' 쌍을 JSON 형식으로 추출합니다."""
    extraction_prompt = f"""
당신은 제공된 텍스트에서 '유해위험요인'과 그에 해당하는 '감소대책'을 추출하여 JSON 리스트로 만드는 시스템입니다.

[추출 규칙]
- 각 항목은 '유해위험요인'과 '감소대책' 키를 가진 JSON 객체여야 합니다.
- 원본 텍스트에 있는 모든 쌍을 빠짐없이 추출해야 합니다.

[원본 텍스트]
{initial_analysis}

[출력 형식]
다른 어떤 설명도 없이, 반드시 아래와 같은 JSON 리스트(배열) 형식으로만 출력해야 합니다.
[
  {{
    "유해위험요인": "추출된 첫 번째 유해위험요인명",
    "감소대책": "첫 번째 유해위험요인에 대한 감소대책"
  }},
  {{
    "유해위험요인": "추출된 두 번째 유해위험요인명",
    "감소대책": "두 번째 유해위험요인에 대한 감소대책"
  }}
]
JSON:
"""
    try:
        res = model(extraction_prompt, max_tokens=1024, temperature=0.0, top_p=0.0)
        answer = res['choices'][0]['text'].strip()
        
        json_match = re.search(r'\[.*\]', answer, re.DOTALL)
        if not json_match:
            print("  [경고] LLM으로부터 유효한 JSON 리스트를 추출하지 못했습니다.")
            return []
        
        parsed_json = json.loads(json_match.group(0))
        if isinstance(parsed_json, list) and all(isinstance(item, dict) and '유해위험요인' in item and '감소대책' in item for item in parsed_json):
            return parsed_json
        else:
            print("  [경고] JSON의 형식이 올바르지 않습니다.")
            return []
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"  [경고] 유해위험요인/감소대책 파싱 실패: {e}")
        return []

def extract_process_and_tasks(model: Llama, question: str) -> Tuple[str, List[str]]:
    """LLM을 사용하여 질문에서 '작업공정'과 '세부작업' 목록을 JSON 형식으로 추출합니다."""
    extraction_prompt = f"""
사용자 질문에서 '작업공정'과 '세부작업'을 추출하여 JSON 형식으로 출력하세요.
- '작업공정'은 단일 문자열입니다.
- '세부작업'은 문자열의 리스트(배열)입니다.
[질문]
{question}
[출력 형식]
{{
  "작업공정": "추출된 작업공정명",
  "세부작업": ["추출된 세부작업 1", "추출된 세부작업 2"]
}}
JSON:
"""
    try:
        res = model(extraction_prompt, max_tokens=256, temperature=0.0, top_p=0.0)
        answer = res['choices'][0]['text'].strip()
        json_match = re.search(r'\{.*\}', answer, re.DOTALL)
        if not json_match:
            print(f"  [경고] LLM으로부터 유효한 JSON 형식을 추출하지 못했습니다.")
            return None, []
        
        parsed_json = json.loads(json_match.group(0))
        process = parsed_json.get("작업공정")
        tasks = parsed_json.get("세부작업", [])
        
        if not process:
            return None, []
        
        if not isinstance(tasks, list):
            tasks = [str(tasks)]
        
        return process, tasks
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"  [경고] 작업공정/세부작업 추출 실패: {e}")
        return None, []

def extract_and_calculate_risk_scores(model: Llama, initial_analysis: str) -> Union[List[dict], Tuple[bool, str]]:
    """1차 분석 결과를 바탕으로 각 유해위험요인별 빈도, 강도, 위험성 점수를 LLM을 통해 추출 및 계산합니다."""
    extraction_prompt = f"""
당신은 제공된 '초기 분석 결과'를 바탕으로 각 유해위험요인에 대한 위험성 평가 수치를 계산하는 시스템입니다.
[계산 규칙]
**빈도(Likelihood) 및 강도(Severity) 평가 기준 준수:**
    - **빈도 (1-3):**
        - 3점: 1일에 1회 정도
        - 2점: 1주일에 1회 정도
        - 1점: 3개월에 1회 정도
    - **강도 (1-3):**
        - 3점: 사망(장애 발생)
        - 2점: 휴업 필요
        - 1점: 비치료
5) **위험성 감소 대책 적용 후의 빈도와 강도를 각각 '개선 후 빈도'와 '개선 후 강도' 열에 평가하여 기입합니다.**
    - 개선 후 빈도/강도는 개선 전보다 반드시 낮거나 같아야 합니다.

6) **위험성 점수 산정:**
    - **개선 전 위험성 점수:** (개선 전) 빈도 x 강도
    - **개선 후 위험성 점수:** 위험성 감소 대책이 적용되었다고 가정하고, 아래 규칙에 따라 '개선 후 빈도'와 '개선 후 강도'를 설정한 후, 두 값을 곱하여 계산합니다.
1.  **빈도 (1-3):** 유해위험요인의 발생 가능성을 평가합니다. (3: 높음, 2: 중간, 1: 낮음)
2.  **강도 (1-3):** 사고 발생 시 심각성을 평가합니다. (3: 높음, 2: 중간, 1: 낮음)
개선 후 빈도(1-3): 개선전 '빈도' 점수보다 낮거나 같은 점수를 할당합니다.
개선 후 강도(1-3): 개선전 '강도' 점수보다 낮거나 같은 점수를 할당합니다.
최종 점수: '개선 후 빈도' x '개선 후 강도'로 계산합니다. 이때, 계산된 최종 점수는 '개선 전 위험성 점수'보다 반드시 낮아야 합니다.
**중요: 아래 JSON 형식으로만 응답하고, 어떤 추가 설명도 하지 마세요.** [초기 분석 결과]
{initial_analysis}
[출력 형식]
당신의 답변은 아래와 같은 파이썬 리스트의 리스트(배열의 배열) 형식으로만 출력해야 합니다.
출력되는 각 리스트의 순서는 반드시 [유해위험요인, 빈도, 강도, 개선 전 위험성 점수, 개선 후 위험성 점수] 순서여야 합니다.
반드시 다음 형식만 출력하세요. **첫 글자는 '[' 이어야 하며, 마지막 글자는 ']' 이어야 합니다. 그 뒤에는 어떠한 문자도(개행, 공백, 파이프 등) 출력하지 마세요.**

다른 어떤 설명도 없이, 아래 형식만 출력하세요. 각 항목은 '초기 분석 결과'에 언급된 유해위험요인에 해당합니다.
[
    [유해위험요인": "추출된 첫 번째 유해위험요인명, 
    "빈도": 계산된 빈도 점수 (1-3), 
    "강도": 계산된 강도 점수 (1-3),
    "개선 전 위험성 점수": 계산된 개선 전 점수,
    "개선 후 위험성 점수": 계산된 개선 후 점수],

    [유해위험요인": "추출된 첫 번째 유해위험요인명, 
    "빈도": 계산된 빈도 점수 (1-3), 
    "강도": 계산된 강도 점수 (1-3),
    "개선 전 위험성 점수": 계산된 개선 전 점수,
    "개선 후 위험성 점수": 계산된 개선 후 점수],
]
"""
    try:
        res = model(extraction_prompt, max_tokens=2048, temperature=0.0, top_p=0.0)
        raw_output = res['choices'][0]['text'].strip()
        print(raw_output)
        
        # JSON 전처리 로직
        cleaned_output = raw_output.strip().replace('|', '')
        open_count = cleaned_output.count('[')
        close_count = cleaned_output.count(']')
        if open_count != close_count:
            print("  [파싱 경고] 괄호 개수가 불일치하여 수정합니다.")
            if close_count > open_count:
                cleaned_output = cleaned_output.rstrip(']').rstrip() + ']'
                if cleaned_output.endswith(']]') and not cleaned_output.startswith('[['):
                    cleaned_output = cleaned_output[:-1]
        raw_output = cleaned_output
        
        # 1차 시도: 리스트의 리스트 형식
        start_index = raw_output.find('[')
        end_index = raw_output.rfind(']')
        if start_index != -1 and end_index != -1 and start_index < end_index:
            cleaned_str = raw_output[start_index : end_index + 1].strip()
            list_of_lists = None
            try:
                data = json.loads(cleaned_str)
                if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
                    list_of_lists = data
                else: raise ValueError()
            except (ValueError, json.JSONDecodeError):
                try:
                    list_of_lists = json.loads(f"[{cleaned_str}]")
                except (ValueError, json.JSONDecodeError): pass
            
            if list_of_lists:
                converted_data = []
                keys = ["유해위험요인", "빈도", "강도", "개선 전 위험성 점수", "개선 후 위험성 점수"]
                for row_list in list_of_lists:
                    if isinstance(row_list, list) and len(row_list) == len(keys):
                        converted_data.append(dict(zip(keys, row_list)))
                if converted_data:
                    print("  [파싱 성공] Fallback (리스트의 리스트) 로직으로 데이터를 변환했습니다.")
                    return converted_data

        # 2차 시도: Markdown 테이블
        print("  [파싱 경고] 리스트의 리스트 파싱 실패. 최종 Fallback(Markdown)을 시도합니다.")
        try:
            lines = raw_output.strip().split('\n')
            table_data, header_found = [], False
            keys = ["유해위험요인", "빈도", "강도", "개선 전 위험성 점수", "개선 후 위험성 점수"]
            
            for line in lines:
                if '---' in line: header_found = True; continue
                if not header_found or not line.strip().startswith('|'): continue
                
                parts = [p.strip() for p in line.strip('|').split('|')]
                if len(parts) == len(keys):
                    table_data.append(dict(zip(keys, parts)))
            
            if table_data:
                for item in table_data:
                    for key in ["빈도", "강도", "개선 전 위험성 점수", "개선 후 위험성 점수"]:
                        try: item[key] = int(item[key])
                        except: item[key] = 0
                print("  [파싱 성공] 최종 Fallback (Markdown) 로직으로 데이터를 변환했습니다.")
                return table_data
            raise ValueError("Not a valid Markdown table")
        except Exception as e:
            print(f"  [최종 파싱 오류] 모든 파싱 로직이 실패했습니다: {e}")
            print(f"  > LLM 원본 출력: {raw_output}")
            return True, "위험성평가를 다시 실행해주세요."

    except Exception as e:
        print(f"  [경고] 위험성 점수 추출 중 알 수 없는 오류 발생: {e}")
        return []

def parse_risk_assessment_table_to_json(table_text: str) -> dict:
    """최종 생성된 마크다운 테이블 텍스트를 JSON 형식으로 변환합니다."""
    cleaned_text = re.sub(r'\|\s*-{3,}\s*(?:\|\s*-{3,}\s*)*\|', '', table_text, flags=re.MULTILINE)
    lines = cleaned_text.strip().split('\n')
    
    if len(lines) < 2:
        return {"type": "risk_assessment", "headers": {}, "data": []}
        
    headers_ko = [h.strip() for h in lines[0].split('|') if h.strip()]
    key_map = {
        '작업공정명': 'process_name', '세부작업명': 'detailed_work', '유해위험요인': 'hazard_risk_factors',
        '빈도(1–3)': 'frequency', '강도(1–3)': 'severity', '개선 전 위험성 점수': 'risk_score_before',
        '위험성 감소대책': 'risk_reduction_measures', '개선 후 위험성 점수': 'risk_score_after',
        '관련 법령': 'related_regulations'
    }
    data_rows = []
    for line in lines[1:]:
        if not line.strip(): continue
        
        values = [v.strip() for v in line.strip('|').split('|')]
        if len(values) != len(headers_ko): continue

        row_dict = {}
        for i, header_ko in enumerate(headers_ko):
            json_key = key_map.get(header_ko)
            if not json_key: continue
            
            value = values[i] if i < len(values) else ""
            if json_key in ['frequency', 'severity', 'risk_score_before', 'risk_score_after']:
                try:
                    row_dict[json_key] = int(value)
                except (ValueError, TypeError):
                    row_dict[json_key] = 0
            else:
                row_dict[json_key] = value
        data_rows.append(row_dict)
    
    headers_for_json = {v: k for k, v in key_map.items()}
    return {"type": "risk_assessment", "headers": headers_for_json, "data": data_rows}

def handle_risk_assessment(model: Llama, question: str, system_instruction: str):
    """위험성 평가 전체 프로세스를 처리하고 최종 결과를 반환합니다."""
    # [Step 1] 핵심 키워드(작업공정, 세부작업) 추출
    print("\n[1단계] 작업공정 및 세부작업 추출 중...")
    process_name, task_list = extract_process_and_tasks(model, question)

    if not process_name or not task_list:
        reason = "질문에서 '작업공정'과 '세부작업'을 명확히 추출하지 못했습니다. 형식을 확인해주세요."
        print(f"  [입력 오류] {reason}")
        return {"type": "error", "message": reason}, reason
    print(f"  [추출 결과] 작업공정: {process_name}, 세부작업: {task_list}")

    # [Step 2] 1차 법령 검색 (LLM에게 Context 제공 목적)
    print("\n[2단계] 컨텍스트용 법령 검색 중...")
    context_query = f"{process_name} {', '.join(task_list)} 안전보건기준"
    hits = retrieve_chunks(context_query, 20)
    legal_context = "\n\n".join(f"[조문{i+1}] {txt}" for i, (txt, _) in enumerate(hits[:10]))
    print(f"  [검색 완료] {len(hits[:10])}개의 참고 법령을 LLM에 전달합니다.")

    # [Step 3] 법령 기반 유해위험요인 및 감소대책 도출
    print("\n[3단계] 법령 기반 위험요인/대책 도출 중...")
    initial_prompt = make_initial_risk_prompt(question, legal_context, system_instruction)
    initial_res = model(initial_prompt, max_tokens=1024, temperature=0.7, top_p=0.95)
    initial_analysis = initial_res['choices'][0]['text'].strip()

    # [Step 3.5] LLM 응답에서 (위험요인, 감소대책) 쌍 파싱
    hazard_pairs = parse_hazards_and_measures(model, initial_analysis)
    if not hazard_pairs:
        reason = "LLM의 답변에서 유해위험요인과 감소대책을 추출하지 못했습니다."
        print(f"  [파싱 오류] {reason}")
        return {"type": "error", "message": reason, "raw_output": initial_analysis}, initial_analysis
    print(f"  [파싱 완료] {len(hazard_pairs)}개의 (위험요인, 대책) 쌍을 추출했습니다.")

    # [Step 4] 2차 법령 검색 (각 쌍에 대한 정밀 매칭)
    print("\n[4단계] 각 항목별 정밀 법령 매칭 중...")
    final_data = []
    for pair in hazard_pairs:
        hazard = pair.get("유해위험요인")
        measure = pair.get("감소대책")
        
        precise_query = f"{hazard} {measure} 기준"
        law_hits = retrieve_chunks(precise_query, top_k=1)
        
        law_ref = "관련 법령 미확인"
        if law_hits:
            law_text = law_hits[0][0]
            law_match = re.search(r'제\s*(\d+)\s*조(?:의\s*\d+)?', law_text)
            base_ref = law_match.group(0) if law_match else ""
            
            if '[산업안전보건기준에 관한 규칙]' in law_text: law_ref = f"산업안전보건기준에 관한 규칙 {base_ref}".strip()
            elif '[산업안전보건법]' in law_text: law_ref = f"산업안전보건법 {base_ref}".strip()
            elif '[중대재해처벌법]' in law_text: law_ref = f"중대재해처벌법 {base_ref}".strip()
            elif '[항만법]' in law_text: law_ref = f"항만법 {base_ref}".strip()
            elif '[해운법]' in law_text: law_ref = f"해운법 {base_ref}".strip()
            elif '[산업안전보건법 시행령]' in law_text: law_ref = f"산업안전보건법 시행령 {base_ref}".strip() 
            elif base_ref: law_ref = base_ref
        
        pair['related_law'] = law_ref
        final_data.append(pair)
    print("  [매칭 완료]")
    
    # [Step 5] 위험성 점수 계산
    print("\n[5단계] 위험성 점수 계산 중...")
    risk_scores_result = extract_and_calculate_risk_scores(model, initial_analysis)
    
    if isinstance(risk_scores_result, tuple):
        is_failure, reason = risk_scores_result
        if is_failure:
            print(f"  [입력 오류] {reason}")
            return {"type": "error", "message": reason}, reason
    
    risk_scores_data = risk_scores_result
    scores_map = {item.get('유해위험요인'): item for item in risk_scores_data}
    
    # [Step 5.5] 데이터 병합
    for item in final_data:
        hazard_key = item.get("유해위험요인")
        scores = scores_map.get(hazard_key)
        if scores:
            item.update(scores)
    print("  [계산 및 병합 완료]")

    # [Step 6] 최종 표 조립
    print("\n[6단계] 최종 평가표 조립 중...")
    header = "| 작업공정명 | 세부작업명 | 유해위험요인 | 빈도(1–3) | 강도(1–3) | 개선 전 위험성 점수 | 위험성 감소대책 | 개선 후 위험성 점수 | 관련 법령 |"
    separator = "|---|---|---|---|---|---|---|---|---|"
    table_rows = [header, separator]

    for item in final_data:
        task_str = ', '.join(task_list)
        row = [
            process_name, task_str, item.get("유해위험요인", ""),
            str(item.get("빈도", 0)), str(item.get("강도", 0)),
            str(item.get("개선 전 위험성 점수", 0)), item.get("감소대책", ""),
            str(item.get("개선 후 위험성 점수", 0)), item.get("related_law", "관련 법령 미확인")
        ]
        table_rows.append(f"| {' | '.join(row)} |")
        
    final_table_text = "\n".join(table_rows)
    json_output = parse_risk_assessment_table_to_json(final_table_text)
    print("  [조립 완료]")
    
    return json_output, final_table_text