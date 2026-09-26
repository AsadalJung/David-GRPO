import re
import string


def validate_search_tags(solution_str):
    """검색 태그들이 올바르게 쌍으로 매칭되는지 확인합니다.
    
    Args:
        solution_str: 모델의 응답 텍스트
        
    Returns:
        bool: 검색 태그가 올바르게 매칭되면 True, 아니면 False
    """
    # <begin_search>와 </end_search> 쌍 확인
    begin_search_count = len(re.findall(r'<begin_search>', solution_str))
    end_search_count = len(re.findall(r'</end_search>', solution_str))
    
    # <search_result>와 </search_result> 쌍 확인
    search_result_open_count = len(re.findall(r'<search_result>', solution_str))
    search_result_close_count = len(re.findall(r'</search_result>', solution_str))
    
    # 검색 쿼리 중복 확인
    search_queries = re.findall(r'<begin_search>(.*?)</end_search>', solution_str, re.DOTALL)
    unique_queries = list(set(search_queries))
    
    # 쌍이 맞지 않으면 False
    if begin_search_count != end_search_count:
        return False
    
    if search_result_open_count != search_result_close_count:
        return False
    
    # 검색 시도 횟수와 검색 결과 횟수가 일치해야 함
    if begin_search_count != search_result_open_count:
        return False
    
    # 검색 쿼리가 중복되면 False
    if len(search_queries) != len(unique_queries):
        return False
    
    return True


def extract_titles_from_search_results(solution_str):
    """검색 결과에서 타이틀들을 추출합니다.
    
    Args:
        solution_str: 모델의 응답 텍스트
        
    Returns:
        list: 추출된 타이틀들의 리스트
    """
    titles = []
    
    # <search_result> 태그들을 찾습니다
    search_result_pattern = r'<search_result>(.*?)</search_result>'
    search_results = re.findall(search_result_pattern, solution_str, re.DOTALL)
    
    for result in search_results:
        # result 1: "Title" 형태를 찾습니다
        title_pattern = r'result\s+\d+:\s*"([^"]+)"'
        title_matches = re.findall(title_pattern, result)
        titles.extend(title_matches)
    
    return titles


def normalize_title(title):
    """타이틀을 정규화합니다.
    
    Args:
        title: 원본 타이틀
        
    Returns:
        str: 정규화된 타이틀
    """
    if not title:
        return ""
    
    # HTML 엔티티 디코딩
    html_entities = {
        '&amp;': '&',
        '&lt;': '<',
        '&gt;': '>',
        '&quot;': '"',
        '&#39;': "'",
        '&apos;': "'",
        '&nbsp;': ' ',
        '&copy;': '©',
        '&reg;': '®',
        '&trade;': '™'
    }
    
    for entity, char in html_entities.items():
        title = title.replace(entity, char)
    
    # 기본 정규화 (소문자, 공백 정리)
    normalized = title.lower().strip()
    
    # 특수 문자 제거 (하지만 공백은 유지)
    normalized = re.sub(r'[^\w\s]', '', normalized)
    
    # 연속된 공백을 하나로
    normalized = re.sub(r'\s+', ' ', normalized)
    
    return normalized.strip()


def compute_partial_reward(solution_str, supporting_facts, partial_reward_score=0.5):
    """검색 결과의 타이틀과 supporting_facts의 타이틀을 비교하여 부분 보상을 계산합니다.
    
    Args:
        solution_str: 모델의 응답 텍스트
        supporting_facts: supporting_facts 딕셔너리
        partial_reward_score: 부분 보상 점수 (기본값: 0.5)
        
    Returns:
        float: 부분 보상 점수
    """
    if not supporting_facts or 'title' not in supporting_facts:
        return 0.0
    
    # supporting_facts의 타이틀들을 정규화하고 중복 제거
    supporting_titles = supporting_facts['title']
    if isinstance(supporting_titles, str):
        supporting_titles = [supporting_titles]
    
    normalized_supporting_titles = [normalize_title(title) for title in supporting_titles]
    # 중복 제거
    normalized_supporting_titles = list(set(normalized_supporting_titles))
    
    # 검색 결과에서 타이틀 추출하고 중복 제거
    extracted_titles = extract_titles_from_search_results(solution_str)
    normalized_extracted_titles = [normalize_title(title) for title in extracted_titles]
    # 중복 제거
    normalized_extracted_titles = list(set(normalized_extracted_titles))
    
    # 매칭되는 타이틀 수 계산
    matched_count = 0
    for extracted_title in normalized_extracted_titles:
        for supporting_title in normalized_supporting_titles:
            if extracted_title == supporting_title:
                matched_count += 1
                break
    
    # 부분 보상 계산
    if len(normalized_supporting_titles) > 0:
        partial_reward = (matched_count / len(normalized_supporting_titles)) * partial_reward_score
        return partial_reward
    
    return 0.0


def normalize_answer(s):
    """Enhanced version of answer normalization processing"""

    if s is None:
        return ""
    
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    def remove_special_tokens(text):
        # Remove various quotes and special markings
        return re.sub(r'[""\'\'「」『』\(\)\[\]\{\}]', '', text)
    
    # Add functionality to remove common filler words
    def remove_fillers(text):
        fillers = ['well', 'so', 'basically', 'actually', 'literally', 'simply', 'just', 'um', 'uh']
        pattern = r'\b(' + '|'.join(fillers) + r')\b'
        return re.sub(pattern, ' ', text)
    
    # Normalize number representations
    def normalize_numbers(text):
        text = re.sub(r'\bzero\b', '0', text)
        text = re.sub(r'\bone\b', '1', text)
        text = re.sub(r'\btwo\b', '2', text)
        text = re.sub(r'\bthree\b', '3', text)
        text = re.sub(r'\bfour\b', '4', text)
        text = re.sub(r'\bfive\b', '5', text)
        text = re.sub(r'\bsix\b', '6', text)
        text = re.sub(r'\bseven\b', '7', text)
        text = re.sub(r'\beight\b', '8', text)
        text = re.sub(r'\bnine\b', '9', text)
        return text
    
    # 执行所有归一化步骤
    return white_space_fix(remove_fillers(normalize_numbers(
        remove_special_tokens(remove_articles(remove_punc(lower(s)))))))

def extract_solution(solution_str, method='comprehensive'):
    """Enhanced version for answer extraction"""
    assert method in ['strict', 'flexible', 'comprehensive']
    
    solution_str = solution_str.strip()
    final_answer = None
    
    # First handle the </think> tag, removing the thinking process before it
    solution_str = solution_str.split('</think>')[-1].strip()
    
    if method == 'strict':
        # Strict mode only accepts answers in \boxed{} format
        boxes = re.findall(r"\\boxed{([^}]*)}", solution_str)
        if boxes:
            final_answer = normalize_answer(boxes[-1].strip())
    
    elif method == 'flexible':
        # Flexible mode first tries common markers, then other patterns
        boxes = re.findall(r"\\boxed{([^}]*)}", solution_str)
        if boxes:
            final_answer = normalize_answer(boxes[-1].strip())
        else:
            # 查找常见答案前缀
            answer_pattern = re.search(r"(The answer is|Therefore,|Thus,|So,|In conclusion,|Hence,)[:\s]+([^\.]+)", solution_str, re.IGNORECASE)
            if answer_pattern:
                final_answer = normalize_answer(answer_pattern.group(2).strip())
            elif solution_str:
                sentences = solution_str.split('.')
                if sentences:
                    final_answer = normalize_answer(sentences[-2].strip() if len(sentences) > 1 else sentences[-1].strip())
    
    elif method == 'comprehensive':
        # Comprehensive mode tries multiple extraction strategies and selects the most likely answer
        candidates = []
        
        # 1. Check for \boxed{} format
        boxes = re.findall(r"\\boxed{([^}]*)}", solution_str)
        if boxes:
            candidates.append(normalize_answer(boxes[-1].strip()))
        
        # 2. Check for direct answer declarations
        patterns = [
            r"(The answer is|Therefore|Thus|So|In conclusion|Hence)[:\s]+([^\.]+)",
            r"(I believe the answer is|The final answer is|The correct answer is)[:\s]+([^\.]+)",
            r"(Answer)[:\s]+([^\.]+)"
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, solution_str, re.IGNORECASE)
            for match in matches:
                candidates.append(normalize_answer(match.group(2).strip()))
        
        # 3. Check the final sentence as an answer
        if solution_str:
            sentences = [s.strip() for s in solution_str.split('.') if s.strip()]
            if sentences:
                # 添加最后一句和倒数第二句作为候选
                if len(sentences) > 0:
                    candidates.append(normalize_answer(sentences[-1]))
                if len(sentences) > 1:
                    candidates.append(normalize_answer(sentences[-2]))
        
        # Choose the most likely candidate from the options (simply selecting the first non-empty answer here)
        for candidate in candidates:
            if candidate:
                final_answer = candidate
                break
    
    return final_answer


def default_compute_score(solution_str, ground_truth, method='strict', format_score=0.50, score=1.0, extra_info=None):
    """增强版评分函数
    
    Args:
        solution_str: 模型的解答文本
        ground_truth: 标准答案
        method: 答案提取方法，可选 'strict'、'flexible' 或 'comprehensive'
        format_score: 格式正确但答案不完全匹配时的分数
        score: 完全匹配的满分
        extra_info: 추가 정보 (batch 정보 포함)
    """
    # Preprocessing check
    if not solution_str or not ground_truth:
        return 0.0
    
    # Format check (if checking for </think> format is needed)
    format_correct = '</think>' in solution_str 
    if not format_correct:
        return 0.0
    
    # </think> 이후에 검색 관련 태그들이 있으면 0점
    after_think = solution_str.split('</think>')[-1]
    search_tags = ['<begin_search>', '</end_search>', '<search_result>', '</search_result>']
    if any(tag in after_think for tag in search_tags):
        return 0.0  # </think> 이후에 검색 관련 태그가 있으면 0점
    
   # <think> 이후에서 검색시도 안했을 경우 0점 
    response_part = solution_str.split('<think>')[-1].strip()

    if '<begin_search>' not in response_part:
        return 0.0  # <think> 이후에 검색 시도가 없으면 0점
    
    # Partial reward 계산
    partial_reward = 0.0
    give_partial_reward = False
    
    if extra_info and 'supporting_facts' in extra_info:
        trainer_config = extra_info.get('trainer_config', {})
        give_partial_reward = trainer_config.get('give_partial_reward', False)
        
        if give_partial_reward:
            # 검색 태그 검증 (기본 포맷 검증) - partial reward 사용 시에만
            if not validate_search_tags(response_part):
                return 0.0
            
            supporting_facts = extra_info.get('supporting_facts', {})
            partial_reward = compute_partial_reward(response_part, supporting_facts, partial_reward_score=0.5)
    
    # Extract and normalize answers
    answer = extract_solution(solution_str=solution_str, method=method)
    normalized_ground_truth = normalize_answer(ground_truth)
    
    # give_partial_reward=True일 때만 새로운 로직 적용
    if give_partial_reward:
        if answer is None:
            return 0.0

        # 답안 정확도 점수 계산 (0.5점 만점)
        answer_score = 0.0

        if answer == normalized_ground_truth:
            answer_score = 0.5  # EM: 0.5점
        elif normalized_ground_truth in answer or answer in normalized_ground_truth:
            answer_score = 0.25  # 부분 매칭: 0.25점
        else:
            gt_words = set(normalized_ground_truth.split())
            answer_words = set(answer.split())
            common_words = gt_words.intersection(answer_words)

            if len(common_words) >= len(gt_words) * 0.7:
                answer_score = 0.5  # 키워드 매칭: 0.5점
            else:
                answer_score = 0.0  # 매칭 없음: 0점

        # 검색 점수와 답안 점수를 독립적으로 합산
        # 최종 점수 = 검색 결과 매칭 점수(0~0.5점) + 답안 정확도 점수(0~0.5점)
        final_score = partial_reward + answer_score

        return final_score
    
    # give_partial_reward=False일 때는 기존 로직 사용
    else:
        if answer is None:
            return 0.0
        
        if answer == normalized_ground_truth:
            return score

        if normalized_ground_truth in answer or answer in normalized_ground_truth:
            return format_score

        gt_words = set(normalized_ground_truth.split())
        answer_words = set(answer.split())
        common_words = gt_words.intersection(answer_words)

        if len(common_words) >= len(gt_words) * 0.7:
            return score

        return 0.0


def compute_score(solution_str, ground_truth, method='strict', format_score=0.50, score=1.0, extra_info=None):
    # Multiple answers are separated by "<ans_split>"
    answers = ground_truth.split("<ans_split>")
    scores = []
    for ans in answers:
        s = default_compute_score(solution_str, ans, method=method, format_score=format_score, score=score, extra_info=extra_info)
        scores.append(s)
    
    return max(scores) if scores else 0.0
