import re
import string
from pathlib import Path

import numpy as np
import pandas as pd
from typing import List, Optional, Union, Dict, Any


ANSWER_TAG_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
DATA_SOURCE_TO_SOURCE_FILE = {
    "2wikimultihopqa": "2WIKI_dev.json",
    "antileakbench_2024": "AntiLeakBench_2024_multihop.json",
    "antileakbench_2024_multihop": "AntiLeakBench_2024_multihop.json",
    "bamboogle": "Bamboogle_dev.json",
    "bamtwoogle": "BamTwoogle_dev.json",
    "hotpotqa": "HotpotQA_dev.json",
    "musique": "MuSiQue_dev.json",
    "nq": "NQ_dev.json",
    "triviaqa": "TriviaQA_dev.json",
}
REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_METADATA_PATH = REPO_ROOT / "data/processed/all_test_data/test_from_tree.parquet"

def normalize_answer(text: Optional[str]) -> str:
    """Enhanced answer normalization process.
    
    Args:
        text: Text to be normalized, can be None
        
    Returns:
        str: Normalized text
    """
    if not text:
        return ""

    # Match Tree-GRPO EM normalization: lowercase -> strip punctuation -> drop articles -> fix spaces
    return white_space_fix(remove_articles(remove_punc(lower(text))))

def remove_articles(text: str) -> str:
    """Remove articles from text.
    
    Args:
        text: Input text
        
    Returns:
        str: Text with articles removed
    """
    return re.sub(r'\b(a|an|the)\b', ' ', text)

def white_space_fix(text: str) -> str:
    """Fix whitespace issues in text.
    
    Args:
        text: Input text
        
    Returns:
        str: Text with fixed whitespace
    """
    return ' '.join(text.split())

def remove_punc(text: str) -> str:
    """Remove punctuation from text.
    
    Args:
        text: Input text
        
    Returns:
        str: Text with punctuation removed
    """
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)

def lower(text: str) -> str:
    """Convert text to lowercase.
    
    Args:
        text: Input text
        
    Returns:
        str: Lowercase text
    """
    return text.lower()

def remove_special_tokens(text: str) -> str:
    """Remove special tokens and quotation marks from text.
    
    Args:
        text: Input text
        
    Returns:
        str: Text with special tokens removed
    """
    return re.sub(r'[""\'\'「」『』\(\)\[\]\{\}]', '', text)

def remove_fillers(text: str) -> str:
    """Remove filler words from text.
    
    Args:
        text: Input text
        
    Returns:
        str: Text with filler words removed
    """
    fillers = ['well', 'so', 'basically', 'actually', 'literally', 'simply', 'just', 'um', 'uh']
    pattern = r'\b(' + '|'.join(fillers) + r')\b'
    return re.sub(pattern, ' ', text)

def normalize_numbers(text: str) -> str:
    """Convert number words to numeric characters.
    
    Args:
        text: Input text
        
    Returns:
        str: Text with normalized numbers
    """
    number_mapping = {
        r'\bzero\b': '0', r'\bone\b': '1', r'\btwo\b': '2',
        r'\bthree\b': '3', r'\bfour\b': '4', r'\bfive\b': '5',
        r'\bsix\b': '6', r'\bseven\b': '7', r'\beight\b': '8',
        r'\bnine\b': '9'
    }
    
    for word_pattern, digit in number_mapping.items():
        text = re.sub(word_pattern, digit, text)
    return text

def extract_solution(solution_str: str, method: str = 'comprehensive') -> Optional[str]:
    """Extract the final answer from solution text.
    
    Args:
        solution_str: Text containing the solution
        method: Extraction method, options: 'strict', 'flexible', 'comprehensive'
    
    Returns:
        Optional[str]: Extracted answer, returns None if extraction fails
    """
    assert method in ['strict', 'flexible', 'comprehensive'], "Method must be 'strict', 'flexible', or 'comprehensive'"
    
    solution_str = solution_str.strip()
    final_answer = None
    
    # Remove thinking process
    solution_str = solution_str.split('</think>')[-1].strip()
    
    tagged_answer = None
    tag_matches = ANSWER_TAG_PATTERN.findall(solution_str)
    if tag_matches:
        tagged_answer = normalize_answer(tag_matches[-1].strip())

    if method == 'strict':
        # Strict mode accepts tagged or boxed answers only
        if tagged_answer:
            final_answer = tagged_answer
        else:
            boxes = re.findall(r"\\boxed{([^}]*)}", solution_str)
            if boxes:
                final_answer = normalize_answer(boxes[-1].strip())
    
    elif method == 'flexible':
        # Flexible mode tries common markers first, then other patterns
        if tagged_answer:
            final_answer = tagged_answer
        else:
            boxes = re.findall(r"\\boxed{([^}]*)}", solution_str)
            if boxes:
                final_answer = normalize_answer(boxes[-1].strip())
        if final_answer is None:
            # Look for common answer prefixes
            answer_pattern = re.search(r"(The answer is|Therefore,|Thus,|So,|In conclusion,|Hence,)[:\s]+([^\.]+)", 
                                      solution_str, re.IGNORECASE)
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
        if tagged_answer:
            candidates.append(tagged_answer)

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
        
        # 3. Check last sentences as answers
        if solution_str:
            sentences = [s.strip() for s in solution_str.split('.') if s.strip()]
            if sentences:
                # Add last and second-to-last sentences as candidates
                if len(sentences) > 0:
                    candidates.append(normalize_answer(sentences[-1]))
                if len(sentences) > 1:
                    candidates.append(normalize_answer(sentences[-2]))
        
        # Select the most likely answer from candidates
        for candidate in candidates:
            if candidate:
                final_answer = candidate
                break
    
    return final_answer

def compute_score(solution_str: str, ground_truth: str, 
                  method: str = 'strict', format_score: float = 0.50, 
                  score: float = 1.0) -> float:
    """Evaluate the score of a solution.
    
    Args:
        solution_str: Model's solution text
        ground_truth: Standard answer
        method: Answer extraction method, options: 'strict', 'flexible', 'comprehensive'
        format_score: Score when format is correct but answer doesn't fully match
        score: Full score for complete match
        
    Returns:
        float: Evaluation score
    """
    # Preprocessing checks
    if not solution_str or not ground_truth:
        return 0.0

    # Extract and normalize answers
    answer = extract_solution(solution_str=solution_str, method=method)
    normalized_ground_truth = normalize_answer(ground_truth)
    
    # Scoring logic
    if answer is None:
        return 0.0
    
    # Complete match
    if answer == normalized_ground_truth:
        return score
    
    # Partial match
    if normalized_ground_truth in answer or answer in normalized_ground_truth:
        return format_score
    
    return 0.0

def evaluate_dataframe(df_path: str) -> pd.DataFrame:
    """Evaluate dataframe containing model responses.
    
    Args:
        df_path: Path to Parquet file containing model responses
        
    Returns:
        pd.DataFrame: Dataframe with evaluation results added
    """
    df = _load_dataframe(df_path)
    df = _maybe_drop_trailing_empty(df)

    response_col = None
    for candidate in ("responses", "response", "generated_sequence"):
        if candidate in df.columns:
            response_col = candidate
            break
    if response_col is None:
        raise ValueError("No response column found (expected responses/response/generated_sequence).")

    df["responses"] = df[response_col].apply(_extract_response_text)

    if "source_file" in df.columns:
        df["source_file"] = df["source_file"].apply(_normalize_source_file)
    elif "data_source" in df.columns:
        df["source_file"] = df["data_source"].apply(_normalize_source_file)
    else:
        df["source_file"] = ""

    answer_list = []
    partial_match_list = []

    ground_truths = df.apply(_extract_ground_truths, axis=1)
    for response_text, answers in zip(df["responses"].tolist(), ground_truths.tolist()):
        flag = False
        partial_flag = False
        if not answers:
            answer_list.append(False)
            partial_match_list.append(False)
            continue
        for ans in answers:
            score = compute_score(response_text, ans, format_score=0.50)
            if score == 1.0:
                flag = True
                break
            elif score == 0.50:  # format_score를 받은 경우 (부분 매치)
                partial_flag = True
        answer_list.append(flag)
        partial_match_list.append(partial_flag)

    # Add evaluation results
    df["answer"] = answer_list
    df["partial_match"] = partial_match_list  # format_score 받은 경우

    df = _attach_reference_metadata(df)
    return df


def _load_dataframe(df_path: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
    if isinstance(df_path, pd.DataFrame):
        return df_path.copy()
    path = Path(df_path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".json":
        return pd.read_json(path)
    raise ValueError(f"Unsupported file extension: {path}")


def _maybe_drop_trailing_empty(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    response_col = None
    for candidate in ("responses", "response", "generated_sequence"):
        if candidate in df.columns:
            response_col = candidate
            break
    if response_col is None:
        return df
    tail_text = _extract_response_text(df.iloc[-1][response_col])
    if not tail_text:
        return df.iloc[:-1]
    return df


def _extract_response_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        if len(value) == 0:
            return ""
        first = value[0]
        if first is None or (isinstance(first, float) and pd.isna(first)):
            return ""
        return first if isinstance(first, str) else str(first)
    return value if isinstance(value, str) else str(value)


def _coerce_answer_list(value) -> List[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, dict):
        if "target" in value:
            value = value["target"]
        elif "answers" in value:
            value = value["answers"]
        else:
            value = list(value.values())
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        return [str(v) for v in value if v is not None and not (isinstance(v, float) and pd.isna(v))]
    return [str(value)]


def _extract_ground_truths(row: pd.Series) -> List[str]:
    if "ground_truth" in row and row["ground_truth"] is not None:
        return _coerce_answer_list(row["ground_truth"])
    reward_model = row.get("reward_model")
    if isinstance(reward_model, dict) and reward_model.get("ground_truth") is not None:
        return _coerce_answer_list(reward_model.get("ground_truth"))
    if "golden_answers" in row and row["golden_answers"] is not None:
        return _coerce_answer_list(row["golden_answers"])
    return []


def _normalize_source_file(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value)
    if text.endswith(".json"):
        return text.split("/")[-1]
    key = text.lower()
    return DATA_SOURCE_TO_SOURCE_FILE.get(key, text)


def _attach_reference_metadata(df: pd.DataFrame) -> pd.DataFrame:
    if not REFERENCE_METADATA_PATH.exists():
        return df
    try:
        ref = pd.read_parquet(
            REFERENCE_METADATA_PATH,
            columns=["source_file", "data_source", "hop_nums", "type", "level"],
        )
    except Exception:
        return df
    if len(ref) != len(df):
        return df

    ref = ref.reset_index(drop=True)
    df_reset = df.reset_index(drop=True)

    for col in ("hop_nums", "type", "level"):
        if col not in df_reset.columns:
            df_reset[col] = ref[col]
        else:
            df_reset[col] = df_reset[col].where(df_reset[col].notna(), ref[col])

    if "source_file" not in df_reset.columns:
        df_reset["source_file"] = ref["source_file"]
    else:
        empty = df_reset["source_file"].isna() | (df_reset["source_file"].astype(str) == "")
        df_reset.loc[empty, "source_file"] = ref.loc[empty, "source_file"]

    return df_reset

def print_evaluation_results(df: pd.DataFrame) -> None:
    """Print evaluation results.
    
    Args:
        df: Dataframe containing evaluation results
    """
    print("EM Scores by source file:")
    print(df.groupby('source_file')['answer'].mean())
    print("\nPartial Match Scores by source file:")
    print(df.groupby('source_file')['partial_match'].mean())
    print("\nOverall EM Score:", df['answer'].mean())
    print("Overall Partial Match Score:", df['partial_match'].mean())
    print("\nSample data:")
    print(df.iloc[0].to_dict())

def main(file_path: str, output_path: str) -> None:
    """Main function.
    
    Args:
        file_path: Input data file path
        output_path: Output results file path
    """
    df = evaluate_dataframe(file_path)
    print_evaluation_results(df)
    df.to_json(output_path)

if __name__ == "__main__":
    input_file = 'stage2_3-7b_sft_123-rl-1-2.parquet'
    output_file = "sft123_rl12_pandas.json"
    main(input_file, output_file)
