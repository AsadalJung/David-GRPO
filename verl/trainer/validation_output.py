"""Filesystem persistence for complete validation generations.

The Ray PPO trainer is a single driver process. Validation worker outputs are
already gathered and unpadded before this module is called, so only that driver
publishes files. The extra rank guard protects alternate launchers that might
instantiate more than one trainer process.
"""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
import uuid
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


VALIDATION_RECORD_SCHEMA = "verl-validation-answer/v1"
VALIDATION_SESSION_SCHEMA = "verl-validation-session/v1"

_SEARCH_QUERY_PATTERN = re.compile(
    r"<begin_search>(.*?)</end_search>",
    flags=re.DOTALL | re.IGNORECASE,
)
_SEARCH_RESULT_PATTERN = re.compile(
    r"<search_result>(.*?)</search_result>",
    flags=re.DOTALL | re.IGNORECASE,
)
_SEARCH_TITLE_PATTERN = re.compile(
    r'result\s+\d+:\s*("(?:\\.|[^"\\])*")',
    flags=re.IGNORECASE,
)
_BOXED_ANSWER_PATTERN = re.compile(r"\\boxed\{([^{}]*)\}")
_ANSWER_TAG_PATTERN = re.compile(r"<answer>(.*?)</answer>", flags=re.DOTALL | re.IGNORECASE)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def is_primary_process(environ: Optional[Mapping[str, str]] = None) -> bool:
    """Return False when a common distributed launcher identifies a nonzero rank."""

    environment = os.environ if environ is None else environ
    for key in ("RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK"):
        raw_rank = environment.get(key)
        if raw_rank is None or str(raw_rank).strip() == "":
            continue
        try:
            if int(raw_rank) != 0:
                return False
        except (TypeError, ValueError):
            # An unparseable rank is safer to treat as non-primary.
            return False
    return True


def resolve_validation_output_base_dir(
    default_local_dir: str,
    configured_dir: Optional[str] = None,
) -> Path:
    """Resolve the configured root, or default below the run checkpoint root."""

    configured = str(configured_dir).strip() if configured_dir is not None else ""
    raw_path = configured or os.path.join(str(default_local_dir), "validation_outputs")
    return Path(os.path.expandvars(os.path.expanduser(raw_path)))


def to_jsonable(value: Any) -> Any:
    """Convert numpy/torch/pandas-style values into strict JSON values."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [to_jsonable(item) for item in sorted(value, key=repr)]

    # torch.Tensor: detach before moving to CPU. numpy/pandas objects skip this.
    detach = getattr(value, "detach", None)
    if callable(detach):
        try:
            value = detach()
            cpu = getattr(value, "cpu", None)
            if callable(cpu):
                value = cpu()
        except Exception:
            pass

    item = getattr(value, "item", None)
    if callable(item):
        try:
            scalar = item()
            if scalar is not value:
                return to_jsonable(scalar)
        except (TypeError, ValueError, RuntimeError):
            pass

    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            converted = tolist()
            if converted is not value:
                return to_jsonable(converted)
        except (TypeError, ValueError, RuntimeError):
            pass

    return str(value)


def extract_search_metadata(response: str) -> dict[str, Any]:
    """Parse tool queries/results while preserving each full result block."""

    text = response or ""
    queries = [match.strip() for match in _SEARCH_QUERY_PATTERN.findall(text)]
    results = [match.strip() for match in _SEARCH_RESULT_PATTERN.findall(text)]
    retrieved_titles = []
    for result in results:
        for title_token in _SEARCH_TITLE_PATTERN.findall(result):
            try:
                title = json.loads(title_token)
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(title, str):
                retrieved_titles.append(title)

    return {
        "query_count": len(queries),
        "result_block_count": len(results),
        "queries": queries,
        "result_blocks": results,
        "retrieved_titles": retrieved_titles,
    }


def extract_answer(response: str, data_source: Optional[str] = None) -> Optional[str]:
    """Use the task scorer's extractor when available, then conservative fallbacks."""

    text = response or ""
    if str(data_source).lower() == "hotpotqa":
        try:
            from verl.utils.reward_score.hotpotqa import extract_solution

            extracted = extract_solution(solution_str=text, method="strict")
            if extracted:
                return str(extracted)
        except Exception:
            # Persistence must not fail solely because an optional extractor changed.
            pass

    boxed = _BOXED_ANSWER_PATTERN.findall(text)
    if boxed:
        return boxed[-1].strip() or None
    tagged = _ANSWER_TAG_PATTERN.findall(text)
    if tagged:
        return tagged[-1].strip() or None
    return None


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _first_present(mapping: Mapping[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _question_from_raw_prompt(raw_prompt: Any) -> Optional[str]:
    if hasattr(raw_prompt, "tolist"):
        try:
            raw_prompt = raw_prompt.tolist()
        except (TypeError, ValueError):
            pass
    if not isinstance(raw_prompt, (list, tuple)):
        return None
    for message in reversed(raw_prompt):
        if not isinstance(message, Mapping):
            continue
        role = str(message.get("role", "")).lower()
        content = message.get("content")
        if role == "user" and content is not None:
            content_text = str(content).strip()
            lowered = content_text.lower()
            for marker in (
                "now, please answer the user's question below:",
                "please answer the following question:",
            ):
                marker_position = lowered.rfind(marker)
                if marker_position >= 0:
                    question = content_text[marker_position + len(marker):].strip()
                    if question:
                        return question
            return content_text
    return None


def build_validation_record(
    *,
    validation_row: int,
    prompt: str,
    response: str,
    total_reward: float,
    example_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one stable, self-contained validation answer record."""

    metadata = dict(example_metadata)
    extra_info = _mapping(metadata.get("extra_info"))
    reward_model = _mapping(metadata.get("reward_model"))
    supporting_meta = _mapping(metadata.get("supporting_facts"))

    row_id = _first_present(metadata, ("_id", "row_id", "id", "index"))
    if row_id is None:
        row_id = _first_present(extra_info, ("_id", "row_id", "id", "index"))

    prompt_id = _first_present(metadata, ("prompt_id", "uid", "original_prompt_id"))
    if prompt_id is None:
        prompt_id = _first_present(supporting_meta, ("prompt_id", "uid", "original_prompt_id"))
    if prompt_id is None:
        prompt_id = _first_present(extra_info, ("prompt_id", "uid", "original_prompt_id", "index"))
    if prompt_id is None:
        prompt_id = row_id

    raw_prompt = metadata.get("raw_prompt")
    question = _first_present(metadata, ("question", "query"))
    if question is None:
        question = _first_present(extra_info, ("question", "query"))
    if question is None:
        question = _question_from_raw_prompt(raw_prompt)

    gold_answer = _first_present(
        metadata,
        ("gold_answer", "ground_truth", "answer", "gold_response"),
    )
    if gold_answer is None:
        gold_answer = _first_present(
            reward_model,
            ("ground_truth", "gold_answer", "answer", "gold_response"),
        )

    reward_components = _first_present(
        metadata,
        ("reward_components", "score_components", "reward_scores"),
    )
    if reward_components is None:
        reward_components = _first_present(
            extra_info,
            ("reward_components", "score_components", "reward_scores"),
        )

    supporting_facts = metadata.get("supporting_facts")
    if supporting_facts is None:
        supporting_facts = extra_info.get("supporting_facts")
    support_documents = _first_present(
        metadata,
        ("support_documents", "support_docs", "retrieved_docs", "documents"),
    )
    if support_documents is None:
        support_documents = _first_present(
            extra_info,
            ("support_documents", "support_docs", "retrieved_docs", "documents"),
        )

    data_source = metadata.get("data_source")
    record = {
        "validation_row": int(validation_row),
        "row_id": to_jsonable(row_id),
        "prompt_id": to_jsonable(prompt_id),
        "data_source": to_jsonable(data_source),
        "question": to_jsonable(question),
        "prompt": str(prompt),
        "raw_prompt": to_jsonable(raw_prompt),
        "gold_answer": to_jsonable(gold_answer),
        "full_raw_response": str(response),
        "extracted_answer": extract_answer(response, data_source=data_source),
        "total_reward": to_jsonable(total_reward),
        "reward_components": to_jsonable(reward_components),
        "supporting_facts": to_jsonable(supporting_facts),
        "support_documents": to_jsonable(support_documents),
        "search_metadata": extract_search_metadata(response),
        # Retain all dataset/rollout metadata so newly added reward/search fields
        # are not silently lost before this persistence schema is updated.
        "example_metadata": to_jsonable(metadata),
    }
    return record


class ValidationOutputWriter:
    """Publish one immutable JSONL file per validation call."""

    def __init__(
        self,
        base_dir: Path,
        *,
        session_metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.session_id, self.session_dir = self._create_unique_session_dir()
        self.validation_call = 0

        metadata = {
            "schema": VALIDATION_SESSION_SCHEMA,
            "session_id": self.session_id,
            "created_at_utc": utc_now_iso(),
            "pid": os.getpid(),
            **dict(session_metadata or {}),
        }
        self._atomic_write_json(self.session_dir / "_session.json", metadata)

    def _create_unique_session_dir(self) -> tuple[str, Path]:
        for _ in range(32):
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            session_id = f"session_{timestamp}_pid{os.getpid()}_{uuid.uuid4().hex[:8]}"
            session_dir = self.base_dir / session_id
            try:
                session_dir.mkdir(mode=0o755)
            except FileExistsError:
                continue
            return session_id, session_dir
        raise FileExistsError(f"Could not allocate a unique validation session in {self.base_dir}")

    def write_step(
        self,
        *,
        global_step: int,
        records: Iterable[Mapping[str, Any]],
    ) -> Path:
        self.validation_call += 1
        call_number = self.validation_call
        target = self.session_dir / (
            f"global_step_{int(global_step):08d}_call_{call_number:04d}_{uuid.uuid4().hex[:8]}.jsonl"
        )

        enriched_records = []
        written_at = utc_now_iso()
        for record in records:
            enriched_record = dict(record)
            # Writer-owned fields cannot be overridden by caller metadata.
            enriched_record.update(
                {
                    "schema": VALIDATION_RECORD_SCHEMA,
                    "validation_session_id": self.session_id,
                    "validation_call": call_number,
                    "global_step": int(global_step),
                    "written_at_utc": written_at,
                }
            )
            enriched_records.append(enriched_record)
        self._atomic_write_jsonl(target, enriched_records)
        return target

    def _atomic_write_json(self, target: Path, payload: Mapping[str, Any]) -> None:
        serialized = json.dumps(
            to_jsonable(payload),
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        self._atomic_write_text(target, serialized + "\n")

    def _atomic_write_jsonl(
        self,
        target: Path,
        records: Iterable[Mapping[str, Any]],
    ) -> None:
        lines = (
            json.dumps(
                to_jsonable(record),
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
            + "\n"
            for record in records
        )
        self._atomic_write_text(target, lines)

    def _atomic_write_text(self, target: Path, content: Any) -> None:
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=str(target.parent),
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
                if isinstance(content, str):
                    handle.write(content)
                else:
                    handle.writelines(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, target)
            self._fsync_directory(target.parent)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise

    @staticmethod
    def _fsync_directory(directory: Path) -> None:
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        try:
            descriptor = os.open(directory, flags)
        except OSError:
            return
        try:
            os.fsync(descriptor)
        except OSError:
            pass
        finally:
            os.close(descriptor)
