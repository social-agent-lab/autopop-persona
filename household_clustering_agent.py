"""
Household Clustering System (based on AgentScope framework)
Uses two specialized agents:
1. FieldSelectionAgent (Field Selection Agent): Selects relevant fields for clustering
2. ClusteringAgent (Clustering Agent): Performs K-Prototypes clustering

"""

import os
import re
import json
import asyncio
import inspect
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
from dataclasses import dataclass

# AgentScope imports
from agentscope.agent import AgentBase
from agentscope.message import Msg
from agentscope.memory import InMemoryMemory
from agentscope.model import OpenAIChatModel

# Clustering-related imports
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# K-Prototypes clustering algorithm
try:
    from kmodes.kprototypes import KPrototypes
    KPROTOTYPES_AVAILABLE = True
except ImportError:
    KPROTOTYPES_AVAILABLE = False
    print("Warning: kmodes not installed. Please install using: pip install kmodes")

# K-Modes (fallback when all features are categorical)
try:
    from kmodes.kmodes import KModes
    KMODES_AVAILABLE = True
except ImportError:
    KMODES_AVAILABLE = False


@dataclass(frozen=True)
class _StructuralMissingRule:
    """Best-effort structural-missingness rule for a target column."""

    gate_col: str
    gate_values: List[Any]
    explained_ratio: float
    max_missing_rate: float
    min_missing_rate: float

def _run_coro_sync(coro):
    """Run coroutine in sync context; if a loop is running, ask caller to use async APIs."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    raise RuntimeError(
        "An event loop is already running; please use `await ..._async(...)` method calls in async context."
    )

def _msgs_to_openai_messages(payload: Any) -> List[Dict[str, str]]:
    """Convert Msg/list/dict payload into OpenAI-compatible messages list."""
    if payload is None:
        return []

    if not isinstance(payload, list):
        payload = [payload]

    messages: List[Dict[str, str]] = []
    for obj in payload:
        if isinstance(obj, dict):
            role = obj.get("role") or "user"
            content = obj.get("content", obj.get("text", ""))
        elif hasattr(obj, "role") and hasattr(obj, "content"):
            role = getattr(obj, "role")
            content = getattr(obj, "content")
        else:
            role = "user"
            content = obj

        if not isinstance(content, str):
            content = str(content)

        messages.append({"role": str(role), "content": content})

    return messages

def _blocks_to_text(value: Any) -> str:
    """Convert AgentScope ChatResponse.content blocks into plain text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        out = ""
        for item in value:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    out += item.get("text", "")
                else:
                    # ignore thinking/tool/audio blocks by default
                    continue
            else:
                out += str(item)
        return out
    return str(value)

async def _call_model_text_async(model, messages: List[Dict[str, str]], **kwargs) -> str:
    """Call AgentScope OpenAIChatModel and return text (handles streaming & block-based responses)."""
    if not messages:
        raise ValueError("LLM call failed: messages is empty (please check if memory correctly wrote system/user messages).")

    response = await model(messages=messages, **kwargs)

    # streaming response
    if hasattr(response, "__aiter__"):
        content = ""
        last_chunk = None
        async for chunk in response:
            last_chunk = chunk
            if isinstance(chunk, dict) and "content" in chunk:
                continue
            content += _blocks_to_text(getattr(chunk, "content", chunk))

        if content.strip():
            return content
        if last_chunk is not None:
            if isinstance(last_chunk, dict) and "content" in last_chunk:
                return _blocks_to_text(last_chunk.get("content"))
            return _blocks_to_text(getattr(last_chunk, "content", last_chunk))
        return ""

    # non-streaming response
    if isinstance(response, dict) or hasattr(response, "keys"):
        if "content" in response:
            return _blocks_to_text(response["content"])
        if "text" in response:
            return _blocks_to_text(response["text"])
        return str(response)

    return _blocks_to_text(getattr(response, "content", response))


# ============================================================================
# Agent 1: Field Selection Agent
# ============================================================================

class FieldSelectionAgent(AgentBase):
    """
    Intelligent field selection agent for clustering
    
    Responsibilities:
    - Analyze all available fields in the dataset
    - Identify fields suitable for clustering based on:
      * Data completeness
      * Variance and information content
      * Relevance to household behavior patterns
    - Classify fields as continuous or categorical
    - Provide reasoning for field selection
    """
    
    def __init__(
        self,
        name: str,
        model,
        sys_prompt: str = None,
        memory=None,
        target_selected_fields: Optional[int] = None,  # Expected number of fields for LLM to select (optional override; default adaptive by sample size/quality)
        min_selected_fields: Optional[int] = None,  # Minimum allowed number of fields (optional override; default adaptive by sample size/quality)
        min_completeness: float = 0.7,  # Minimum data completeness (non-null rate)
        min_variance_ratio: float = 0.01,  # Minimum coefficient of variation
        max_categorical_levels: int = 20,  # Maximum number of unique values for categorical variables
        max_selected_fields: int = 50,  # Maximum number of fields to auto-select
        max_continuous_fields: int = 30,  # Maximum number of continuous fields to auto-select
        max_categorical_fields: int = 20,  # Maximum number of categorical fields to auto-select
        correlation_threshold: float = 0.9,  # Correlation threshold for removing redundant continuous variables (Spearman)
        **kwargs
    ):
        """
        Initialize field selection agent
        
        Args:
            name: Agent name
            model: LLM model for decision making
            sys_prompt: System prompt
            memory: Memory module
            min_completeness: Minimum data completeness threshold
            min_variance_ratio: Minimum variance threshold
            max_categorical_levels: Maximum number of levels for categorical variables
        """
        super().__init__()
        
        self.name = name
        self.model = model
        self.memory = memory or InMemoryMemory()
        self.min_completeness = min_completeness
        self.min_variance_ratio = min_variance_ratio
        self.max_categorical_levels = max_categorical_levels
        self.max_selected_fields = max_selected_fields
        self.max_continuous_fields = max_continuous_fields
        self.max_categorical_fields = max_categorical_fields
        self.correlation_threshold = correlation_threshold
        self.target_selected_fields = target_selected_fields
        self.min_selected_fields = min_selected_fields
        
        # Default system prompt
        if sys_prompt is None:
            sys_prompt = self._get_default_system_prompt()
        self.sys_prompt = sys_prompt

        # InMemoryMemory's add/get_memory are async, here just mark, write system message on first call
        self._system_initialized = False
    
    def _get_default_system_prompt(self) -> str:
        """Get default system prompt for field selection"""
        return """You are an expert data scientist specializing in household behavior analysis and clustering.

Your task is to intelligently select fields from household survey data for clustering analysis.

Selection Criteria:
1. **Behavioral Relevance**: Fields that capture household behavior patterns, preferences, and decision-making
2. **Data Quality**: High completeness (low missing rate), sufficient variance
3. **Information Content**: Fields that differentiate households meaningfully
4. **Clustering Suitability**: Appropriate data types and distributions

Field Categories to Consider:
- Demographics: Age, family size, education, marital status
- Economic: Income, assets, debt, consumption patterns
- Behavioral: Employment, financial decisions, housing choices, health behaviors
- Attitudinal: Risk preferences, life satisfaction, future expectations

For each field, assess:
- Relevance score (0-10): How important for understanding household behavior
- Data quality (0-10): Completeness and variance
- Field type: continuous or categorical

Provide clear reasoning for inclusion or exclusion of each field."""
    
    async def reply_async(self, x: Msg = None) -> Msg:
        """
        Process field selection request
        
        Args:
            x: Input message containing dataset information
        
        Returns:
            Msg: Response message containing selected fields and reasoning process
        """
        if not self._system_initialized:
            await self.memory.add(Msg("system", self.sys_prompt, role="system"))
            self._system_initialized = True

        if x is not None:
            await self.memory.add(x)

        messages = _msgs_to_openai_messages(await self.memory.get_memory())
        content = await _call_model_text_async(
            self.model,
            messages
        )

        msg = Msg(self.name, content, role="assistant")
        await self.memory.add(msg)
        return msg

    def reply(self, x: Msg = None) -> Msg:
        """Synchronous wrapper: can be called directly when no event loop; use await reply_async() when event loop exists."""
        return _run_coro_sync(self.reply_async(x))
    
    async def analyze_fields_async(
        self,
        df: pd.DataFrame,
        excluded_patterns: List[str] = None,
        force_include: List[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze all fields in dataset and recommend fields for clustering
        
        Args:
            df: Input dataframe
            excluded_patterns: Regular expression patterns for excluding fields
            force_include: Fields to always include
        
        Returns:
            Dictionary containing field analysis and recommendations
        """
        print(f"\n{'='*80}")
        print(f"Field Selection Agent: Analyzing {len(df.columns)} fields")
        print(f"{'='*80}")
        
        excluded_patterns = excluded_patterns or [r'^id$', r'^.*_id$', r'^interview_']
        force_include = force_include or []
        
        field_analysis = []
        
        for col in df.columns:
            # Skip if matches exclusion pattern
            if any(re.match(pattern, col, re.IGNORECASE) for pattern in excluded_patterns):
                if col not in force_include:
                    continue
            
            analysis = self._analyze_single_field(df, col)
            field_analysis.append(analysis)
        
        # Create summary for LLM
        summary = self._create_field_summary(field_analysis)
        
        # Query LLM for field selection
        # Determine how many fields we want the LLM to select (adaptive by default; override-able via init args).
        target_fields, min_fields, candidate_pool_size = self._get_selection_bounds(
            field_analysis=field_analysis,
            n_samples=len(df),
        )

        # 1st attempt
        prompt_msg = Msg(
            "user",
            self._create_selection_prompt(
                summary=summary,
                n_samples=len(df),
                target_fields=target_fields,
                min_fields=min_fields,
                attempt=1,
            ),
            role="user"
        )
        
        response = await self.reply_async(prompt_msg)
        response_content = getattr(response, "content", "")
        if not isinstance(response_content, str):
            response_content = str(response_content)
        
        selected_fields = self._parse_selection_response_v2(
            response_content,
            field_analysis,
            df=df,
            force_include=force_include
        )

        # If LLM selected too few fields *and* we likely have enough usable candidates, retry once.
        if len(selected_fields) < min_fields and candidate_pool_size >= min_fields:
            prompt_msg_retry = Msg(
                "user",
                self._create_selection_prompt(
                    summary=summary,
                    n_samples=len(df),
                    target_fields=target_fields,
                    min_fields=min_fields,
                    attempt=2,
                    previous_count=len(selected_fields),
                ),
                role="user",
            )
            response_retry = await self.reply_async(prompt_msg_retry)
            response_retry_content = getattr(response_retry, "content", "")
            if not isinstance(response_retry_content, str):
                response_retry_content = str(response_retry_content)

            selected_retry = self._parse_selection_response_v2(
                response_retry_content,
                field_analysis,
                df=df,
                force_include=force_include,
            )
            if len(selected_retry) >= len(selected_fields):
                selected_fields = selected_retry

        if len(selected_fields) < min_fields:
            print(f"\n[WARN] LLM only selected {len(selected_fields)} fields (< {min_fields}), suggest checking field quality or adjusting prompt")
        
        return {
            'all_fields_analysis': field_analysis,
            'selected_fields': selected_fields,
            'llm_reasoning': response_content,
            'statistics': self._compute_selection_statistics(selected_fields)
        }

    def _get_selection_bounds(
        self,
        field_analysis: List[Dict[str, Any]],
        n_samples: int,
    ) -> Tuple[int, int, int]:
        """
        Compute (target_fields, min_fields, candidate_pool_size) adaptively.

        - `target_fields`: what we ask the LLM to aim for (soft target)
        - `min_fields`: a soft guardrail to avoid overly conservative outputs
        - `candidate_pool_size`: how many fields look usable by basic quality rules

        Clustering quality largely depends on the richness of field selection. More high-quality fields usually lead to better clustering results.
        """
        n_samples = int(max(1, n_samples))

        def _is_candidate(f: Dict[str, Any]) -> bool:
            if not isinstance(f, dict):
                return False
            ftype = f.get("field_type")
            if ftype not in {"continuous", "categorical", "binary"}:
                return False
            if float(f.get("completeness", 0.0)) < float(self.min_completeness):
                return False
            if ftype in {"categorical", "binary"} and int(f.get("n_unique", 0)) > int(self.max_categorical_levels):
                return False
            return True

        candidates = [f for f in field_analysis if _is_candidate(f)]
        good = [f for f in candidates if float(f.get("quality_score", 0.0)) >= 6.5]
        pool = good if len(good) >= 6 else candidates
        pool_size = len(pool)

        # More aggressive field selection strategy:
        # 1. Base target: 50-70% of candidate pool, ensuring sufficient feature dimensions
        # 2. Don't overly restrict field count for small samples, as more features help distinguish samples
        # 3. Upper limit determined by both candidate pool size and max_selected_fields
        
        # Calculate target based on candidate pool size (take 60% of pool)
        pool_based_target = int(round(pool_size * 0.6))
        
        # Soft adjustment based on sample size (slightly conservative for small samples, but not too conservative)
        # N=30 -> factor=0.85, N=100 -> factor=1.0, N=1000 -> factor=1.0
        sample_factor = min(1.0, 0.7 + 0.3 * np.log10(max(10, n_samples)) / 2)
        adjusted_target = int(round(pool_based_target * sample_factor))
        
        # Ensure at least a certain number of fields are selected (clustering needs sufficient features)
        min_reasonable = max(8, int(round(pool_size * 0.3)))  # At least 30% of pool or 8 fields
        target = max(min_reasonable, adjusted_target)
        
        # Apply upper limit
        target = min(target, self.max_selected_fields, pool_size)

        # Optional overrides (still bounded by pool_size/max_selected_fields)
        if isinstance(self.target_selected_fields, int) and self.target_selected_fields > 0:
            target = max(target, int(self.target_selected_fields))  # Changed to max, allow users to request more
            target = min(target, pool_size, self.max_selected_fields)

        # Minimum number of fields: 80% of target (stricter lower bound)
        min_fields = int(round(target * 0.8))
        min_fields = max(6, min_fields)  # At least 6 fields
        min_fields = min(min_fields, target, pool_size)

        if isinstance(self.min_selected_fields, int) and self.min_selected_fields > 0:
            min_fields = max(min_fields, int(self.min_selected_fields))
            min_fields = min(min_fields, target, pool_size)

        print(f"  Field selection target: target={target}, min={min_fields}, candidate pool={pool_size}")
        return int(target), int(min_fields), int(pool_size)

    def analyze_fields(
        self,
        df: pd.DataFrame,
        excluded_patterns: List[str] = None,
        force_include: List[str] = None
    ) -> Dict[str, Any]:
        """Synchronous wrapper: can be called directly when no event loop; use await analyze_fields_async() when event loop exists."""
        return _run_coro_sync(self.analyze_fields_async(df, excluded_patterns, force_include))
    
    def _analyze_single_field(self, df: pd.DataFrame, col: str) -> Dict[str, Any]:
        """Analyze a single field"""
        col_data = df[col]
        
        # Basic statistics
        n_total = len(col_data)
        n_non_null = col_data.notna().sum()
        completeness = n_non_null / n_total if n_total > 0 else 0
        
        # Determine field type
        n_unique = col_data.nunique()
        is_numeric = pd.api.types.is_numeric_dtype(col_data)
        
        # Field type classification
        if n_unique <= 1:
            field_type = 'constant'  # Constant
            variance_score = 0
        elif n_unique == 2:
            field_type = 'binary'  # Binary variable
            variance_score = 5
        elif is_numeric and n_unique > self.max_categorical_levels:
            field_type = 'continuous'  # Continuous variable
            # Variance score (more robust: use IQR/median when mean is close to 0)
            numeric = pd.to_numeric(col_data, errors='coerce').dropna()
            if numeric.empty:
                variance_score = 0
            else:
                mean = float(numeric.mean())
                std = float(numeric.std())
                median = float(numeric.median())
                q75 = float(numeric.quantile(0.75))
                q25 = float(numeric.quantile(0.25))
                iqr = max(0.0, q75 - q25)

                denom = max(abs(mean), abs(median), 1e-9)
                dispersion = max(std, iqr) / denom
                variance_score = float(min(10.0, dispersion * 5.0))
        elif n_unique <= self.max_categorical_levels:
            field_type = 'categorical'  # Categorical variable
            # Entropy-based variance score
            value_counts = col_data.value_counts(normalize=True)
            entropy = -(value_counts * np.log(value_counts + 1e-10)).sum()
            variance_score = min(10, entropy * 3)
        else:
            field_type = 'high_cardinality'  # High cardinality
            variance_score = 3
        
        # Quality score
        quality_score = (
            completeness * 5 +  # Completeness weight: 50%
            (variance_score / 10) * 3 +  # Variance weight: 30%
            (1 if field_type in ['continuous', 'categorical', 'binary'] else 0) * 2  # Type weight: 20%
        )

        # Dominant value ratio (for identifying near-constant fields)
        dominant_ratio = None
        nonnull = col_data.dropna()
        if len(nonnull) > 0:
            dominant_ratio = float(nonnull.value_counts(normalize=True, dropna=True).iloc[0])
        
        return {
            'field_name': col,
            'field_type': field_type,
            'completeness': completeness,
            'n_unique': n_unique,
            'variance_score': variance_score,
            'quality_score': quality_score,
            'is_numeric': is_numeric,
            'dominant_ratio': dominant_ratio,
            'sample_values': col_data.dropna().head(5).tolist() if n_non_null > 0 else []
        }
    
    def _create_field_summary(self, field_analysis: List[Dict]) -> str:
        """Create readable summary of field analysis"""
        summary_lines = [
            "Field Analysis Summary",
            "=" * 80,
            f"Total fields analyzed: {len(field_analysis)}",
            ""
        ]
        
        # Group by field type
        type_groups = {}
        for field in field_analysis:
            ftype = field['field_type']
            if ftype not in type_groups:
                type_groups[ftype] = []
            type_groups[ftype].append(field)
        
        for ftype, fields in type_groups.items():
            summary_lines.append(f"\n{ftype.upper()} Fields ({len(fields)}):")
            summary_lines.append("-" * 80)
            
            # Sort by quality score
            sorted_fields = sorted(fields, key=lambda x: x['quality_score'], reverse=True)
            
            for field in sorted_fields[:20]:  # Take top 20 of each type
                summary_lines.append(
                    f"  {field['field_name']:30s} | "
                    f"Completeness: {field['completeness']:.1%} | "
                    f"Quality: {field['quality_score']:.2f}/10 | "
                    f"Unique: {field['n_unique']}"
                )
        
        return "\n".join(summary_lines)
    
    def _create_selection_prompt(
        self,
        summary: str,
        n_samples: int,
        target_fields: int,
        min_fields: int,
        attempt: int = 1,
        previous_count: Optional[int] = None,
    ) -> str:
        """Create prompt for LLM field selection"""
        retry_note = ""
        if attempt > 1 and previous_count is not None:
            retry_note = (
                f"\nIMPORTANT: Your previous answer only selected {previous_count} fields. "
                f"Please expand to at least {min_fields} fields (preferably ~{target_fields}).\n"
            )

        return f"""
Please analyze the following household survey fields and select the most appropriate fields for clustering analysis.

Dataset Size: {n_samples} households

{summary}

{retry_note}
Task:
1. Review all fields and their statistics
2. Select fields that are:
   - Relevant for understanding household behavior and decision-making
   - High quality (good completeness and variance)
   - Suitable for clustering (continuous or low-cardinality categorical; avoid categoricals with too many unique values > {self.max_categorical_levels})
    
3. Categorize selected fields as:
   - CONTINUOUS: Numeric fields with many distinct values
   - CATEGORICAL: Discrete fields with few categories
    
4. Provide brief reasoning for each selection

**CRITICAL Selection size requirements:**
- You MUST select approximately {target_fields} fields (this is important for clustering quality!)
- You MUST select AT LEAST {min_fields} fields - selecting fewer will result in poor clustering
- More fields = better clustering differentiation. Be GENEROUS in your selection.
- Include diverse dimensions: demographics, income, wealth, consumption, housing, employment, etc.
- Do NOT be overly conservative - if a field has reasonable quality (completeness > 70%), include it
- Do NOT exceed {self.max_selected_fields} fields total

Output Format:
Return ONLY a valid JSON object (no markdown, no extra text).
Example:
{{"selected_fields":[{{"field_name":"field1","type":"continuous","reasoning":"..."}},{{"field_name":"field2","type":"categorical","reasoning":"..."}}],"summary":"..."}}

Please provide your selection now. Remember: select {target_fields} fields!
"""

    # ---------------------------------------------------------------------
    # v2: More robust LLM response parsing (LLM only)
    # ---------------------------------------------------------------------

    def _parse_selection_response_v2(
        self,
        llm_response: str,
        field_analysis: List[Dict],
        df: Optional[pd.DataFrame] = None,
        force_include: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Parse LLM response to extract selected fields (more robust):
        - First parse ```json ... ``` code blocks
        - Then try to parse JSON objects in text
        - If parsing fails, directly raise error (no automatic fallback)
        """
        force_include = force_include or []

        def _extract_first_balanced_json(text: str) -> Optional[str]:
            """Extract the first balanced JSON object substring from text (best-effort)."""
            if not isinstance(text, str) or "{" not in text:
                return None

            for start, ch in enumerate(text):
                if ch != "{":
                    continue
                depth = 0
                in_str = False
                esc = False
                for i in range(start, len(text)):
                    c = text[i]
                    if in_str:
                        if esc:
                            esc = False
                        elif c == "\\":
                            esc = True
                        elif c == '"':
                            in_str = False
                        continue

                    if c == '"':
                        in_str = True
                    elif c == "{":
                        depth += 1
                    elif c == "}":
                        depth -= 1
                        if depth == 0:
                            return text[start : i + 1]
            return None

        def _try_parse_json(text: str) -> Optional[Dict[str, Any]]:
            # Prefer fenced ```json ... ``` blocks (capture full block, not a partial object)
            m = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
            if m:
                block = m.group(1).strip()
                try:
                    return json.loads(block)
                except Exception:
                    balanced = _extract_first_balanced_json(block)
                    if balanced:
                        return json.loads(balanced)

            # Otherwise try the first balanced JSON object in the whole response
            balanced = _extract_first_balanced_json(text)
            if balanced:
                return json.loads(balanced)

            return None

        try:
            selection_data = _try_parse_json(llm_response)
            if isinstance(selection_data, dict):
                selected = selection_data.get('selected_fields') or selection_data.get('fields') or []
                validated = self._validate_selected_fields_v2(
                    selected_fields=selected,
                    field_analysis=field_analysis,
                    df=df,
                    force_include=force_include
                )
                if validated:
                    print(f"\n[OK] LLM selected {len(validated)} fields")
                    return validated
        except Exception as e:
            print(f"\n[WARN] Failed to parse LLM response: {e}")

        raise ValueError(
            "Field selection parsing failed: Please make LLM strictly output JSON (JSON only), and ensure it contains `selected_fields` field."
        )

    def _validate_selected_fields_v2(
        self,
        selected_fields: Any,
        field_analysis: List[Dict[str, Any]],
        df: Optional[pd.DataFrame],
        force_include: List[str]
    ) -> List[Dict[str, Any]]:
        """Clean/validate field list, ensure fields exist and types are usable."""
        analysis_by_name = {f.get('field_name'): f for f in field_analysis if isinstance(f, dict)}
        df_cols = set(df.columns) if isinstance(df, pd.DataFrame) else None

        normalized: List[Dict[str, Any]] = []

        if isinstance(selected_fields, list):
            for item in selected_fields:
                if isinstance(item, str):
                    item = {'field_name': item.strip()}
                if not isinstance(item, dict):
                    continue
                field_name = item.get('field_name') or item.get('name') or item.get('var_id') or item.get('var_name')
                if not isinstance(field_name, str) or not field_name.strip():
                    continue
                field_name = field_name.strip()
                if df_cols is not None and field_name not in df_cols:
                    continue

                inferred = analysis_by_name.get(field_name, {})
                field_type = (item.get('type') or "").strip().lower()
                if field_type not in {"continuous", "categorical"}:
                    inferred_type = inferred.get('field_type')
                    field_type = 'continuous' if inferred_type == 'continuous' else 'categorical'

                normalized.append({
                    'field_name': field_name,
                    'type': field_type,
                    'reasoning': item.get('reasoning') or item.get('reason') or "Selected by LLM"
                })

        # Force include fields first
        existing = {f['field_name'] for f in normalized if isinstance(f, dict) and 'field_name' in f}
        for col in force_include:
            if col in existing:
                continue
            if df_cols is not None and col not in df_cols:
                continue
            inferred = analysis_by_name.get(col, {})
            inferred_type = inferred.get('field_type')
            normalized.append({
                'field_name': col,
                'type': 'continuous' if inferred_type == 'continuous' else 'categorical',
                'reasoning': "Force-included"
            })

        # Deduplicate and trim
        seen = set()
        deduped: List[Dict[str, Any]] = []
        for item in normalized:
            name = item.get('field_name')
            if name in seen:
                continue
            seen.add(name)
            deduped.append(item)

        return deduped[: max(1, self.max_selected_fields)]

    def _compute_selection_statistics(self, selected_fields: List[Dict]) -> Dict:
        """Calculate statistics for selected fields"""
        n_continuous = sum(1 for f in selected_fields if f['type'] == 'continuous')
        n_categorical = sum(1 for f in selected_fields if f['type'] == 'categorical')
        
        return {
            'total_selected': len(selected_fields),
            'n_continuous': n_continuous,
            'n_categorical': n_categorical
        }


# ============================================================================
# Automatically perform clustering (K-Prototypes)
# ============================================================================

class ClusteringAgent:
    """
    Used to perform K-Prototypes clustering on household data
    
    Responsibilities:
    - Prepare data for clustering (handle missing values, encode categories)
    - Determine optimal number of clusters
    - Perform K-Prototypes clustering
    - Analyze and interpret cluster profiles
    - Generate visualizations and reports
    """
    
    def __init__(
        self,
        *args,
        max_clusters: int = 15,
        min_clusters: int = 2,
        random_state: int = 42,
        **kwargs
    ):
        self.max_clusters = int(max_clusters)
        self.min_clusters = int(min_clusters)
        self.random_state = int(random_state)

        self.clustering_results = None
    
    def perform_clustering(
        self,
        df: pd.DataFrame,
        continuous_fields: List[str],
        categorical_fields: List[str],
        n_clusters: int = None,
        output_dir: str = None
    ) -> Dict[str, Any]:
        """
        Perform K-Prototypes clustering
        
        Args:
            df: Input dataframe
            continuous_fields: List of continuous field names
            categorical_fields: List of categorical field names
            n_clusters: Number of clusters (None means auto-determine)
            output_dir: Directory to save results
        
        Returns:
            Dictionary containing clustering results
        """
        if not KPROTOTYPES_AVAILABLE:
            raise ImportError("K-Prototypes requires kmodes package. Install command: pip install kmodes")
        
        print(f"\n{'='*80}")
        print("Clustering Agent: Performing clustering")
        print(f"{'='*80}")
        
        # Prepare data
        X, feature_names, cat_indices, X_num = self._prepare_data(
            df, continuous_fields, categorical_fields
        )
        
        print(f"\nData preparation completed:")
        print(f"  Number of samples: {X.shape[0]}")
        print(f"  Number of features: {X.shape[1]}")
        print(f"  Continuous variables: {len(continuous_fields)}")
        print(f"  Categorical variables: {len(categorical_fields)}")
        
        # Decide algorithm: K-Prototypes will error when no continuous features, use K-Modes instead
        use_kmodes = len(continuous_fields) == 0 and len(categorical_fields) > 0
        if use_kmodes and not KMODES_AVAILABLE:
            raise ImportError("Currently selected fields are all categorical, need KModes. Install command: pip install kmodes")

        algo_name = "K-Modes" if use_kmodes else "K-Prototypes"
        print(f"\nAlgorithm selection: {algo_name}")

        # If cluster number not specified: use "field feature heuristic" to estimate k (no multi-round fitting search, avoid complexity/instability)
        auto_k = n_clusters is None
        if auto_k:
            n_clusters = self._estimate_k_from_features(
                df, continuous_fields, categorical_fields, n_samples=int(X.shape[0])
            )
            n_clusters = int(max(self.min_clusters, min(int(n_clusters), self.max_clusters)))
            print(f"[OK] Suggested number of clusters based on field features: {n_clusters}")
        
        print(f"\nNumber of clusters: {n_clusters}")

        # Minimum cluster size threshold: clusters smaller than this will be marked as noise (-1) and discarded
        # Reason: Small clusters may be unreliable due to missing value imputation issues, not suitable for persona generation
        n_samples = int(X.shape[0])
        min_cluster_size_target = max(5, int(round(0.01 * n_samples)))  # >=1% or >=5 samples
        if auto_k:
            print(f"  [INFO] Minimum cluster size threshold: {min_cluster_size_target} (clusters smaller than this will be marked as noise)")
        
        # Perform clustering (no longer reduce k due to small clusters, allow small clusters to be generated)
        if use_kmodes:
            labels, model = self._fit_kmodes(X, n_clusters)
        else:
            labels, model = self._fit_kprototypes(X, cat_indices, n_clusters)
        
        # Post-processing: mark clusters smaller than threshold as noise (-1)
        labels = self._filter_small_clusters(labels, min_cluster_size_target)
        
        # Update valid cluster count (excluding noise points)
        valid_cluster_ids = [cid for cid in np.unique(labels) if cid != -1]
        n_valid_clusters = len(valid_cluster_ids)
        
        # Print filtered valid cluster sizes
        if n_valid_clusters < n_clusters:
            print(f"\n[INFO] Filtered valid clusters:")
            unique, counts = np.unique(labels, return_counts=True)
            for cluster_id, count in zip(unique, counts):
                if cluster_id == -1:
                    print(f"  Noise points: {count} households ({count/len(labels)*100:.1f}%)")
                else:
                    print(f"  Cluster {cluster_id}: {count} households ({count/len(labels)*100:.1f}%)")
            print(f"  Valid clusters: {n_valid_clusters} (original k={n_clusters})")
        
        # Analyze clusters (automatically filter noise points)
        cluster_profiles = self._analyze_cluster_profiles(
            df, labels, continuous_fields, categorical_fields
        )
        
        # Use LLM to generate explanation (disabled: this file only handles clustering; explanation/Persona handled by persona_generation.py and other post-processing)
        # interpretation = self._generate_cluster_interpretation(cluster_profiles, n_valid_clusters)
        interpretation = None
        
        # Visualize results
        if output_dir:
            self._visualize_results(
                df, X_num, labels, feature_names, continuous_fields,
                categorical_fields, output_dir
            )
        
        # Store results (using valid cluster count)
        self.clustering_results = {
            'labels': labels,
            'model': model,
            'n_clusters': n_valid_clusters,  # Use filtered valid cluster count
            'n_clusters_original': n_clusters,  # Keep original k value
            'cluster_profiles': cluster_profiles,
            'interpretation': interpretation,
            'feature_names': feature_names,
            'categorical_indices': cat_indices
        }
        
        return self.clustering_results
    
    def _prepare_data(
        self,
        df: pd.DataFrame,
        continuous_fields: List[str],
        categorical_fields: List[str]
    ) -> Tuple[np.ndarray, List[str], List[int], Optional[np.ndarray]]:
       
        
        # Merge features
        all_fields = continuous_fields + categorical_fields
        if len(all_fields) == 0:
            raise ValueError("（continuous_fields/categorical_fields null）")

        missing_cols = [c for c in all_fields if c not in df.columns]
        if missing_cols:
            raise KeyError(f"{missing_cols}")

        X_df = df[all_fields].copy()

        def _is_blank_obj_series(s: pd.Series) -> pd.Series:
            if s.dtype != object:
                return pd.Series(False, index=s.index)
            try:
                return s.astype(str).str.strip().eq("")
            except Exception:
                return pd.Series(False, index=s.index)

        def _real_missing_token_hit(v: Any) -> bool:
            t = str(v).strip().lower()
            return any(tok in t for tok in ["refus", "don't know", "dont know", "dk", "unknown", "not ascertained"])

        def _find_structural_rule(
            *,
            df_full: pd.DataFrame,
            target_col: str,
            target_missing: pd.Series,
            gate_candidates: List[str],
            high: float,
            low: float,
            min_gap: float,
            min_explained: float,
            min_group_size: int,
        ) -> Optional[_StructuralMissingRule]:
            n_missing = int(target_missing.sum())
            if n_missing <= 0:
                return None

            best: Optional[_StructuralMissingRule] = None
            best_score = -1.0

            for gate in gate_candidates:
                if gate == target_col or gate not in df_full.columns:
                    continue

                g = df_full[gate]
                # Avoid using "missingness indicator" / non-informative gates when possible.
                # If gate values themselves look like refusal/unknown, skip to avoid misclassifying real missingness.
                try:
                    sample_vals = g.dropna().unique().tolist()[:12]
                except Exception:
                    sample_vals = []
                if any(_real_missing_token_hit(v) for v in sample_vals):
                    continue

                key = g.where(g.notna(), "<NA>")
                try:
                    grouped = target_missing.groupby(key)
                    sizes = grouped.size()
                    if int((sizes >= min_group_size).sum()) < 2:
                        continue
                    rates = grouped.mean()
                except Exception:
                    continue

                # Filter tiny groups
                rates = rates[sizes >= min_group_size]
                if rates.empty or len(rates) < 2:
                    continue

                max_rate = float(rates.max())
                min_rate = float(rates.min())
                if max_rate < high or min_rate > low:
                    continue
                if (max_rate - min_rate) < min_gap:
                    continue

                structural_vals = rates.index[rates >= high].tolist()
                if not structural_vals:
                    continue
                if any(_real_missing_token_hit(v) for v in structural_vals):
                    continue

                explained = float((target_missing & key.isin(structural_vals)).sum() / n_missing)
                if explained < min_explained:
                    continue

                # Score: prioritize explaining missingness, then separation.
                score = explained * 10.0 + (max_rate - min_rate)
                if score > best_score:
                    best_score = score
                    best = _StructuralMissingRule(
                        gate_col=gate,
                        gate_values=structural_vals,
                        explained_ratio=explained,
                        max_missing_rate=max_rate,
                        min_missing_rate=min_rate,
                    )

            return best

        # Candidate gate columns: low-cardinality columns with reasonable completeness.
        n_rows = int(len(df))
        min_group_size = max(3, int(round(n_rows * 0.02)))
        gate_candidates: List[str] = []
        try:
            nunique = df.nunique(dropna=True)
        except Exception:
            nunique = pd.Series({c: df[c].nunique(dropna=True) for c in df.columns})

        for c in df.columns:
            if c not in nunique:
                continue
            u = int(nunique[c])
            if u < 2 or u > 12:
                continue
            try:
                if float(df[c].isna().mean()) > 0.35:
                    continue
            except Exception:
                continue
            gate_candidates.append(c)

        impute_notes: List[str] = []

        # Continuous: detect structural NA -> 0, then real NA -> median (prefer within applicable group)
        for col in continuous_fields:
            s_num = pd.to_numeric(X_df[col], errors="coerce")
            target_missing = s_num.isna()
            if bool(target_missing.any()):
                rule = _find_structural_rule(
                    df_full=df,
                    target_col=col,
                    target_missing=target_missing,
                    gate_candidates=gate_candidates,
                    high=0.9,
                    low=0.1,
                    min_gap=0.7,
                    min_explained=0.6,
                    min_group_size=min_group_size,
                )
                structural_mask = pd.Series(False, index=s_num.index)
                gate_key = None
                if rule is not None and rule.gate_col in df.columns:
                    gate_key = df[rule.gate_col].where(df[rule.gate_col].notna(), "<NA>")
                    structural_mask = target_missing & gate_key.isin(rule.gate_values)
                    if int(structural_mask.sum()) > 0:
                        s_num.loc[structural_mask] = 0.0
                        impute_notes.append(
                            f"{col}: structural={int(structural_mask.sum())} -> 0 (gate={rule.gate_col})"
                        )

                # Real missingness: median imputation, preferably within applicable group
                real_missing = s_num.isna()
                if bool(real_missing.any()):
                    median_val = np.nan
                    if rule is not None and gate_key is not None:
                        applicable = ~gate_key.isin(rule.gate_values)
                        median_val = float(s_num[applicable & s_num.notna()].median())
                    if not np.isfinite(median_val):
                        try:
                            median_val = float(s_num.median())
                        except Exception:
                            median_val = np.nan
                    if not np.isfinite(median_val):
                        median_val = 0.0
                    s_num = s_num.fillna(median_val)

            X_df[col] = s_num.astype(float)

        # Categorical: detect structural NA -> "Inapplicable", then real NA -> "Missing"
        for col in categorical_fields:
            s = X_df[col]
            target_missing = s.isna() | _is_blank_obj_series(s)
            if bool(target_missing.any()):
                rule = _find_structural_rule(
                    df_full=df,
                    target_col=col,
                    target_missing=target_missing,
                    gate_candidates=gate_candidates,
                    high=0.8,
                    low=0.2,
                    min_gap=0.6,
                    min_explained=0.5,
                    min_group_size=min_group_size,
                )
                if rule is not None and rule.gate_col in df.columns:
                    gate_key = df[rule.gate_col].where(df[rule.gate_col].notna(), "<NA>")
                    structural_mask = target_missing & gate_key.isin(rule.gate_values)
                    if int(structural_mask.sum()) > 0:
                        s = s.copy()
                        s.loc[structural_mask] = "Inapplicable"
                        impute_notes.append(
                            f"{col}: structural={int(structural_mask.sum())} -> Inapplicable (gate={rule.gate_col})"
                        )

            # Remaining missing -> "Missing"
            s = s.where(~(s.isna() | _is_blank_obj_series(s)), "Missing")
            X_df[col] = s.astype(str)

        if impute_notes:
            preview = "; ".join(impute_notes[:8])
            more = f" (+{len(impute_notes) - 8} more)" if len(impute_notes) > 8 else ""
            print(f"  [IMPUTE] Structural missingness handled: {preview}{more}")

        # Standardize continuous features (z-score) to avoid scale dominance (also better for subsequent continuous metric evaluation)
        X_num: Optional[np.ndarray]
        if len(continuous_fields) > 0:
            scaler = StandardScaler()
            X_num = scaler.fit_transform(X_df[continuous_fields].astype(float).values)
        else:
            X_num = None

        # Categorical features kept as strings (K-Prototypes uses matching distance for categorical columns)
        X_cat = X_df[categorical_fields].values.astype(object) if len(categorical_fields) > 0 else None

        if X_num is None and X_cat is None:
            raise ValueError("Data preparation failed: both continuous and categorical features are empty")
        if X_num is None:
            X = X_cat
        elif X_cat is None:
            X = X_num.astype(object)
        else:
            X = np.concatenate([X_num.astype(object), X_cat], axis=1)

        # Categorical indices are in the second half after concatenation
        categorical_indices = list(range(len(continuous_fields), len(all_fields)))

        return X, all_fields, categorical_indices, X_num
    
    def _estimate_k_from_features(
        self,
        df: pd.DataFrame,
        continuous_fields: List[str],
        categorical_fields: List[str],
        n_samples: int
    ) -> int:
        """
        Automatically estimate reasonable number of clusters based on selected field characteristics
        
        Strategy:
        1. Analyze cardinality of categorical variables (number of unique values)
        2. Analyze distribution characteristics of continuous variables
        3. Combine with sample size for estimation
        
        Returns:
            Suggested number of clusters
        """
        # print("\n  Analyzing field characteristics to estimate cluster count...")
        
        # ========== 1. Analyze categorical variables ==========
        cat_cardinalities = []
        key_cat_fields = []  # Key stratification fields (high cardinality, high discriminability)
        
        placeholder_cats = {"missing", "inapplicable", "<na>", "nan"}
        for field in categorical_fields:
            if field not in df.columns:
                continue

            # Exclude placeholder categories introduced by missing handling (e.g., "Missing"/"Inapplicable")
            s = df[field]
            try:
                s_str = s.astype(str).str.strip()
            except Exception:
                s_str = s.astype(object).astype(str).str.strip()
            s_str = s_str.where(~s_str.eq(""), np.nan).dropna()
            s_lower = s_str.str.lower()
            s_valid = s_str[~s_lower.isin(placeholder_cats)]

            n_unique = int(s_valid.nunique())
            cat_cardinalities.append(n_unique)

            # Identify key stratification fields (moderate cardinality + more uniform distribution = higher discriminability)
            if 2 <= n_unique <= 25 and len(s_valid) >= 10:
                value_counts = s_valid.value_counts(normalize=True)
                entropy = float(-(value_counts * np.log2(value_counts + 1e-10)).sum())
                max_entropy = float(np.log2(n_unique)) if n_unique > 1 else 0.0
                normalized_entropy = float(entropy / max_entropy) if max_entropy > 0 else 0.0
                effective_levels = float(min(2.0 ** entropy, float(n_unique))) if entropy > 0 else float(n_unique)
                score = float(normalized_entropy * np.log2(n_unique + 1.0))

                if normalized_entropy > 0.5:  # More uniform distribution, higher discriminability
                    key_cat_fields.append((field, n_unique, normalized_entropy, effective_levels, score))
        
        # ========== 2. Analyze continuous variables ==========
        cont_variations = []
        bimodal_fields = []  # Possible bimodal/multimodal distributions
        
        for field in continuous_fields:
            if field not in df.columns:
                continue
            data = pd.to_numeric(df[field], errors='coerce').dropna()
            if len(data) < 10:
                continue
            
            # Calculate coefficient of variation
            mean_val = data.mean()
            std_val = data.std()
            if abs(mean_val) > 1e-6:
                cv = std_val / abs(mean_val)
            else:
                cv = std_val
            cont_variations.append(cv)
            
            # Detect obvious multimodal distribution (using interquartile range comparison)
            q1, q2, q3 = data.quantile([0.25, 0.5, 0.75])
            iqr = q3 - q1
            # If data range is large and median is skewed to one side, there may be subgroups
            if iqr > 0 and (((q2 - q1) / iqr) < 0.3 or ((q3 - q2) / iqr) < 0.3):
                bimodal_fields.append(field)
        
        # ========== 3. Synthesize "prior suggested k" (does not directly determine final K) ==========

        # Base value (prior): smoothly scale based on sample size (avoid hardcoded segment thresholds)
        # Empirically, log2(N)/2 gives a reasonable scale of 3~8 for common sample sizes.
        n_samples = int(max(1, n_samples))
        base_k = int(round(float(np.log2(max(2, n_samples))) / 2.0))
        base_k = max(self.min_clusters, min(base_k, self.max_clusters))
        
        # Adjust based on key categorical fields
        if key_cat_fields:
            # Take top 2 most discriminative fields
            key_cat_fields.sort(key=lambda x: -x[4])  # Sort by composite (entropy, cardinality) score descending
            top_cats = key_cat_fields[:3]
            
            # These fields suggest natural stratification in the data
            # Example: Education level (5 levels) x Employment status (3 levels) suggests ~10 meaningful combinations
            # Use "effective level count of key categorical fields" to estimate reasonable k (avoid k explosion from products)
            eff_levels = [min(float(eff_lvls), 10.0) for _, _, _, eff_lvls, _ in top_cats if eff_lvls]
            if eff_levels:
                implied_groups = int(round(float(np.mean(eff_levels))))
            else:
                implied_groups = base_k

            cat_suggested_k = max(self.min_clusters, min(implied_groups, self.max_clusters))
            # print(f"    Key categorical fields: {[f[0] for f in top_cats]}")
            # print(f"    Categorical fields suggest stratification count: ~{implied_groups} -> suggested k={cat_suggested_k}")
        else:
            cat_suggested_k = base_k
        
        # Adjust based on continuous variable variation
        if cont_variations:
            cvs = np.asarray(cont_variations, dtype=float)
            avg_cv = float(np.mean(cvs))
            high_ratio = float(np.mean(cvs > 1.0))
            if avg_cv > 1.5 or high_ratio > 0.5:
                # High variation, may have multiple subgroups
                cont_adjustment = 2
            elif avg_cv > 0.8 or high_ratio > 0.25:
                cont_adjustment = 1
            else:
                cont_adjustment = 0
        else:
            cont_adjustment = 0
        
        bimodal_adjustment = min(len(bimodal_fields), 2)
        
        estimated_k = max(
            base_k,
            cat_suggested_k,
            base_k + cont_adjustment + bimodal_adjustment
        )
        
        estimated_k = max(self.min_clusters, min(estimated_k, self.max_clusters))
        
        return estimated_k

    def _determine_optimal_clusters(
        self,
        X: np.ndarray,
        categorical_indices: List[int],
        output_dir: str,
        X_num: Optional[np.ndarray],
        df: pd.DataFrame,
        continuous_fields: List[str],
        categorical_fields: List[str]
    ) -> int:
        """
        Determine optimal number of clusters using cost/distortion analysis and field characteristics
        """
        print("\nDetermining optimal number of clusters...")

        costs: List[float] = []
        silhouettes: List[float] = []
        calinskis: List[float] = []
        davies: List[float] = []
        min_cluster_fracs: List[float] = []
        small_cluster_fracs: List[float] = []
        imbalance_ratios: List[float] = []

        # With small samples/high duplication, k-prototypes may fail to initialize (needs at least k "distinguishable" sample points)
        # Here we estimate unique row count based on input X (already standardized/concatenated) to avoid attempting k values doomed to fail.
        def _row_key(row: np.ndarray) -> tuple:
            out = []
            for v in row.tolist():
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    out.append("<NA>")
                elif isinstance(v, (int, np.integer)):
                    out.append(int(v))
                elif isinstance(v, (float, np.floating)):
                    fv = float(v)
                    out.append(round(fv, 6) if np.isfinite(fv) else "<NA>")
                else:
                    out.append(str(v).strip())
            return tuple(out)

        # For large datasets, avoid full deduplication statistics (high memory usage), sampling estimate suffices.
        try:
            n_rows = int(X.shape[0])
            if n_rows > 20000:
                rng_u = np.random.RandomState(self.random_state)
                idx_u = rng_u.choice(n_rows, size=20000, replace=False)
                X_u = X[idx_u]
                n_unique = len({_row_key(row) for row in X_u})
                if n_unique >= self.max_clusters:
                    n_unique = self.max_clusters
            else:
                n_unique = len({_row_key(row) for row in X})
        except Exception:
            n_unique = int(X.shape[0])

        max_k = min(self.max_clusters, n_unique)
        if max_k < self.min_clusters:
            print(
                f"  [WARN] Unique combinations({n_unique}) < min_clusters({self.min_clusters}), "
                f"compressing candidate k upper bound to {max_k}"
            )
        effective_min_k = min(self.min_clusters, max_k)
        if effective_min_k < 2:
            effective_min_k = 2
        if max_k < effective_min_k:
            raise ValueError(
                f"Cannot select cluster count: too few valid samples/unique combinations (unique_rows={n_unique}, min_k={effective_min_k})"
            )
        if max_k < self.max_clusters:
            print(f"  [INFO] Many duplicate samples detected: unique_rows={n_unique}, limiting max_clusters to {max_k}")

        K_range = range(effective_min_k, max_k + 1)

        rng = np.random.RandomState(self.random_state)
        metric_sample_size = 5000
        if X_num is not None and X_num.shape[0] > metric_sample_size:
            metric_idx = rng.choice(X_num.shape[0], size=metric_sample_size, replace=False)
            X_num_metric = X_num[metric_idx]
        else:
            metric_idx = None
            X_num_metric = X_num

        for k in K_range:
            try:
                kproto = None
                last_err: Optional[Exception] = None
                init_methods = ("Cao", "Huang", "random")
                seed_offsets = (0, 101, 202)
                for attempt, init_method in enumerate(init_methods):
                    if kproto is not None:
                        break
                    for seed_offset in seed_offsets:
                        try:
                            kproto = self._make_kprototypes(
                                n_clusters=k,
                                verbose=0,
                                init=init_method,
                                random_state=self.random_state + 17 * attempt + seed_offset + k,
                                n_init=10,
                            )
                            kproto.fit(X, categorical=categorical_indices)
                            last_err = None
                            break
                        except Exception as e:
                            last_err = e
                            kproto = None

                if kproto is None:
                    assert last_err is not None
                    raise last_err

                costs.append(float(kproto.cost_))
                # cluster size diagnostics (used later for k selection penalties)
                try:
                    labels_full = np.asarray(getattr(kproto, "labels_", []), dtype=int)
                    if labels_full.size > 0:
                        counts = np.bincount(labels_full, minlength=k)
                        min_frac = float(counts.min()) / float(labels_full.size)
                        small_threshold = max(10, int(0.01 * labels_full.size))
                        small_frac = float(np.mean(counts < small_threshold))
                        imbalance = float(counts.max() / max(int(counts.min()), 1))
                    else:
                        min_frac, small_frac, imbalance = np.nan, np.nan, np.nan
                except Exception:
                    min_frac, small_frac, imbalance = np.nan, np.nan, np.nan
                min_cluster_fracs.append(min_frac)
                small_cluster_fracs.append(small_frac)
                imbalance_ratios.append(imbalance)

                if X_num_metric is not None and X_num_metric.shape[1] >= 1:
                    labels = kproto.labels_
                    if metric_idx is not None:
                        labels = labels[metric_idx]
                    try:
                        sil = float(silhouette_score(X_num_metric, labels))
                    except Exception:
                        sil = np.nan
                    try:
                        ch = float(calinski_harabasz_score(X_num_metric, labels))
                    except Exception:
                        ch = np.nan
                    try:
                        db = float(davies_bouldin_score(X_num_metric, labels))
                    except Exception:
                        db = np.nan
                else:
                    sil, ch, db = np.nan, np.nan, np.nan

                silhouettes.append(sil)
                calinskis.append(ch)
                davies.append(db)

                # if not np.isnan(sil):
                #     print(f" 成本={kproto.cost_:.2f}, silhouette={sil:.3f}")
                # else:
                #     print(f" 成本={kproto.cost_:.2f}")
            except Exception as e:
                print(f" Failed(k={k}): {e}")
                costs.append(np.nan)
                silhouettes.append(np.nan)
                calinskis.append(np.nan)
                davies.append(np.nan)
                min_cluster_fracs.append(np.nan)
                small_cluster_fracs.append(np.nan)
                imbalance_ratios.append(np.nan)
        
        # 使用肘部法则
        if not any(np.isfinite(c) for c in costs):
            raise RuntimeError(
                "K-Prototypes initialization failed: unable to fit on any candidate k. "
                "Suggestions: 1) Lower max_clusters; 2) Check if categorical fields are nearly identical/all missing; 3) Verify with continuous or categorical variables only."
            )

        elbow_k = self._find_elbow(K_range, costs)
        
        n_samples = X.shape[0]
        
        # ========== 基于字段特征估算合理的聚类数 ==========
        raw_feature_suggested_k = self._estimate_k_from_features(
            df, continuous_fields, categorical_fields, n_samples
        )
        
        # 确保 feature_suggested_k 不超过 max_k
        feature_suggested_k = min(raw_feature_suggested_k, max_k)
        if raw_feature_suggested_k > max_k:
            print(
                f"  [INFO] Field features suggest k={raw_feature_suggested_k} exceeds searchable limit {max_k}, truncated to {feature_suggested_k}"
            )
        
        k_values = np.array(list(K_range), dtype=int)
        costs_arr = np.asarray(costs, dtype=float)
        valid_cost = np.isfinite(costs_arr)

        def _minmax(values, higher_is_better: bool = True):
            arr = np.asarray(values, dtype=float)
            mask = np.isfinite(arr)
            if int(mask.sum()) == 0:
                return None
            vmin = float(np.nanmin(arr[mask]))
            vmax = float(np.nanmax(arr[mask]))
            if vmax - vmin == 0:
                scaled = np.full_like(arr, 0.5, dtype=float)
            else:
                scaled = (arr - vmin) / (vmax - vmin)
            if not higher_is_better:
                scaled = 1.0 - scaled
            scaled[~mask] = np.nan
            return scaled

        sil_scaled = _minmax(silhouettes, higher_is_better=True)
        ch_raw = np.asarray(calinskis, dtype=float)
        ch_log = np.where(np.isfinite(ch_raw) & (ch_raw > 0), np.log1p(ch_raw), np.nan)
        ch_scaled = _minmax(ch_log, higher_is_better=True)
        db_scaled = _minmax(davies, higher_is_better=False)
        cost_scaled = _minmax(costs, higher_is_better=False)  

        quality_sum = np.zeros(len(k_values), dtype=float)
        quality_n = np.zeros(len(k_values), dtype=float)
        for comp in (sil_scaled, ch_scaled, db_scaled, cost_scaled):
            if comp is None:
                continue
            m = np.isfinite(comp)
            quality_sum[m] += comp[m]
            quality_n[m] += 1.0
        quality = np.divide(
            quality_sum,
            quality_n,
            out=np.full_like(quality_sum, np.nan),
            where=quality_n > 0,
        )

        cost_improve = np.full(len(k_values), np.nan)
        for i in range(1, len(k_values)):
            if np.isfinite(costs_arr[i]) and np.isfinite(costs_arr[i - 1]) and abs(costs_arr[i - 1]) > 1e-12:
                cost_improve[i] = (costs_arr[i - 1] - costs_arr[i]) / abs(costs_arr[i - 1])

        cost_improve_to_next = np.full(len(k_values), np.nan)
        for i in range(0, len(k_values) - 1):
            if np.isfinite(costs_arr[i]) and np.isfinite(costs_arr[i + 1]) and abs(costs_arr[i]) > 1e-12:
                cost_improve_to_next[i] = (costs_arr[i] - costs_arr[i + 1]) / abs(costs_arr[i])

        plateau_k: Optional[int] = None
        for i in range(0, len(k_values) - 1):
            if np.isfinite(cost_improve_to_next[i]) and float(cost_improve_to_next[i]) < 0.05:
                plateau_k = int(k_values[i])
                break

        span = float(max(1, (max_k - effective_min_k)))
        plateau_penalty = np.zeros(len(k_values), dtype=float)
        if plateau_k is not None:
            plateau_penalty = np.where(k_values < plateau_k, (plateau_k - k_values) / span, 0.0)
        prior_penalty = np.abs(k_values - feature_suggested_k) / span
        elbow_penalty = np.abs(k_values - elbow_k) / span
        complexity_penalty = (k_values - effective_min_k) / span

        small_frac_arr = np.asarray(small_cluster_fracs, dtype=float)
        min_frac_arr = np.asarray(min_cluster_fracs, dtype=float)
        imbalance_arr = np.asarray(imbalance_ratios, dtype=float)
        
     
        min_cluster_count = min_frac_arr * n_samples
        extreme_imbalance_pen = np.where(
            np.isfinite(min_cluster_count) & (min_cluster_count < 3),
            0.5,  
            0.0
        )
        
        tiny_penalty = extreme_imbalance_pen

        if np.isfinite(quality).any():
            base = quality
        else:
            base = np.zeros(len(k_values), dtype=float)

        score = np.asarray(base, dtype=float)
        score[~np.isfinite(score)] = -np.inf

      
        w_prior, w_elbow, w_complexity, w_tiny = 0.7, 0.15, 0.05, 0.1
        score -= w_prior * prior_penalty  
        score -= w_elbow * elbow_penalty
        score -= w_complexity * complexity_penalty
        score -= w_tiny * tiny_penalty 
        if plateau_k is not None:
            score -= 0.1 * plateau_penalty  

        min_cluster_count = min_frac_arr * n_samples
        balanced_mask = np.isfinite(min_cluster_count) & (min_cluster_count >= 3)
        
        if np.isfinite(quality).any():
            candidate_mask = valid_cost & np.isfinite(quality) & balanced_mask
            if not bool(candidate_mask.any()):
                candidate_mask = valid_cost & np.isfinite(quality)
            if not bool(candidate_mask.any()):
                candidate_mask = valid_cost
        else:
            candidate_mask = valid_cost & balanced_mask
            if not bool(candidate_mask.any()):
                candidate_mask = valid_cost

        score_masked = np.where(candidate_mask, score, -np.inf)
        best_idx = int(np.argmax(score_masked))
        optimal_k = int(k_values[best_idx])
        
        print(f"\n  Comprehensive decision:")
        print(f"    Elbow method suggests k={elbow_k}")
        print(f"    Field features suggest k={feature_suggested_k}")
        
        valid_indices = np.where(candidate_mask)[0]
        if len(valid_indices) > 0:
            valid_scores = score_masked[valid_indices]
            valid_ks = k_values[valid_indices]
            top3_idx = np.argsort(valid_scores)[-3:][::-1]  
            print(f"    Top 3 candidate k scores:")
            for idx in top3_idx:
                k_val = int(valid_ks[idx])
                score_val = float(valid_scores[idx])
                i = int(np.where(k_values == k_val)[0][0])
                min_frac = float(min_cluster_fracs[i]) if i < len(min_cluster_fracs) else np.nan
                print(f"      k={k_val}: score={score_val:.3f}, min_cluster={min_frac*100:.1f}%")
        
        if any(not np.isnan(s) for s in silhouettes):
            valid = [(k, s) for k, s in zip(K_range, silhouettes) if not np.isnan(s)]
            best_sil_k, best_sil = max(valid, key=lambda x: x[1])
            # print(f"    Silhouette  k={best_sil_k} (score={best_sil:.3f})")
        if any(not np.isnan(ch) for ch in calinskis):
            valid = [(k, ch) for k, ch in zip(K_range, calinskis) if not np.isnan(ch)]
            best_ch_k, best_ch = max(valid, key=lambda x: x[1])
            # print(f"    Calinski-Harabasz  k={best_ch_k} (score={best_ch:.1f})")
        if any(not np.isnan(db) for db in davies):
            valid = [(k, db) for k, db in zip(K_range, davies) if not np.isnan(db)]
            best_db_k, best_db = min(valid, key=lambda x: x[1])
            # print(f"    Davies-Bouldin  k={best_db_k} (score={best_db:.3f})")

        # Cluster size balance for chosen k
        try:
            chosen_min_frac = float(min_cluster_fracs[best_idx])
            chosen_small_frac = float(small_cluster_fracs[best_idx])
            parts = []
            if np.isfinite(chosen_min_frac):
                parts.append(f"min_cluster={chosen_min_frac*100:.1f}%")
            if np.isfinite(chosen_small_frac):
                parts.append(f"small_clusters={chosen_small_frac*100:.1f}%")
            if parts:
                print(f"    Selected k={optimal_k} ({', '.join(parts)})")
        except Exception:
            pass
        
        print(f"\n[OK] Optimal cluster count: {optimal_k}")
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            score_out = np.asarray(score, dtype=float).copy()
            score_out[~np.isfinite(score_out)] = np.nan
            metrics_df = pd.DataFrame({
                'k': list(K_range),
                'cost': costs,
                'silhouette_continuous': silhouettes,
                'calinski_harabasz_continuous': calinskis,
                'davies_bouldin_continuous': davies,
                'cost_improvement': cost_improve,
                'cost_improvement_to_next': cost_improve_to_next,
                'min_cluster_frac': min_cluster_fracs,
                'small_cluster_frac': small_cluster_fracs,
                'imbalance_ratio': imbalance_ratios,
                'quality_score': quality,
                'selection_score': score_out,
            })
            # Use UTF-8 with BOM so Excel/Windows tools don't mis-decode smart quotes 
            metrics_df.to_csv(
                os.path.join(output_dir, 'cluster_selection_metrics.csv'),
                index=False,
                encoding='utf-8-sig',
            )
            self._plot_elbow_curve(K_range, costs, optimal_k, output_dir, silhouettes=silhouettes)
        
        return optimal_k
    
    def _find_elbow(self, K_range: range, costs: List[float]) -> int:
        """Find elbow point in cost curve"""

        valid_data = [
            (int(k), float(c))
            for k, c in zip(K_range, costs)
            if c is not None and not np.isnan(c)
        ]

        if not valid_data:
            raise ValueError("Cannot select elbow point: all k costs are NaN")
        if len(valid_data) == 1:
            return valid_data[0][0]
        if len(valid_data) == 2:
            (k0, c0), (k1, c1) = valid_data
            denom = abs(c0) if c0 != 0 else 1.0
            improvement = (c0 - c1) / denom

            return k1 if improvement >= 0.05 else k0

        ks = np.array([k for k, _ in valid_data], dtype=float)
        cs = np.array([c for _, c in valid_data], dtype=float)


        if float(cs.max() - cs.min()) == 0.0:
            return int(ks[0])


        x = (ks - ks.min()) / (ks.max() - ks.min()) if ks.max() != ks.min() else np.zeros_like(ks)
        y = (cs - cs.min()) / (cs.max() - cs.min())

        p0 = np.array([x[0], y[0]], dtype=float)
        p1 = np.array([x[-1], y[-1]], dtype=float)
        line = p1 - p0
        norm = float(np.linalg.norm(line))
        if norm == 0.0:
            return int(ks[len(ks) // 2])

        pts = np.stack([x, y], axis=1)
        p0p = pts - p0

        dist = np.abs(line[0] * p0p[:, 1] - line[1] * p0p[:, 0]) / norm
        dist[0] = -1.0
        dist[-1] = -1.0

        best_idx = int(np.argmax(dist))
        return int(ks[best_idx])
    
    def _fit_kprototypes(
        self,
        X: np.ndarray,
        categorical_indices: List[int],
        n_clusters: int
    ) -> Tuple[np.ndarray, Any]:
        """Fit K-Prototypes model"""
        print("\nFitting K-Prototypes model...")

        kproto = None
        last_err: Optional[Exception] = None
        init_methods = ("Cao", "Huang", "random")
        seed_offsets = (0, 101, 202)
        for attempt, init_method in enumerate(init_methods):
            if kproto is not None:
                break
            for seed_offset in seed_offsets:
                try:
                    kproto = self._make_kprototypes(
                        n_clusters=n_clusters,
                        verbose=0,
                        init=init_method,
                        random_state=self.random_state + 17 * attempt + seed_offset + n_clusters,
                        n_init=10,
                    )
                    labels = kproto.fit_predict(X, categorical=categorical_indices)
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
                    kproto = None

        if kproto is None:
            raise RuntimeError(f"K-Prototypes initialization failed (k={n_clusters}): {last_err}")

        print(f"[OK] 聚类完成。成本: {kproto.cost_:.2f}")
        
        # 打印聚类大小
        unique, counts = np.unique(labels, return_counts=True)
        print("\n聚类大小:")
        for cluster_id, count in zip(unique, counts):
            print(f"  聚类 {cluster_id}: {count} 个家庭 ({count/len(labels)*100:.1f}%)")
        
        return labels, kproto

    def _make_kprototypes(
        self,
        n_clusters: int,
        verbose: int = 0,
        *,
        init: str = "Cao",
        random_state: Optional[int] = None,
        n_init: int = 5,
    ):
        """Compatible KPrototypes initialization parameters for different kmodes versions."""
        base_kwargs = dict(n_clusters=n_clusters, init=init, verbose=verbose)
        if random_state is None:
            random_state = self.random_state
        if random_state is not None:
            base_kwargs["random_state"] = random_state


        if n_init is not None:
            base_kwargs["n_init"] = int(n_init)

        try:
            return KPrototypes(**base_kwargs, n_jobs=-1)
        except TypeError:
            base_kwargs.pop("n_jobs", None)
            try:
                return KPrototypes(**base_kwargs)
            except TypeError:
                # older kmodes: no n_init
                base_kwargs.pop("n_init", None)
                return KPrototypes(**base_kwargs)

    def _make_kmodes(self, n_clusters: int, verbose: int = 0):
        """Compatible KModes initialization parameters for different kmodes versions."""
        base_kwargs = dict(
            n_clusters=n_clusters,
            init="Cao",
            n_init=5,
            verbose=verbose,
            random_state=self.random_state
        )
        try:
            return KModes(**base_kwargs, n_jobs=-1)
        except TypeError:
            return KModes(**base_kwargs)

    def _fit_kmodes(self, X: np.ndarray, n_clusters: int) -> Tuple[np.ndarray, Any]:
        """Use KModes for all categorical features."""
        print("\nFitting K-Modes model...")
        km = self._make_kmodes(n_clusters=n_clusters, verbose=0)
        labels = km.fit_predict(X)
        print(f"[OK] Clustering completed. Cost: {float(getattr(km, 'cost_', np.nan)):.2f}")

        unique, counts = np.unique(labels, return_counts=True)
        print("\nCluster sizes:")
        for cluster_id, count in zip(unique, counts):
            print(f"  Cluster {cluster_id}: {count} households ({count/len(labels)*100:.1f}%)")

        return labels, km

    def _determine_optimal_clusters_kmodes(
        self,
        X: np.ndarray,
        output_dir: str = None
    ) -> int:
        """K-Modes cluster count selection: elbow based on cost."""

        costs: List[float] = []
        K_range = range(self.min_clusters, self.max_clusters + 1)

        for k in K_range:
            try:
                km = self._make_kmodes(n_clusters=k, verbose=0)
                km.fit(X)
                costs.append(float(getattr(km, "cost_", np.nan)))
            except Exception as e:
                costs.append(np.nan)

        optimal_k = self._find_elbow(K_range, costs)
        print(f"\n[OK] Optimal cluster count: {optimal_k}")

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            metrics_df = pd.DataFrame({"k": list(K_range), "cost": costs})
            # Use UTF-8 with BOM so Excel/Windows tools don't mis-decode smart quotes 
            metrics_df.to_csv(
                os.path.join(output_dir, "cluster_selection_metrics.csv"),
                index=False,
                encoding='utf-8-sig',
            )
            self._plot_elbow_curve(K_range, costs, optimal_k, output_dir, silhouettes=None)

        return optimal_k
    
    def _filter_small_clusters(
        self,
        labels: np.ndarray,
        min_cluster_size: int
    ) -> np.ndarray:
        """
        Mark clusters smaller than threshold as noise (-1)
        
        Reason: Small clusters may be unreliable due to missing value imputation, unsuitable for persona generation
        """
        labels = np.asarray(labels).copy()
        
        if labels.dtype.kind == 'u':  
            labels = labels.astype(np.int32)  
        elif labels.dtype.kind != 'i':  
            labels = labels.astype(np.int32)  
        
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        small_clusters = []
        small_cluster_sizes = {}
        for label, count in zip(unique_labels, counts):
            if count < min_cluster_size:
                small_clusters.append(label)
                small_cluster_sizes[label] = int(count)
                labels[labels == label] = -1
        
        if small_clusters:
            n_noise = int(np.sum(labels == -1))
            n_valid_clusters = len(unique_labels) - len(small_clusters)
            print(f"\n[INFO] Marked {len(small_clusters)} small clusters as noise ({n_noise} samples total):")
            for label in small_clusters:
                print(f"  Cluster {label}: {small_cluster_sizes[label]} samples (below threshold {min_cluster_size})")
            print(f"  Total noise samples: {n_noise} ({n_noise/len(labels)*100:.1f}%)")
            print(f"  Valid cluster count: {n_valid_clusters}")
        else:
            print(f"\n[INFO] All cluster sizes >= {min_cluster_size}, no filtering needed")
        
        return labels
    
    def _analyze_cluster_profiles(
        self,
        df: pd.DataFrame,
        labels: np.ndarray,
        continuous_fields: List[str],
        categorical_fields: List[str]
    ) -> pd.DataFrame:
        """Analyze cluster profiles (automatically filters noise points, i.e., samples with label -1)"""
        print("\nAnalyzing cluster profiles...")
        
        df_clustered = df.copy()
        df_clustered['cluster'] = labels
        
        profiles = []
        

        valid_cluster_ids = sorted([cid for cid in df_clustered['cluster'].unique() if cid != -1])
        
        for cluster_id in valid_cluster_ids:
            cluster_data = df_clustered[df_clustered['cluster'] == cluster_id]
            n_samples = len(cluster_data)
            
            profile = {
                'cluster_id': cluster_id,
                'n_samples': n_samples,
                'percentage': n_samples / len(df) * 100
            }
            

            for field in continuous_fields:
                profile[f'{field}_mean'] = cluster_data[field].mean()
                profile[f'{field}_std'] = cluster_data[field].std()
            

            for field in categorical_fields:
                mode_value = cluster_data[field].mode()
                profile[f'{field}_mode'] = mode_value[0] if len(mode_value) > 0 else None
            
            profiles.append(profile)
        
        profiles_df = pd.DataFrame(profiles)
        
        return profiles_df
    
    def _visualize_results(
        self,
        df: pd.DataFrame,
        X_num: Optional[np.ndarray],
        labels: np.ndarray,
        feature_names: List[str],
        continuous_fields: List[str],
        categorical_fields: List[str],
        output_dir: str
    ):
        
        os.makedirs(output_dir, exist_ok=True)
        
        self._plot_pca_visualization(X_num, labels, output_dir)
        
        self._plot_cluster_sizes(labels, output_dir)
        
        if continuous_fields:
            self._plot_feature_distributions(
                df, labels, continuous_fields[:5], output_dir  
            )
        
    
    def _plot_elbow_curve(
        self,
        K_range: range,
        costs: List[float],
        optimal_k: int,
        output_dir: str,
        silhouettes: Optional[List[float]] = None
    ):
        """Plot elbow curve for cluster selection (optionally with silhouette on continuous features)."""
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(list(K_range), costs, 'bo-', linewidth=2, markersize=6, label='Cost')
        ax1.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal k={optimal_k}')
        ax1.set_xlabel('Number of Clusters (k)', fontsize=12)
        ax1.set_ylabel('Cost (within-cluster)', fontsize=12)
        ax1.grid(True, alpha=0.3)

        if silhouettes is not None and any(not np.isnan(s) for s in silhouettes):
            ax2 = ax1.twinx()
            ax2.plot(list(K_range), silhouettes, 'g^-', linewidth=1.5, markersize=5, label='Silhouette (continuous)')
            ax2.set_ylabel('Silhouette (continuous)', fontsize=12, color='g')
            ax2.tick_params(axis='y', labelcolor='g')

            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
        else:
            ax1.legend(loc='best')

        plt.title('K-Prototypes Cluster Selection', fontsize=14, fontweight='bold')
        plt.tight_layout()

        output_path = os.path.join(output_dir, 'elbow_curve.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    def _plot_pca_visualization(self, X_num: Optional[np.ndarray], labels: np.ndarray, output_dir: str):
        """Plot PCA visualization of clusters (continuous features only)."""
        if X_num is None or X_num.shape[1] < 2:
            return

        pca = PCA(n_components=2, random_state=self.random_state)
        X_pca = pca.fit_transform(X_num)
        
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(
            X_pca[:, 0], X_pca[:, 1],
            c=labels, cmap='viridis',
            alpha=0.6, s=50
        )
        plt.colorbar(scatter, label='Cluster')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        plt.title('Household Clusters - PCA Visualization', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, 'pca_visualization.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_cluster_sizes(self, labels: np.ndarray, output_dir: str):
        """Plot cluster size distribution"""
        unique, counts = np.unique(labels, return_counts=True)
        percentages = counts / len(labels) * 100
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(unique, counts, color='steelblue', alpha=0.7)
        
        # Add percentage labels
        for bar, pct in zip(bars, percentages):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{pct:.1f}%',
                    ha='center', va='bottom', fontsize=10)
        
        plt.xlabel('Cluster ID', fontsize=12)
        plt.ylabel('Number of Households', fontsize=12)
        plt.title('Cluster Size Distribution', fontsize=14, fontweight='bold')
        plt.xticks(unique)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, 'cluster_sizes.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_feature_distributions(
        self,
        df: pd.DataFrame,
        labels: np.ndarray,
        features: List[str],
        output_dir: str
    ):
        """Plot feature distributions by cluster"""
        df_plot = df.copy()
        df_plot['cluster'] = labels
        
        n_features = len(features)
        fig, axes = plt.subplots(1, n_features, figsize=(5*n_features, 5))
        
        if n_features == 1:
            axes = [axes]
        
        for idx, feature in enumerate(features):
            for cluster_id in sorted(df_plot['cluster'].unique()):
                cluster_data = df_plot[df_plot['cluster'] == cluster_id][feature]
                axes[idx].hist(cluster_data, alpha=0.5, label=f'Cluster {cluster_id}', bins=20)
            
            axes[idx].set_xlabel(feature, fontsize=10)
            axes[idx].set_ylabel('Frequency', fontsize=10)
            axes[idx].set_title(f'{feature} by Cluster', fontsize=11, fontweight='bold')
            axes[idx].legend()
            axes[idx].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, 'feature_distributions.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_results(self, output_dir: str, df_original: pd.DataFrame = None):
        if self.clustering_results is None:
            return
        
        os.makedirs(output_dir, exist_ok=True)
        

        # labels_df = pd.DataFrame({
        #     'cluster': self.clustering_results['labels']
        # })
        # labels_path = os.path.join(output_dir, 'cluster_labels.csv')
        # labels_df.to_csv(labels_path, index=False)

        

        profiles_path = os.path.join(output_dir, 'cluster_profiles.csv')

        self.clustering_results['cluster_profiles'].to_csv(
            profiles_path,
            index=False,
            encoding='utf-8-sig',
        )
        print(f"[OK] 聚类画像已保存到 {profiles_path}")
        

        # interpretation_path = os.path.join(output_dir, 'cluster_interpretation.txt')
        # with open(interpretation_path, 'w', encoding='utf-8') as f:
        #     f.write(self.clustering_results['interpretation'])

        

        if df_original is not None:
            df_with_clusters = df_original.copy()
            df_with_clusters['cluster'] = self.clustering_results['labels']
            full_data_path = os.path.join(output_dir, 'data_with_clusters.csv')

            df_with_clusters.to_csv(full_data_path, index=False, encoding='utf-8-sig')

# ============================================================================

# ============================================================================

class HouseholdClusteringSystem:
    """
    Main system for orchestrating field selection and clustering agents
    """
    
    def __init__(
        self,
        llm_model,
        output_dir: str = None,
        random_state: int = 42,
        missing_threshold: float = 0.5,
        constant_threshold: float = 0.95
    ):
        """
        Initialize household clustering system
        
        Args:
            llm_model: Language model used by the agent
            output_dir: Directory to save results
            random_state: Random seed
            missing_threshold: Missing value threshold (0-1)
            constant_threshold: Constant threshold (0-1)
        """
        self.llm_model = llm_model
        self.output_dir = output_dir or "./clustering_results"
        self.random_state = random_state
        self.missing_threshold = missing_threshold
        self.constant_threshold = constant_threshold
        self._field_metadata_cache: Dict[str, Dict[str, Dict[str, Any]]] = {}
        

        self.field_selector = FieldSelectionAgent(
            name="FieldSelector",
            model=llm_model,
        )
        
        self.clusterer = ClusteringAgent(random_state=random_state)
        

    def _infer_dataset_key(self, *, data_path: str) -> Optional[str]:
        """Infer dataset key from known project paths (best-effort)."""
        hay = f"{data_path} {self.output_dir}".lower()
        for key in ("psid", "acs", "ces", "shed", "sipp"):
            if re.search(rf"(?:^|[\\\\/._-]){re.escape(key)}(?:$|[\\\\/._-])", hay):
                return key
        return None

    def _load_field_metadata(self, dataset_key: str) -> Dict[str, Dict[str, Any]]:
        """Load `var_id -> metadata` mapping from `data/<dataset>/extracted_fields_by_agent.json`."""
        dataset_key = (dataset_key or "").lower().strip()
        if not dataset_key:
            return {}

        cached = self._field_metadata_cache.get(dataset_key)
        if isinstance(cached, dict):
            return cached

        path = os.path.join("data", dataset_key, "extracted_fields_by_agent.json")
        if not os.path.exists(path):
            self._field_metadata_cache[dataset_key] = {}
            return {}

        payload: Any = None
        for enc in ("utf-8", "utf-8-sig"):
            try:
                with open(path, "r", encoding=enc) as f:
                    payload = json.load(f)
                break
            except Exception:
                continue

        if isinstance(payload, dict) and isinstance(payload.get("selected_fields"), list):
            items: Any = payload["selected_fields"]
        elif isinstance(payload, list):
            items = payload
        else:
            items = []

        out: Dict[str, Dict[str, Any]] = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            key = item.get("var_id") or item.get("field_name") or item.get("variable_name")
            if not isinstance(key, str) or not key.strip():
                continue
            out[key.strip()] = item

        self._field_metadata_cache[dataset_key] = out
        return out

    def _is_identifier_field(
        self,
        *,
        field_name: str,
        df: pd.DataFrame,
        metadata: Optional[Dict[str, Dict[str, Any]]],
    ) -> bool:
        """Return True if the field is an identifier and should not participate in clustering."""
        if not isinstance(field_name, str) or not field_name.strip():
            return False
        if not isinstance(df, pd.DataFrame) or field_name not in df.columns:
            return False

        # 1) Prefer codebook metadata (most reliable)
        if isinstance(metadata, dict):
            meta = metadata.get(field_name)
            if isinstance(meta, dict):
                cat = str(meta.get("category") or "").lower().strip()
                dtype = str(meta.get("data_type") or "").lower().strip()
                if cat == "identifier" or dtype == "identifier":
                    return True

        # 2) Name-based (covers other datasets)
        if re.search(r"(?:^|[_\\-])(id|identifier)(?:$|[_\\-])", field_name, re.IGNORECASE):
            return True

        return False

    def _filter_selected_fields_for_clustering(
        self,
        *,
        df: pd.DataFrame,
        selected_fields: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Dict[str, Any]]],
        force_include_fields: Optional[List[str]],
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Drop identifier fields from selected_fields (keep force-included fields)."""
        force_include = set(force_include_fields or [])
        kept: List[Dict[str, Any]] = []
        removed: List[str] = []

        for item in selected_fields or []:
            if not isinstance(item, dict):
                continue
            name = item.get("field_name")
            if not isinstance(name, str) or not name.strip():
                continue
            name = name.strip()
            if name in force_include:
                kept.append(item)
                continue
            if self._is_identifier_field(field_name=name, df=df, metadata=metadata):
                removed.append(name)
                continue
            kept.append(item)

        # De-dup (preserve order)
        seen: set = set()
        deduped: List[Dict[str, Any]] = []
        for item in kept:
            name = item.get("field_name")
            if not name or name in seen:
                continue
            seen.add(name)
            deduped.append(item)

        return deduped, removed

    def _load_cached_field_selection(
        self,
        df: pd.DataFrame,
        excluded_patterns: Optional[List[str]],
        force_include_fields: Optional[List[str]]
    ) -> Optional[Dict[str, Any]]:
        """
        If `selected_fields.json` exists in the output directory, reuse the field selection results to avoid repeated LLM calls; return None if the file is missing/unavailable.
        """
        selection_path = os.path.join(self.output_dir, 'selected_fields.json')
        if not os.path.exists(selection_path):
            return None

        try:
            with open(selection_path, 'r', encoding='utf-8-sig') as f:
                cached = json.load(f)
        except Exception as e:
            print(
                f"  [WARN] 检测到 {selection_path} 但读取失败，将重新选择字段："
                f"{type(e).__name__}: {e}"
            )
            return None

        if not isinstance(cached, dict):
            return None

        selected_fields_raw = cached.get('selected_fields')
        if not isinstance(selected_fields_raw, list) or not selected_fields_raw:
            return None

        excluded_patterns = excluded_patterns or []
        force_include_fields = force_include_fields or []
        force_include = set(force_include_fields)

        df_cols = set(df.columns)

        normalized: List[Dict[str, Any]] = []
        for item in selected_fields_raw:
            if isinstance(item, str):
                field_name = item.strip()
                field_type = ""
                reasoning = "Loaded from cache"
            elif isinstance(item, dict):
                field_name = item.get('field_name') or item.get('name')
                field_type = str(item.get('type') or '').strip().lower()
                reasoning = item.get('reasoning') or item.get('reason') or "Loaded from cache"
            else:
                continue

            if not isinstance(field_name, str) or not field_name.strip():
                continue
            field_name = field_name.strip()
            if field_name not in df_cols:
                continue


            if field_name not in force_include and any(
                re.match(pattern, field_name, re.IGNORECASE) for pattern in excluded_patterns
            ):
                continue

            if field_type not in {"continuous", "categorical"}:
                field_type = 'continuous' if pd.api.types.is_numeric_dtype(df[field_name]) else 'categorical'

            normalized.append({
                'field_name': field_name,
                'type': field_type,
                'reasoning': reasoning
            })


        existing = {f['field_name'] for f in normalized if isinstance(f, dict) and 'field_name' in f}
        for col in force_include_fields:
            if col in existing or col not in df_cols:
                continue
            normalized.append({
                'field_name': col,
                'type': 'continuous' if pd.api.types.is_numeric_dtype(df[col]) else 'categorical',
                'reasoning': "Force-included"
            })


        seen: set = set()
        deduped: List[Dict[str, Any]] = []
        for item in normalized:
            name = item.get('field_name')
            if not name or name in seen:
                continue
            seen.add(name)
            deduped.append(item)

        if not deduped:
            return None

        cached = dict(cached)
        cached['selected_fields'] = deduped
        cached['statistics'] = self.field_selector._compute_selection_statistics(deduped)
        cached['loaded_from_cache'] = True
        return cached
    
    async def run_clustering_pipeline_async(
        self,
        data_path: str,
        encoding: Optional[str] = None,
        n_clusters: int = None,
        excluded_patterns: List[str] = None,
        force_include_fields: List[str] = None
    ) -> Dict[str, Any]:
        """
        Run the complete clustering pipeline
        
        参数：
            data_path: Input CSV data path
            n_clusters: Number of clusters (None for automatic determination)
            excluded_patterns: Excluded field patterns
            force_include_fields: Always include fields
            
        Returns:
            Dictionary containing all results
        """
        
        df = pd.read_csv(data_path, encoding=encoding) if encoding else pd.read_csv(data_path)
        print(f"[OK] Loaded {len(df)} households, containing {len(df.columns)} fields")

        # Normalize optional args early (avoids None edge-cases)
        excluded_patterns = list(excluded_patterns or [])
        force_include_fields = list(force_include_fields or [])

        # If there is a field info file (prepare_data_for_clustering output), restrict columns to available_fields
        # so we only consider the curated set (e.g., 50 fields) instead of the entire codebook.
        info_path = re.sub(r"\.csv$", "_field_info.json", str(data_path))
        restricted = False
        if os.path.exists(info_path):
            try:
                with open(info_path, "r", encoding="utf-8") as f:
                    info_payload = json.load(f)
                available_fields = info_payload.get("available_fields") or []
                if isinstance(available_fields, list) and available_fields:
                    keep_cols = [c for c in available_fields if c in df.columns]
                    if keep_cols:
                        df = df[keep_cols].copy()
                        restricted = True
            except Exception as e:
                print(f"[WARN] Failed to read field info file, continuing with all columns: {type(e).__name__}: {e}")

        # If no field_info.json was used, try to restrict to extracted_fields_by_agent.json for the dataset.
        if not restricted:
            try:
                dataset_key = self._infer_dataset_key(data_path=data_path)
                default_fields_path = {
                    "psid": "data/psid/extracted_fields_by_agent.json",
                    "acs": "data/acs/extracted_fields_by_agent.json",
                    "ces": "data/ces/extracted_fields_by_agent.json",
                    "shed": "data/shed/extracted_fields_by_agent.json",
                    "sipp": "data/sipp/extracted_fields_by_agent.json",
                }.get(dataset_key or "", "")
                if default_fields_path and os.path.exists(default_fields_path):
                    with open(default_fields_path, "r", encoding="utf-8") as f:
                        payload = json.load(f)
                    if isinstance(payload, list):
                        extracted_fields = payload
                    elif isinstance(payload, dict) and isinstance(payload.get("selected_fields"), list):
                        extracted_fields = payload["selected_fields"]
                    else:
                        extracted_fields = []
                    requested = []
                    for field in extracted_fields:
                        if not isinstance(field, dict):
                            continue
                        col = field.get("var_id") or field.get("variable_name")
                        if isinstance(col, str) and col.strip():
                            requested.append(col.strip())
                    if requested:
                        keep_cols = [c for c in requested if c in df.columns]
                        if keep_cols:
                            df = df[keep_cols].copy()
            except Exception as e:
                print(f"[WARN] Failed to load extracted field list, continuing with all columns: {type(e).__name__}: {e}")

        # Exclude identifier fields from LLM prompting / analysis to reduce context and prevent selection.
        dataset_key = self._infer_dataset_key(data_path=data_path)
        metadata = self._load_field_metadata(dataset_key) if dataset_key else {}
        identifier_fields: List[str] = []
        force_include_set = set(force_include_fields)
        for col in list(df.columns):
            if col in force_include_set:
                continue
            if self._is_identifier_field(field_name=col, df=df, metadata=metadata):
                identifier_fields.append(col)
        if identifier_fields:
            preview = identifier_fields[:10]
            suffix = "..." if len(identifier_fields) > 10 else ""
            print(f"[WARN] Excluding identifier fields from selection candidates: {preview}{suffix}")
            identifier_patterns = [rf"^{re.escape(name)}$" for name in identifier_fields]
            excluded_patterns = [*excluded_patterns, *identifier_patterns]
        

        df, cleaning_report = self._basic_cleaning(
            df,
            excluded_patterns=excluded_patterns,
            force_include_fields=force_include_fields
        )

        field_selection_results = self._load_cached_field_selection(
            df,
            excluded_patterns=excluded_patterns,
            force_include_fields=force_include_fields,
        )
        if field_selection_results is not None and field_selection_results.get('loaded_from_cache'):
            print(f"[OK] Detected clustering field configuration, skipping field selection: {os.path.join(self.output_dir, 'selected_fields.json')}")
        else:
            field_selection_results = await self.field_selector.analyze_fields_async(
                df,
                excluded_patterns=excluded_patterns,
                force_include=force_include_fields
            )
        
        # Ensure output JSON does not include identifier fields in the analysis payload (especially when loaded from cache).
        if identifier_fields and isinstance(field_selection_results, dict):
            if isinstance(field_selection_results.get("all_fields_analysis"), list):
                field_selection_results["all_fields_analysis"] = [
                    f
                    for f in field_selection_results["all_fields_analysis"]
                    if not (isinstance(f, dict) and f.get("field_name") in identifier_fields)
                ]

        selected_fields = field_selection_results['selected_fields']

        # Never allow identifier fields to participate in clustering.
        selected_fields, removed_identifier_fields = self._filter_selected_fields_for_clustering(
            df=df,
            selected_fields=selected_fields,
            metadata=metadata,
            force_include_fields=force_include_fields,
        )
        if removed_identifier_fields:
            preview = removed_identifier_fields[:10]
            suffix = "..." if len(removed_identifier_fields) > 10 else ""
            print(f"[WARN] Excluding identifier fields from clustering: {preview}{suffix}")
            existing_filtered = field_selection_results.get("filtered_out_identifier_fields")
            if not isinstance(existing_filtered, list):
                existing_filtered = []
            field_selection_results["filtered_out_identifier_fields"] = sorted(
                {*(str(x) for x in existing_filtered if x), *removed_identifier_fields}
            )
            field_selection_results['selected_fields'] = selected_fields
            field_selection_results['statistics'] = self.field_selector._compute_selection_statistics(selected_fields)

        stats = field_selection_results['statistics']
    
        

        continuous_fields = [f['field_name'] for f in selected_fields if f['type'] == 'continuous']
        categorical_fields = [f['field_name'] for f in selected_fields if f['type'] == 'categorical']
        if not continuous_fields and not categorical_fields:
            raise ValueError("No usable clustering fields after filtering identifiers/exclusions.")
    
        
        clustering_results = self.clusterer.perform_clustering(
            df,
            continuous_fields,
            categorical_fields,
            n_clusters=n_clusters,
            output_dir=self.output_dir
        )
        

        self.clusterer.save_results(self.output_dir, df)
        

        if not field_selection_results.get('loaded_from_cache'):
            selection_path = os.path.join(self.output_dir, 'selected_fields.json')
            with open(selection_path, 'w', encoding='utf-8') as f:
                json.dump(field_selection_results, f, indent=2, default=str)
            print(f"[OK] Field selection results saved to {selection_path}")
        else:
            print(f"[OK] Field selection results loaded from cache, skipping save")
        
        return {
            'field_selection': field_selection_results,
            'clustering': clustering_results,
            'output_dir': self.output_dir,
            'cleaning_report': cleaning_report
        }

    def run_clustering_pipeline(
        self,
        data_path: str,
        encoding: Optional[str] = None,
        n_clusters: int = None,
        excluded_patterns: List[str] = None,
        force_include_fields: List[str] = None
    ) -> Dict[str, Any]:
        """Synchronous wrapper: can be called directly when no event loop; use await run_clustering_pipeline_async() for event loop."""
        return _run_coro_sync(
            self.run_clustering_pipeline_async(
                data_path=data_path,
                encoding=encoding,
                n_clusters=n_clusters,
                excluded_patterns=excluded_patterns,
                force_include_fields=force_include_fields
            )
        )

    def _basic_cleaning(
        self,
        df: pd.DataFrame,
        excluded_patterns: Optional[List[str]],
        force_include_fields: Optional[List[str]]
    ) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
        """Remove high-missing/near-constant fields (keep forced fields)."""
        excluded_patterns = excluded_patterns or []
        force_include_fields = force_include_fields or []
        force_include = set(force_include_fields)

        removed_missing: List[str] = []
        removed_constant: List[str] = []


        missing_rate = df.isna().mean()
        for col, rate in missing_rate.items():
            if col in force_include:
                continue
            if rate > self.missing_threshold:
                removed_missing.append(col)
        df = df.drop(columns=removed_missing, errors='ignore')


        for col in list(df.columns):
            if col in force_include:
                continue
            if any(re.match(p, col, re.IGNORECASE) for p in excluded_patterns):
                continue
            nonnull = df[col].dropna()
            if nonnull.empty:
                removed_constant.append(col)
                continue
            top_ratio = float(nonnull.value_counts(normalize=True, dropna=True).iloc[0])
            if top_ratio >= self.constant_threshold:
                removed_constant.append(col)

        df = df.drop(columns=removed_constant, errors='ignore')

        return df, {
            'removed_missing': removed_missing,
            'removed_constant': removed_constant
        }


# ============================================================================

# ============================================================================


DATASET_CONFIG = {
    'psid': {
        'name': 'PSID (Panel Study of Income Dynamics)',
        'data_file': 'data/psid/2023.csv',
        'output_dir': './result/psid_clustering_output',
        'excluded_patterns': [r'^.*_id$', r'^interview_'],
    },
    'acs': {
        'name': 'ACS (American Community Survey)',
        'data_file': 'data/acs/acs2023_household_level.csv',
        'output_dir': './result/acs_clustering_output',
        'excluded_patterns': [r'^.*_id$', r'^serial$'],
    },
    'ces': {
        'name': 'CES (Consumer Expenditure Survey)',
        'data_file': 'data/ces/fmli232.csv',
        'output_dir': './result/ces_clustering_output',
        'excluded_patterns': [r'^.*_id$', r'^newid$'],
    },
    'shed': {
        'name': 'SHED (Survey of Household Economics and Decisionmaking)',
        'data_file': 'data/shed/public2023.csv',
        'output_dir': './result/shed_clustering_output',
        'excluded_patterns': [r'^.*_id$', r'^caseid$'],

        'encoding': 'utf-8-sig',
    },
    'sipp': {
        'name': 'SIPP (Survey of Income and Program Participation)',
        'data_file': 'data/sipp/public2023.csv',
        'output_dir': './result/sipp_clustering_output',
        'excluded_patterns': [r'^.*_id$', r'^ssuid$', r'^pnum$'],
    }
}


def main(dataset: str = 'psid', n_clusters: int = None, llm_model=None):
    """
    Main function for household clustering system
    
    Args:
        dataset: Dataset name ('psid', 'acs', 'ces', 'shed', 'sipp')
        n_clusters: Number of clusters (None means auto-determine)
        llm_model: LLM model instance (if None, will create default one)
    """
    # Validate dataset name
    if dataset not in DATASET_CONFIG:
        raise ValueError(
            f"Unknown dataset: {dataset}. "
            f"Available datasets: {', '.join(DATASET_CONFIG.keys())}"
        )
    
    config = DATASET_CONFIG[dataset]
    
    print(f"\n{'='*80}")
    print(f"Starting to process dataset: {config['name']}")
    print(f"{'='*80}\n")
    
    # Configure LLM (use AgentScope OpenAIChatModel) if not provided
    if llm_model is None:
        from llm_config import create_llm_model
        llm_model = create_llm_model(temperature=0.7, max_tokens=3000)
    
    # Initialize clustering system
    system = HouseholdClusteringSystem(
        llm_model=llm_model,
        output_dir=config['output_dir'],
        random_state=42
    )
    
    # Check field selection cache (avoid accidentally triggering LLM to re-select fields)
    selection_path = os.path.join(config['output_dir'], 'selected_fields.json')
    if os.path.exists(selection_path):
        print(f"[OK] Detected field selection result cache: {selection_path} (will skip field selection)")
    else:
        print(f"[WARN] Field selection result cache not found: {selection_path} (will re-select fields and save)")

    # Check if data file exists (clustering still needs data table)
    data_path = config['data_file']
    if not os.path.exists(data_path):
        print(f"Error: Data file does not exist: {data_path}")
        print("Please run `prepare_data_for_clustering.py` first to generate the corresponding dataset extraction file.")
        return None
    
    # Run clustering pipeline
    try:
        results = system.run_clustering_pipeline(
            data_path=data_path,
            encoding=config.get('encoding'),
            n_clusters=n_clusters,  # None means auto-determine
            excluded_patterns=config['excluded_patterns'],
            force_include_fields=[]
        )
        
        print(f"\n{'='*80}")
        print(f"[OK] {config['name']} clustering pipeline completed successfully!")
        print(f"{'='*80}")
        print(f"\nResults saved to: {config['output_dir']}")
        print(f"  - Cluster labels: cluster_labels.csv")
        print(f"  - Cluster profiles: cluster_profiles.csv")
        # print(f"  - Cluster interpretation: cluster_interpretation.txt")
        print(f"  - Complete data: data_with_clusters.csv")
        print(f"  - Visualization charts: *.png")
        
        return results
        
    except Exception as e:
        print(f"\nError: Clustering pipeline failed")
        print(f"Error message: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

