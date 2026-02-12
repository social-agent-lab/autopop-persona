"""
Persona iterative validation and refinement engine (Iterative Persona Refinement Engine)

Core mechanism (inspired by algorithm idea):
1. Problem Sample Bank (problem sample bank): Collect households that do not fit the current Persona
2. Reward Function (reward function): Evaluate Persona quality
3. Dynamic Generation (dynamic generation): Generate new Persona assumptions based on problem samples
4. Replace Mechanism (replace mechanism): Keep Persona with high reward, replace Persona with low reward

Workflow:
- Initialize: Generate initial Persona collection
- Evaluate: Calculate the fit score for each household for each Persona
- Identify problems: Collect households with low fit scores
- Generate new assumptions: Generate new Persona assumptions based on problem samples
- Update reward: Recalculate the reward for all Personas
- Iterate: Repeat until convergence
"""

import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple
from agentscope.agent import AgentBase
from agentscope.memory import InMemoryMemory
from agentscope.message import Msg
import numpy as np
import pandas as pd

from persona_generation import PersonaGeneratorAgent
from question_design import HouseholdQuestionEvaluator, HouseholdSurveyValidator
from questionnaire_ground_truth_loader import enrich_with_questionnaire_source_fields

try:
    # Reuse the same dataset config + clustering implementation as the initial pipeline.
    from household_clustering_agent import (  # type: ignore
        ClusteringAgent,
        DATASET_CONFIG as CLUSTERING_DATASET_CONFIG,
    )
except Exception:
    ClusteringAgent = None  # type: ignore[assignment]
    CLUSTERING_DATASET_CONFIG = {}  # type: ignore[assignment]

class PersonaQualityTracker:
    """
    Persona quality tracker
    """
    
    def __init__(
        self,
        persona_id: int,
        persona_data: Dict,
        fit_score: float = 0.0,
        num_assigned: int = 0,
        reward: float = 0.0,
        assigned_households: List[int] = None
    ):
        """
        Initialize Persona quality tracker
        
        """
        self.persona_id = persona_id
        self.persona_data = persona_data
        self.fit_score = fit_score
        self.num_assigned = num_assigned
        self.reward = reward
        if assigned_households is None:
            self.assigned_households = set()
        elif isinstance(assigned_households, list):
            self.assigned_households = set(assigned_households)
        else:
            self.assigned_households = assigned_households
    
    def update_fit_score(self, new_fit: float, household_idx: int):
        """Update the fit score"""
        if household_idx not in self.assigned_households:
            # Weighted average update
            self.fit_score = (self.fit_score * self.num_assigned + new_fit) / (self.num_assigned + 1)
            self.num_assigned += 1
            self.assigned_households.add(household_idx)  # Use add instead of append (set method)
        else:
            # Household already exists, update its fit_score (replace with current value)
            # Recalculate the average: remove old value, add new value
            # Note: Here we simplify the processing, only counting the first addition
            pass
    
    def remove_household(self, household_idx: int):
        """
        Remove a household from this persona (when the household is reassigned to another persona)
        
        Args:
            household_idx: The index of the household to remove
        """
        if household_idx in self.assigned_households:
            self.assigned_households.remove(household_idx)
            
            self.num_assigned = len(self.assigned_households)
           
    
    def compute_reward(self, alpha: float, total_samples: int):
        """
        reward = fit_score + alpha * sqrt(log(total_samples) / num_assigned)
        
       
        """
        if self.num_assigned == 0:
            self.reward = float('inf')      
        else:
            exploration_bonus = alpha * math.sqrt(math.log(total_samples) / self.num_assigned)
            self.reward = self.fit_score + exploration_bonus
        
        return self.reward
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'persona_id': self.persona_id,
            'fit_score': self.fit_score,
            'num_assigned': self.num_assigned,      
            'reward': self.reward,
            'assigned_households_count': len(self.assigned_households)
        }


class HouseholdFitEvaluator(AgentBase):
    """
    Household fit evaluator agent
    Similar to the Inference class in algorithm, evaluate the fit of a household with a Persona
    
    Responsibilities:
    - Evaluate the fit of a single household with each Persona
    - Identify unsuitable households (problem samples)
    - Calculate the fit score
    """
    
    def __init__(
        self,
        name: str,
        model,
        sys_prompt: str = None,
        memory=None,
        **kwargs
    ):
        """Initialize the household fit evaluator agent"""
        # AgentBase.__init__() 不接受任何参数
        super().__init__()
        
        # Manually set all attributes
        self.name = name
        self.model = model
        self.sys_prompt = sys_prompt or self._default_sys_prompt()
        self.memory = memory or InMemoryMemory()
    
    def _default_sys_prompt(self) -> str:
        """Default system prompt"""
        return """You are an expert in household segmentation and persona matching, skilled at evaluating how well a household fits into a persona profile.

Your tasks are:
(1) Carefully compare household characteristics with persona profiles
(2) Assign a fit score from 0 to 1 (0=completely mismatched, 1=perfect fit)
(3) Identify specific mismatches or inconsistencies
(4) Be objective and evidence based in your evaluation

Requirements:
* Consider multiple dimensions: demographics, financials, behaviors, lifestyle
* Provide specific evidence for your scoring
* Highlight the most important matching or mismatching factors
* Be consistent across evaluations
"""
    
    async def _call_model_async(self, messages: List[Dict]) -> str:
        """ Async call to the LLM model (continue retrying until success
        Args:
            messages: Message list
        Returns:
            str: Model response content (ensured success)
        """
        import asyncio
        
        attempt = 0
        while True:  
            try:
                response = await self.model(messages)
                
                if hasattr(response, '__aiter__'):
                    content = ""
                    last_chunk = None
                    async for chunk in response:
                        last_chunk = chunk
                    
                    if last_chunk:
                        if isinstance(last_chunk, dict):
                            text_value = last_chunk.get('text', last_chunk.get('content', ''))
                            if isinstance(text_value, str):
                                content = text_value
                            elif isinstance(text_value, list):
                                for item in text_value:
                                    if isinstance(item, dict) and item.get('type') == 'text':
                                        content += item.get('text', '')
                                    else:
                                        content += str(item)
                            else:
                                content = str(text_value)
                        elif hasattr(last_chunk, 'text'):
                            content = str(last_chunk.text)
                        elif isinstance(last_chunk, str):
                            content = last_chunk
                        else:
                            content = str(last_chunk)

                    if content.strip():
                        return content
                    else:
                        raise ValueError("Empty response content")
            
                elif hasattr(response, 'text'):
                    content = response.text
                elif isinstance(response, dict):
                    if 'choices' in response:
                        content = response['choices'][0]['message']['content']
                    elif 'content' in response:
                        # AgentScope ChatResponse with 'content' key
                        text_value = response['content']
                        if isinstance(text_value, str):
                            content = text_value
                        elif isinstance(text_value, list):
                            content = ""
                            for item in text_value:
                                if isinstance(item, dict) and item.get('type') == 'text':
                                    content += item.get('text', '')
                                else:
                                    content += str(item)
                        else:
                            content = str(text_value)
                    elif 'text' in response:
                        content = response['text']
                    else:
                        content = str(response)
                elif isinstance(response, str):
                    content = response
                else:
                    content = str(response)
                
                if content.strip():
                    return content
                else:
                    raise ValueError("Empty response content")
                
            except Exception as e:
                attempt += 1
                wait_time = min(2 ** min(attempt - 1, 6), 60)
                print(f"  ⚠️ LLM call failed (attempt {attempt}): {e}")
                print(f"  Waiting {wait_time} seconds before retrying...")
                await asyncio.sleep(wait_time)
    
    async def evaluate_household_fit(
        self,
        household: pd.Series,
        persona: Dict
    ) -> Tuple[float, str]:
        """
        Evaluate the fit of a household with a Persona
        
        Args:
            household: Household data
            persona: Persona data
            
        Returns:
            Tuple[float, str]: (fit score, evaluation reason)
        """
        # Build the evaluation prompt
        prompt = self._build_fit_evaluation_prompt(household, persona)
        
        # Build the message sequence and call the LLM
        messages = [
            {"role": "system", "content": self.sys_prompt},
            {"role": "user", "content": prompt}
        ]
        
        content = await self._call_model_async(messages)
        
        # Parse the response
        fit_score, reason = self._parse_fit_response(content)
        
        return fit_score, reason
    
    def _build_fit_evaluation_prompt(self, household: pd.Series, persona: Dict) -> str:
        """Build the fit evaluation prompt"""
        def _to_float(val: Any) -> Optional[float]:
            if pd.isna(val):
                return None
            if isinstance(val, bool):
                return float(int(val))
            if isinstance(val, (int, float, np.integer, np.floating)):
                v = float(val)
                return v if math.isfinite(v) else None
            s = str(val).strip()
            if not s:
                return None
            try:
                v = float(s)
                return v if math.isfinite(v) else None
            except Exception:
                return None

        household_key_by_lower = {str(k).strip().lower(): k for k in household.index}

        def _is_missing(val: Any) -> bool:
            if val is None:
                return True
            try:
                return bool(pd.isna(val))
            except Exception:
                return False

        def _get_household_field(field_name: str) -> Tuple[Optional[str], Any]:
            if not field_name:
                return None, None
            key = household_key_by_lower.get(str(field_name).strip().lower())
            if key is None:
                return None, None
            val = household.get(key)
            if _is_missing(val):
                return None, None
            return str(key), val

        def _is_money_category(category: Optional[str]) -> bool:
            if not category:
                return False
            c = str(category).strip().lower()
            if not c:
                return False
            if c in {"income", "assets", "wealth", "debt", "savings", "net_worth"}:
                return True
            tokens = ("income", "asset", "wealth", "debt", "saving", "net worth", "property", "rent", "mortgage", "cost")
            return any(tok in c for tok in tokens)

        def _format_value(val: Any, category: Optional[str] = None) -> str:
            if _is_missing(val):
                return "N/A"
            if isinstance(val, str):
                s = val.strip()
                return s if s else "N/A"
            if isinstance(val, bool):
                return "1" if val else "0"

            money_hint = _is_money_category(category)
            num = _to_float(val)
            if num is not None:
                if money_hint:
                    return f"${num:,.0f}"
                if abs(num) >= 1000 and float(num).is_integer():
                    return f"{int(num):,}"
                txt = f"{num:.3f}".rstrip("0").rstrip(".")
                return txt if txt else "0"

            s = str(val).strip()
            return s if s else "N/A"

        def _format_metrics(metrics: Any, category: Optional[str]) -> str:
            if not isinstance(metrics, dict) or not metrics:
                return ""
            if "mode" in metrics:
                return f"mode={_format_value(metrics.get('mode'), category)}"

            parts: List[str] = []
            if "mean" in metrics:
                parts.append(f"mean={_format_value(metrics.get('mean'), category)}")
            if "std" in metrics:
                parts.append(f"std={_format_value(metrics.get('std'), category)}")
            if parts:
                return ", ".join(parts)

            for k, v in list(metrics.items())[:2]:
                parts.append(f"{k}={_format_value(v, category)}")
            return ", ".join(parts)

        def _load_extracted_fields_by_agent(dataset_key: Optional[str]) -> List[Dict[str, Any]]:
            if not dataset_key:
                return []
            cache = getattr(self, "_extracted_fields_by_agent_cache", None)
            if not isinstance(cache, dict):
                cache = {}
                setattr(self, "_extracted_fields_by_agent_cache", cache)
            if dataset_key in cache:
                return cache[dataset_key]

            folder = "acs" if dataset_key == "acs" else dataset_key
            fields_path = os.path.join("data", folder, "extracted_fields_by_agent.json")
            items: List[Dict[str, Any]] = []
            try:
                if os.path.exists(fields_path):
                    with open(fields_path, "r", encoding="utf-8") as f:
                        payload = json.load(f)
                    if isinstance(payload, dict) and isinstance(payload.get("selected_fields"), list):
                        items = [i for i in payload["selected_fields"] if isinstance(i, dict)]
                    elif isinstance(payload, list):
                        items = [i for i in payload if isinstance(i, dict)]
            except Exception:
                items = []

            cache[dataset_key] = items
            return items

        dataset_key = str(persona.get("dataset") or "").strip().lower() or None
        max_aligned_fields = 200  # Maximum number of aligned fields (statistics_readable) to put as much as possible; set a very high safety limit to prevent失控
        max_supplementary_fields = 10  # Maximum number of supplementary fields (high-importance fields) to prevent prompt from becoming too long
        used_fields_lower: set[str] = set()

        # 1) Priority: Use persona.statistics_readable to dynamically align household fields (field name+short label+statistics)
        aligned_lines: List[str] = []
        stats_readable = persona.get("statistics_readable")
        if isinstance(stats_readable, list):
            for item in stats_readable:
                if not isinstance(item, dict):
                    continue
                field = item.get("field") or item.get("var_id") or item.get("field_name")
                if not isinstance(field, str) or not field.strip():
                    continue
                field = field.strip()

                _, hh_val = _get_household_field(field)
                if hh_val is None:
                    continue

                label = (
                    item.get("short_label")
                    or item.get("short_description")
                    or item.get("concept_name")
                    or item.get("description")
                    or field
                )
                label = str(label).strip() or field
                if len(label) > 80:
                    label = label[:77] + "..."

                category = (item.get("category") or "").strip().lower() or None
                metrics_txt = _format_metrics(item.get("metrics"), category)
                hh_txt = _format_value(hh_val, category)

                if metrics_txt:
                    aligned_lines.append(f"- {label} ({field}): household={hh_txt}, persona={metrics_txt}")
                else:
                    aligned_lines.append(f"- {label} ({field}): household={hh_txt}")
                used_fields_lower.add(field.lower())

                if len(aligned_lines) >= max_aligned_fields:
                    break

        if not aligned_lines:
            stats = persona.get("statistics", {}) or {}
            if isinstance(stats, dict) and stats:
                for key, stat_val in stats.items():
                    if not isinstance(key, str):
                        continue
                    suffix = None
                    if key.endswith("_mean"):
                        suffix = "_mean"
                    elif key.endswith("_mode"):
                        suffix = "_mode"
                    else:
                        continue
                    base = key[: -len(suffix)].strip()
                    if not base or base.lower() in used_fields_lower:
                        continue
                    _, hh_val = _get_household_field(base)
                    if hh_val is None:
                        continue
                    hh_txt = _format_value(hh_val, None)
                    stat_txt = _format_value(stat_val, None)
                    if suffix == "_mean":
                        aligned_lines.append(f"- {base}: household={hh_txt}, persona_mean={stat_txt}")
                    else:
                        aligned_lines.append(f"- {base}: household={hh_txt}, persona_mode={stat_txt}")
                    used_fields_lower.add(base.lower())
                    if len(aligned_lines) >= max_aligned_fields:
                        break

        supplementary_lines: List[str] = []
        extracted_fields = _load_extracted_fields_by_agent(dataset_key)
        candidates: List[Tuple[bool, float, str, str, Optional[str], Any]] = []
        for meta in extracted_fields:
            if not isinstance(meta, dict):
                continue
            var = meta.get("var_id") or meta.get("field_name") or meta.get("variable_name")
            if not isinstance(var, str) or not var.strip():
                continue
            var = var.strip()
            var_lower = var.lower()
            if var_lower in used_fields_lower:
                continue

            category = str(meta.get("category") or "").strip().lower()
            if category == "identifier":
                continue

            _, hh_val = _get_household_field(var)
            if hh_val is None:
                continue

            score_raw = meta.get("composite_score", None)
            try:
                score = float(score_raw) if score_raw is not None else 0.0
            except Exception:
                score = 0.0
            if score <= 0:
                imp_raw = meta.get("importance_score", None)
                usable_raw = meta.get("usable_percentage", None)
                try:
                    imp = float(imp_raw) if imp_raw is not None else 0.0
                except Exception:
                    imp = 0.0
                try:
                    usable = float(usable_raw) if usable_raw is not None else 0.0
                except Exception:
                    usable = 0.0
                score = imp * 10.0 + usable / 10.0

            label = (
                meta.get("short_label")
                or meta.get("short_description")
                or meta.get("concept_name")
                or meta.get("description")
                or meta.get("variable_label")
                or meta.get("label")
                or var
            )
            label = str(label).strip() or var
            if len(label) > 80:
                label = label[:77] + "..."

            is_demo = category == "demographics"
            candidates.append((is_demo, score, var, label, category or None, hh_val))

        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        per_category_cap = 2
        per_category_count: Dict[str, int] = {}
        for is_demo, _, var, label, category, hh_val in candidates:
            if len(supplementary_lines) >= max_supplementary_fields:
                break
            cat_key = (category or "").strip().lower() or "unknown"
            cap = 3 if cat_key == "demographics" else per_category_cap
            if per_category_count.get(cat_key, 0) >= cap:
                continue
            hh_txt = _format_value(hh_val, category)
            supplementary_lines.append(f"- {label} ({var}): {hh_txt}")
            used_fields_lower.add(var.lower())
            per_category_count[cat_key] = per_category_count.get(cat_key, 0) + 1

        household_parts: List[str] = ["HOUSEHOLD EVIDENCE:"]
        if aligned_lines:
            household_parts.append("Aligned fields (household vs persona stats):")
            household_parts.extend(aligned_lines)
        if supplementary_lines:
            household_parts.append("")
            household_parts.append("Supplementary context (high-importance fields):")
            household_parts.extend(supplementary_lines)
        if not aligned_lines and not supplementary_lines:
            household_parts.append("(No usable household fields found; treat as unknown.)")

        household_desc = "\n".join(household_parts)
        
        # Persona description
        persona_desc = f"""
Persona: {persona.get('persona_name', 'Unknown')}
Core Characteristics: {persona.get('core_characteristics', 'N/A')}
Typical Profile: {persona.get('typical_profile', 'N/A')[:300]}...
"""
    
        decision_logic = persona.get('decision_making_logic', {})
        financial_logic = decision_logic.get('financial_decisions', {})
        behavioral_drivers = persona.get('behavioral_drivers', {})
        
        decision_desc = ""
        if financial_logic:
            core_logic = financial_logic.get('core_logic', '')
            priorities = financial_logic.get('priorities', [])
            if core_logic or priorities:
                decision_desc = f"""
Decision Making Logic:
Core Logic: {core_logic}
Priorities: {', '.join(priorities[:2]) if priorities else 'N/A'}
"""
    
        behavioral_desc = ""
        if behavioral_drivers:
            constraints = behavioral_drivers.get('core_constraints', [])
            if constraints:
                behavioral_desc = f"""
Key Constraints: {', '.join(constraints[:2]) if constraints else 'N/A'}
"""
        
        prompt = f"""Evaluate how well the following household fits into the given persona profile.

{household_desc}

{persona_desc}
{decision_desc}{behavioral_desc}

Please provide:
1. A fit score from 0.0 to 1.0 (0=completely mismatched, 1=perfect match)
2. Brief reasoning (2 to 3 sentences) explaining the score

Consider:
* Demographic similarity (age, family structure)
* Financial similarity (income, wealth levels)
* Behavioral similarity (risk appetite, savings patterns)
* Lifestyle alignment
* If some household fields are N/A, treat them as unknown (do NOT assume 0); avoid extreme 0/1 scores unless there is clear evidence.

Output in JSON format:
{{
    "fit_score": 0.75,
    "reasoning": "This household matches well on income and age, but shows lower risk appetite than typical persona members..."
}}
"""
        
        return prompt
    
    def _parse_fit_response(self, response: str) -> Tuple[float, str]:
        try:
            response_clean = response.strip()
            if response_clean.startswith("```json"):
                response_clean = response_clean[7:]
            if response_clean.startswith("```"):
                response_clean = response_clean[3:]
            if response_clean.endswith("```"):
                response_clean = response_clean[:-3]
            
            result = json.loads(response_clean.strip())
            fit_score = float(result.get('fit_score', 0.5))
            reasoning = result.get('reasoning', 'No reasoning provided')
        
            fit_score = max(0.0, min(1.0, fit_score))
            
            return fit_score, reasoning
            
        except (json.JSONDecodeError, ValueError):
            return 0.5, f"Parse error: {response[:100]}"


class IterativeRefinementEngine:
    """
    Iterative optimization engine main class
    Core algorithm loop: evaluate -> collect problem samples -> generate new assumptions -> replace -> repeat
    
    Reference algorithm core logic:
    1. Problem sample bank: collect households with low fit scores
    2. UCB reward: balance fit_score and exploration
    3. Dynamic generation: generate new Persona assumptions based on problem samples
    4. Replace mechanism: keep Persona with high reward, replace Persona with low reward
    """
    
    def __init__(
        self,
        model,
        max_iterations: Optional[int] = 5,
        alpha: float = 0.2,
        fit_threshold: float = 0.6,
        min_problem_rate: float = 0.05,
        stage2_max_outliers: Optional[int] = None,
        max_personas: int = 12,
        dataset: Optional[str] = None,
        output_dir: str = "result",
        random_state: int = 42,
    ):
        """
        Initialize iterative optimization engine
        
        Args:
            model: LLM model instance
            max_iterations: Maximum number of iterations (None/<=0 means until convergence)
            fit_threshold: Fit threshold, below which is considered a problem sample
            min_problem_rate: Problem sample rate threshold 
            stage2_max_outliers: Maximum number of outliers to process in stage2 re-clustering (None means no limit)
            max_personas: Maximum number of Personas (similar to max_num_hypotheses)
        """
        self.model = model
        try:
            if max_iterations is None:
                self.max_iterations = None
            else:
                max_iterations_int = int(max_iterations)
                self.max_iterations = None if max_iterations_int <= 0 else max_iterations_int
        except Exception:
            self.max_iterations = 5
        self.alpha = alpha
        self.fit_threshold = fit_threshold
        try:
            self.min_problem_rate = float(min_problem_rate)
        except Exception:
            self.min_problem_rate = 0.05
        if self.min_problem_rate <= 0:
            raise ValueError("min_problem_rate must be >0 (calculate threshold by rate)")
        try:
            self.stage2_max_outliers = int(stage2_max_outliers) if stage2_max_outliers is not None else None
        except Exception:
            self.stage2_max_outliers = None
        if self.stage2_max_outliers is not None and self.stage2_max_outliers <= 0:
            self.stage2_max_outliers = None
        self.max_personas = max_personas
        self.dataset = (dataset or "").strip().lower() or None
        self.output_dir = output_dir
        self.random_state = random_state
        
        # Agents
        self.fit_evaluator = None
        self.question_evaluator = None  
        self.persona_generator = None  
        
        self.persona_trackers = {}  # key: persona_id, value: PersonaQualityTracker
        
       
        self.cluster_to_persona = {}  # key: cluster_id, value: persona_id
        
       
        self.problem_household_ids = set()
         
        
        self.validated_household_ids = set()

        
        self.pending_validation_household_ids = set()
         
       
        self.persona_validated_households = {}  # key: persona_id, value: set of household_ids


        self.household_initial_cluster_id: Dict[Any, int] = {}
        self.household_initial_persona_id: Dict[Any, int] = {}
        self.household_last_validated_cluster_id: Dict[Any, int] = {}
        self.household_last_validated_persona_id: Dict[Any, int] = {}
        
      
        self.iteration_history = []
    
    def initialize_agents(self):
        """Initialize necessary components"""
        
    
        self.fit_evaluator = HouseholdFitEvaluator(
            name="FitEvaluator",
            model=self.model
        )
        
        self.question_evaluator = HouseholdQuestionEvaluator(
            name="QuestionEvaluator",
            model=self.model
        )

        field_metadata = self._load_field_metadata()
        self.persona_generator = PersonaGeneratorAgent(
            name="PersonaGenerator",
            model=self.model,
            dataset_context={"dataset": self.dataset} if self.dataset else None,
            field_metadata=field_metadata,
        )

    def _load_field_metadata(self) -> Dict[str, Dict[str, Any]]:
        """
        Load var_id -> metadata mapping from extracted_fields_by_agent.json (best-effort).

        This is used to help the Persona generator interpret coded variables (e.g., ER85629_mean).
        """
        dataset = (self.dataset or "").strip().lower()
        if not dataset:
            return {}

        path = os.path.join("data", dataset, "extracted_fields_by_agent.json")
        if not os.path.exists(path):
            return {}

        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            return {}

        if isinstance(payload, dict) and isinstance(payload.get("selected_fields"), list):
            items = payload["selected_fields"]
        elif isinstance(payload, list):
            items = payload
        else:
            return {}

        out: Dict[str, Dict[str, Any]] = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            key = item.get("var_id") or item.get("field_name") or item.get("variable_name")
            if not isinstance(key, str) or not key.strip():
                continue
            out[key.strip()] = item
        return out

    def _resolve_household_id_col(
        self,
        df: pd.DataFrame,
        requested: str = "fid",
        *,
        require_present: bool = False,
    ) -> str:
        """
        Resolve the household unique id column for the active dataset.

        PSID commonly uses `ER82002` (raw) or `fid` (renamed). SHED uses `CaseID`.
        """
        if requested and requested in df.columns:
            return requested

        ds = (self.dataset or "").strip().lower()
        candidates = [requested] if requested else []
        if ds == "psid":
            candidates.extend(["ER82002", "fid"])
        elif ds == "shed":
            candidates.extend(["CaseID", "caseid"])
        elif ds == "acs":
            candidates.extend(["SERIALNO", "serialno"])
        elif ds == "ces":
            candidates.extend(["NEWID", "CUID", "newid", "cuid"])
        elif ds == "sipp":
            candidates.extend(["family_id", "SHHADID", "ssuid", "shhadid"])
        else:
            # Best-effort fallback if dataset isn't provided.
            candidates.extend(
                [
                    "ER82002",
                    "fid",
                    "CaseID",
                    "caseid",
                    "SERIALNO",
                    "serialno",
                    "NEWID",
                    "CUID",
                    "newid",
                    "cuid",
                    "SSUID",
                    "SHHADID",
                    "ssuid",
                    "shhadid",
                ]
            )

        col_by_lower = {str(c).lower(): c for c in df.columns}
        for cand in candidates:
            if not cand:
                continue
            if cand in df.columns:
                return cand
            key = str(cand).lower()
            if key in col_by_lower:
                return str(col_by_lower[key])

        if require_present:
            raise KeyError(
                f"Unable to resolve household id column for dataset={ds!r}; "
                f"requested={requested!r}; candidates={candidates!r}"
            )
        return requested
    

    def _load_selected_fields_for_reclustering(self, cluster_data: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Load selected fields from `selected_fields.json` under `self.output_dir` (same schema as
        `household_clustering_agent.py` output). If missing, fall back to dtype-based inference.
        """
        selected_fields_path = os.path.join(self.output_dir, "selected_fields.json")
        if os.path.exists(selected_fields_path):
            try:
                with open(selected_fields_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                selected = payload.get("selected_fields", []) if isinstance(payload, dict) else []

                continuous_fields = []
                categorical_fields = []
                for item in selected:
                    if not isinstance(item, dict):
                        continue
                    name = item.get("field_name")
                    ftype = str(item.get("type", "")).lower()
                    if not name or name not in cluster_data.columns:
                        continue
                    if ftype == "continuous":
                        continuous_fields.append(name)
                    else:
                        categorical_fields.append(name)

                if continuous_fields or categorical_fields:
                    return continuous_fields, categorical_fields
            except Exception as e:
                print(f"  [WARN] selected_fields.json read failed; fallback field inference: {type(e).__name__}: {e}")

        
        blocked_prefixes = tuple(f"q{i}_" for i in range(1, 11))
        blocked_cols = {"cluster"}
        blocked_cols.update({c for c in cluster_data.columns if any(str(c).startswith(p) for p in blocked_prefixes)})
        candidate_cols = [c for c in cluster_data.columns if c not in blocked_cols]

        numeric_cols = [c for c in candidate_cols if pd.api.types.is_numeric_dtype(cluster_data[c])]
        object_cols = [c for c in candidate_cols if not pd.api.types.is_numeric_dtype(cluster_data[c])]

        categorical_cols = []
        for c in object_cols:
            try:
                nunique = int(cluster_data[c].nunique(dropna=True))
            except Exception:
                continue
            if 2 <= nunique <= 20:
                categorical_cols.append(c)

        return numeric_cols[:20], categorical_cols[:30]

    def initialize_persona_trackers(self, personas: List[Dict], cluster_data: pd.DataFrame):
        """
        Initialize Persona trackers
        
        Key: Establish mapping between cluster ID and Persona
        
        Args:
            personas: Initial Personas list
            cluster_data: Household data with cluster column
        """
        self.persona_trackers = {}
        self.cluster_to_persona = {}  
        
        for i, persona in enumerate(personas):
            
            cluster_id = persona.get('cluster_id', None)
            
            if cluster_id is None:
                stats = persona.get('statistics', {})
                cluster_id = stats.get('cluster_id', i) 
            
            cluster_id = int(cluster_id)
            
            tracker = PersonaQualityTracker(
                persona_id=i,
                persona_data=persona,
                fit_score=0.0,
                num_assigned=0,
                reward=float('inf') 
            )
            self.persona_trackers[i] = tracker
            self.cluster_to_persona[cluster_id] = i 
        
    
        if not self.household_initial_persona_id:
            try:
                clusters = (
                    pd.to_numeric(cluster_data["cluster"], errors="coerce")
                    .fillna(-1)
                    .astype(int)
                )
                personas_by_cluster = clusters.map(self.cluster_to_persona).fillna(-1).astype(int)
                idx_list = cluster_data.index.tolist()
                self.household_initial_cluster_id = dict(zip(idx_list, clusters.tolist()))
                self.household_initial_persona_id = dict(
                    zip(idx_list, personas_by_cluster.tolist())
                )
            except Exception as e:
                print(f"  [WARN] Failed to initialize household initial Persona mapping: {type(e).__name__}: {e}")
    
    def _reassign_noise_households(
        self,
        cluster_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Reassign noise households (small clusters) with the most similar persona
        
        Reason: Small clusters may be unreliable due to missing value填补问题, but should not be completely discarded,
        
        Args:
            cluster_data: Household data with cluster column
            
        Returns:
            pd.DataFrame: Updated cluster_data (cluster column updated)
        """
        noise_mask = cluster_data['cluster'] == -1
        noise_households = cluster_data[noise_mask]
        
        if len(noise_households) == 0:
            return cluster_data  
        
        print(f"\n  [INFO] Found {len(noise_households)} noise households (cluster_id=-1), starting to reassign to the most similar persona...")
        
        cluster_data_updated = cluster_data.copy()
        reassigned_count = 0
        failed_count = 0
        
        for household_idx, household in noise_households.iterrows():

            similar_personas = self._find_similar_personas(
                household=household,
                current_persona_id=-1,  
                top_k=1  
            )
            
            if not similar_personas:
                failed_count += 1
                continue
            
            best_persona_id = similar_personas[0]
            
            best_cluster_id = None
            for cluster_id, persona_id in self.cluster_to_persona.items():
                if persona_id == best_persona_id:
                    best_cluster_id = cluster_id
                    break
            
            if best_cluster_id is not None:
                cluster_data_updated.at[household_idx, 'cluster'] = best_cluster_id
                reassigned_count += 1
            else:
                failed_count += 1
        
        return cluster_data_updated
    
    def _compute_adaptive_sample_sizes_v2(
        self,
        cluster_sizes: Dict[int, int],
        *,
        delta: float = 0.05,
        epsilon: float = 0.05,
        min_per_cluster: int = 3,
        sampling_rate_floor: float = 0.01,
        sampling_rate_ceiling: float = 0.20,
        absolute_max: int = 2000,
    ) -> Dict[int, int]:
        """
        Adaptive sample size calculation V2 - combine precision requirements and cluster size
        
        **Your intuition is correct**: Large clusters should sample more, because:
        1. Large clusters have higher internal diversity, need more samples to cover
        2. Large clusters have a greater impact on global metrics, need more precise estimation
        3. Large clusters may have more edge cases to discover
        
        **But sampling should not grow indefinitely**: Marginal benefits diminish
        - From 30 to 100: Information increases significantly
        - From 1000 to 2000: Information increases limited
        
        **Formula design**:
        ================================================================
        
        **Step 1: Calculate basic precision requirements (Cochran formula + finite population correction)**
        
        Infinite population sampling:
            n_0 = (z² × p × (1-p)) / ε²
        
        Conservative estimate p=0.5:
            n_0 = z² / (4ε²)
        
        For ε=0.05, z=1.96: n_0 ≈ 385
        
        Finite population correction (Finite Population Correction, FPC):
            n_fpc = n_0 / (1 + (n_0 - 1) / N)
        
        When N is large, n_fpc ≈ n_0
        When N is small, n_fpc < n_0 (save samples)
        
        **Step 2: Add diversity coverage term (log scaling)**
        
        Large clusters have higher internal diversity, need extra samples:
            n_diversity = c × ln(N_i)
        
        Where c is the scaling factor (suggested 10-30)
        
        Allocate a portion of the sampling:
            n_prop = N_i × r
        
        Where r is the sampling rate, dynamically calculated:
            r = max(r_floor, min(r_ceiling, r_base / sqrt(N_i)))
        
        This makes:
        - Small cluster (N=100): r ≈ 10-20%
        - Medium cluster (N=1000): r ≈ 3-6%
        - Large cluster (N=10000): r ≈ 1-2%
        
        **Step 4: Final formula**
        
        n_i = max(n_min, min(N_i, n_fpc + n_diversity, n_prop, n_max))
        
        **Key characteristics**:
        - Large cluster sample more ✓
        - Small cluster保证最低采样量 ✓
        - Sampling rate growth rate递减（避免浪费）✓
        - There is theoretical statistical guarantee ✓
        
        Args:
            cluster_sizes: Number of households per cluster {cluster_id: size}
            delta: Global failure probability (default 0.05)
            epsilon: Allowed estimation error (default 0.05, ±5%)
            min_per_cluster: Minimum sampling per cluster (default 3)
            sampling_rate_floor: Minimum sampling rate (default 1%)
            sampling_rate_ceiling: Maximum sampling rate (default 20%)
            absolute_max: Absolute maximum (default 2000, avoid single cluster consuming too many resources)
            
        Returns:
            Dict[int, int]: Sampling per cluster {cluster_id: sample_size}
        """
        if not cluster_sizes:
            return {}
        
        k = len(cluster_sizes)
        eps = max(1e-6, float(epsilon))
        d = max(1e-12, float(delta))

        alpha_per_cluster = d / k
        if alpha_per_cluster >= 0.025:
            z = 1.96
        elif alpha_per_cluster >= 0.005:
            z = 2.576
        elif alpha_per_cluster >= 0.001:
            z = 3.0
        else:
            z = 3.5
        

        n_0 = (z ** 2) / (4 * eps ** 2)
        

        total_households = sum(cluster_sizes.values())
        
        diversity_coefficient = 10.0
        
        sample_sizes = {}
        
        for cluster_id, N_i in cluster_sizes.items():
            N_i = max(1, N_i)

            n_fpc = n_0 / (1.0 + (n_0 - 1.0) / N_i)

            n_diversity = diversity_coefficient * math.log(max(1, N_i))
            

            base_rate = 1.5  # sqrt(1000) ≈ 31.6, 1.5/31.6 ≈ 4.7%
            dynamic_rate = base_rate / math.sqrt(N_i)
            sampling_rate = max(sampling_rate_floor, min(sampling_rate_ceiling, dynamic_rate))
            n_prop = N_i * sampling_rate
            
            n_statistical = n_fpc + n_diversity
            n_candidate = max(n_statistical, n_prop)
            n_candidate = min(n_candidate, N_i * sampling_rate_ceiling)

            n_final = int(math.ceil(n_candidate))
            if N_i <= min_per_cluster:

                n_final = N_i
            else:
                n_final = max(min_per_cluster, n_final)  
                n_final = min(n_final, N_i)              
                n_final = min(n_final, absolute_max)     
            
            sample_sizes[cluster_id] = n_final
        
        return sample_sizes
    
    
    async def run_iteration(
        self,
        personas: List[Dict],
        cluster_data: pd.DataFrame,
        sample_size: bool = True,
        iteration: int = 1
    ) -> Tuple[bool, List[Dict]]:
        """
        Run single iteration - core algorithm loop
        
        Key improvements:
        1. **Validate each cluster separately**：Validate each cluster separately
        2. Collect all outliers, analyze in the end
        3. Cluster outliers, find new meaningful sub-clusters
        4. Generate complete persona for each new sub-cluster (include statistics field)
        
        Args:
            personas: Current Personas
            cluster_data: Household data
            sample_size: Whether to enable sampling evaluation (True=adaptive sampling by cluster number; False=full evaluation)
            iteration: Iteration number
        
        """
        print(f"\n{'='*70}")
        print(f"Iteration {iteration}: Validate Persona fit by cluster")
        print(f"{'='*70}")
        
        if iteration == 1:
            self.initialize_persona_trackers(personas, cluster_data)
        
        cluster_data = self._reassign_noise_households(cluster_data)
        
        print(f"\n  Current Cluster distribution: Cluster to Persona mapping: {self.cluster_to_persona}")
        cluster_counts = cluster_data['cluster'].value_counts().sort_index()
        for cluster_id, count in cluster_counts.items():
            persona_id = self.cluster_to_persona.get(int(cluster_id), None)
            persona_name = self.persona_trackers[persona_id].persona_data.get('persona_name', 'Unknown') if persona_id in self.persona_trackers else 'Unknown'
            print(f"    Cluster {cluster_id}: {count}个家庭 → Persona {persona_id} ({persona_name})")
        
        outlier_dict = {}  # Store outlier households
        
        # Statistics
        total_validated = 0  # Total validated households
        total_outliers = 0   # Total outlier households
        total_inferred_outliers = 0  # Total inferred outlier households (no LLM evaluation)
        
        # Debug: Collect all fit_scores
        all_fit_scores = []
        
        # Initialize questionnaire validator (only add ground truth labels on first iteration)
        # Now there are 10 validation questions (q1_gt to q10_gt)
        gt_cols = tuple(f"q{i}_gt" for i in range(1, 11))
        if iteration == 1 and any(c not in cluster_data.columns for c in gt_cols):
            try:
                # Create validator and add ground truth labels
                resolved_id_col = self._resolve_household_id_col(cluster_data, requested="fid")
                cluster_data = enrich_with_questionnaire_source_fields(
                    cluster_data, dataset=self.dataset, id_col=resolved_id_col
                )
                temp_validator = HouseholdSurveyValidator(
                    cluster_data, id_col=resolved_id_col, dataset=self.dataset
                )
                cluster_data = temp_validator.add_ground_truth_labels()
                print("✓ Questionnaire ground truth labels added")
            except Exception as e:
                print(f"  [WARN] ground truth labels unavailable; questionnaire score will fallback: {type(e).__name__}: {e}")
        
            # ==================== Stage 1: Validate each cluster separately ====================
        print(f"\nStage 1: Validate each cluster separately")
        print(f"  📊 Current Persona total: {len(self.persona_trackers)} (including newly created in previous iterations)")
        
        valid_clusters = sorted(self.cluster_to_persona.keys())
        
        # Calculate the size of each cluster
        cluster_size_map: Dict[int, int] = {}
        for cluster_id in valid_clusters:
            cluster_households = cluster_data[cluster_data['cluster'] == cluster_id]
            cluster_size_map[cluster_id] = len(cluster_households)
        
        # Use adaptive sampling V2 to calculate the sampling size for each cluster
        # V2 method features: Large clusters sample more samples, small clusters ensure minimum sampling
        per_cluster_sample_caps: Optional[Dict[int, int]] = None
        if sample_size:
            total_households = sum(cluster_size_map.values())
            
            # Dynamically adjust epsilon: For small datasets, relax precision requirements to avoid full sampling
            # Large dataset (>1000): epsilon=0.05 (standard precision)
            # Medium dataset (100-1000): epsilon=0.10 (medium precision)
            # Small dataset (<100): epsilon=0.20 (low precision, for debugging)
            if total_households >= 1000:
                adaptive_epsilon = 0.05
            elif total_households >= 100:
                adaptive_epsilon = 0.10
            else:
                adaptive_epsilon = 0.20  # For small datasets, relax precision requirements to ensure sampling
            
            per_cluster_sample_caps = self._compute_adaptive_sample_sizes_v2(
                cluster_sizes=cluster_size_map,
                delta=0.05,                   # Global failure probability
                epsilon=adaptive_epsilon,     # Dynamic precision requirements
                min_per_cluster=3,            # Minimum sampling per cluster (3)
                sampling_rate_floor=0.01,     # Minimum sampling rate (1%)
                sampling_rate_ceiling=0.20,   # Maximum sampling rate (20%)
                absolute_max=2000,            # Absolute maximum (2000, for large datasets like ACS)
            )
            total_budget = sum(per_cluster_sample_caps.values())
            sampling_rate = total_budget / total_households if total_households > 0 else 1.0
           
        
        for cluster_id in valid_clusters:
            persona_id = self.cluster_to_persona[cluster_id]
            
            # Defensive check: Ensure persona_id exists in persona_trackers
            if persona_id not in self.persona_trackers:
                print(f"  ⚠️ Warning: cluster {cluster_id} mapped to persona {persona_id}, but this persona does not exist!")
                print(f"     Current existing persona IDs: {list(self.persona_trackers.keys())}")
                print(f"     Skip validation for this cluster")
                continue
            
            persona = self.persona_trackers[persona_id].persona_data
            persona_name = persona.get('persona_name', f'Persona {persona_id}')
            
            # Get all households in this cluster
            cluster_households = cluster_data[cluster_data['cluster'] == cluster_id]
            
            if len(cluster_households) == 0:
                continue
            
            # Key optimization: Only evaluate households that need to be validated
            # 1. Get the set of households already evaluated for this persona (including validated and marked as outliers)
            if persona_id not in self.persona_validated_households:
                self.persona_validated_households[persona_id] = set()
            
            already_evaluated_in_this_persona = self.persona_validated_households[persona_id]
            
            # 2. Find households that need to be evaluated:
            #    - Newly added households (in cluster but not in the set of already evaluated households for this persona)
            #    - Including households reassigned from other personas (even if globally validated, need to be validated in this persona)
            cluster_household_ids = set(cluster_households.index)
            need_validation_ids = cluster_household_ids - already_evaluated_in_this_persona
            
            # Because households may be reassigned from other personas, need to be validated in the new persona
            
            if len(need_validation_ids) == 0:
                print(f"\n  Validate Cluster {cluster_id} → Persona {persona_id} ({persona_name})")
                print(f"    Cluster size: {len(cluster_households)}, all validated, skip")
                continue
            
            # 4. Get households data that need to be validated
            cluster_sample = cluster_households.loc[list(need_validation_ids)]
            
            current_cluster_cap = None
            if per_cluster_sample_caps is not None:
                current_cluster_cap = per_cluster_sample_caps.get(cluster_id, 50)
            
            if (
                current_cluster_cap is not None
                and current_cluster_cap > 0
                and len(cluster_sample) > current_cluster_cap
            ):
                # Make sampling vary across iterations/clusters while remaining reproducible.
                seed = int(
                    (self.random_state * 1000003 + iteration * 1009 + int(cluster_id)) % (2**32 - 1)
                )
                cluster_sample = cluster_sample.sample(
                    n=current_cluster_cap, replace=False, random_state=seed
                )
                print(
                    f"    🔎 分层采样: {len(cluster_sample)}/{len(need_validation_ids)} "
                    f"(cluster_{cluster_id}_cap={current_cluster_cap})"
                )
            
            already_evaluated_in_cluster = len(cluster_household_ids & already_evaluated_in_this_persona)
            
            need_validation_total = len(need_validation_ids)
            sampled_total = len(cluster_sample)
            cap_note = (
                f", cap={current_cluster_cap}" if current_cluster_cap is not None else ""
            )
            
            if iteration >= 2:
                new_from_other_personas = cluster_household_ids - already_evaluated_in_this_persona - (cluster_household_ids - self.validated_household_ids)
                truly_new = cluster_household_ids - already_evaluated_in_this_persona - self.validated_household_ids
        
            cluster_validated = 0
            cluster_outliers = 0
            cluster_inferred_outliers = 0
            cluster_sample_ids = set(cluster_sample.index)
            did_sample_for_cluster = (
                current_cluster_cap is not None and len(cluster_sample) < len(need_validation_ids)
            )

            for idx, (household_idx, household) in enumerate(cluster_sample.iterrows()):
              
                fit_score, reason = await self.fit_evaluator.evaluate_household_fit(
                    household, persona
                )
                
             
                dataset_lower = (self.dataset or "").strip().lower()
                skip_questionnaire = dataset_lower in ("acs", "ces", "sipp")
                
                if skip_questionnaire:
                    questionnaire_score = None
                    combined_score = fit_score
                else:
                    questionnaire_score = await self._evaluate_questionnaire_score(
                        household, persona
                    )
                    combined_score = 0.5 * fit_score + 0.5 * questionnaire_score
                
                all_fit_scores.append(combined_score)
                
       
                if combined_score >= self.fit_threshold:
                
                    self.persona_trackers[persona_id].update_fit_score(
                        combined_score, household_idx
                    )
                    self.validated_household_ids.add(household_idx)
                    self.persona_validated_households[persona_id].add(household_idx)  # Record to this persona
                    # Record the last time the household was validated through this persona (for final saving when unvalidated samples are reverted)
                    try:
                        self.household_last_validated_persona_id[household_idx] = int(
                            persona_id
                        )
                        self.household_last_validated_cluster_id[household_idx] = int(
                            cluster_id
                        )
                    except Exception:
                        pass
                    # The household has been validated: ensure it is not in the problem sample library
                    self.problem_household_ids.discard(household_idx)
                    # If the household was previously in "pending validation" state, now validated, remove
                    self.pending_validation_household_ids.discard(household_idx)
                    cluster_validated += 1
                    
                else:
                    # Optimization strategy: Determine whether to try other personas based on fit score
                    # If LLM-fit is very low, directly mark as outlier (to avoid being triggered by quest occasional low error)
                    if fit_score < 0.30:
                        # Fit score too low, directly mark as outlier
                        outlier_dict[household_idx] = {
                            'household_idx': household_idx,
                            'household_data': household,
                            'original_cluster': cluster_id,
                            'assigned_persona_id': persona_id,
                            'fit_score': combined_score,
                            'reason': reason
                        }
                        cluster_outliers += 1
                    
                        self.persona_validated_households[persona_id].add(household_idx)
                       
                        self.problem_household_ids.add(household_idx)
                     
                        self.validated_household_ids.discard(household_idx)
                     
                        self.pending_validation_household_ids.discard(household_idx)

                      
                        if did_sample_for_cluster:
                            added = self._infer_similar_outlier_households(
                                seed_household_idx=int(household_idx),
                                seed_household=household,
                                cluster_id=int(cluster_id),
                                persona_id=int(persona_id),
                                cluster_households=cluster_households,
                                exclude_household_ids=cluster_sample_ids,
                                outlier_dict=outlier_dict,
                                already_evaluated_in_this_persona=already_evaluated_in_this_persona,
                                max_additional=5,
                            )
                            if added > 0:
                                cluster_inferred_outliers += added
                                total_inferred_outliers += added

                        
                    else:
                      
                        outlier_dict[household_idx] = {
                            'household_idx': household_idx,
                            'household_data': household,
                            'original_cluster': cluster_id,
                            'assigned_persona_id': persona_id,
                            'fit_score': combined_score,
                            'reason': reason
                        }
                        cluster_outliers += 1
                        
                     
                        candidate_personas = self._find_similar_personas(
                            household, persona_id, top_k=2
                        )
                        
                        best_combined_score = combined_score
                        best_persona_id = persona_id
                        best_reason = reason
                        
                        for other_persona_id in candidate_personas:
                            other_tracker = self.persona_trackers[other_persona_id]
                            other_fit_score, other_reason = await self.fit_evaluator.evaluate_household_fit(
                                household, other_tracker.persona_data
                            )
                            
                            
                            if skip_questionnaire:
                                other_combined_score = other_fit_score
                            else:
                                
                                other_questionnaire_score = await self._evaluate_questionnaire_score(
                                    household, other_tracker.persona_data
                                )
                                other_combined_score = 0.5 * other_fit_score + 0.5 * other_questionnaire_score
                            
                            if other_combined_score > best_combined_score:
                                best_combined_score = other_combined_score
                                best_persona_id = other_persona_id
                                best_reason = other_reason
                        
            
                        if best_combined_score >= self.fit_threshold and best_persona_id != persona_id:

                            del outlier_dict[household_idx]
                            cluster_outliers -= 1
                           
                            self.problem_household_ids.discard(household_idx)
                          
                            self.pending_validation_household_ids.add(household_idx)
                             
                          
                            if household_idx in self.validated_household_ids:
                                self.validated_household_ids.remove(household_idx)
                            
                           
                            self.persona_trackers[persona_id].remove_household(household_idx)
                            
                           
                            if best_persona_id not in self.persona_validated_households:
                                self.persona_validated_households[best_persona_id] = set()
                            self.persona_validated_households[best_persona_id].discard(household_idx)                       
                         
                            self.persona_validated_households[persona_id].add(household_idx)
                            
                     
                            new_cluster_id = None
                            for cid, pid in self.cluster_to_persona.items():
                                if pid == best_persona_id:
                                    new_cluster_id = cid
                                    break
                            if new_cluster_id is not None:
                                cluster_data.loc[household_idx, 'cluster'] = new_cluster_id
                            
                          
                            best_persona_name = self.persona_trackers[best_persona_id].persona_data.get('persona_name', 'Unknown')
                            
                        else:
                        
                            outlier_dict[household_idx] = {
                                'household_idx': household_idx,
                                'household_data': household,
                                'original_cluster': cluster_id,
                                'assigned_persona_id': persona_id,
                                'fit_score': best_combined_score,
                                'reason': best_reason
                            }
                            
                        
                            self.persona_validated_households[persona_id].add(household_idx)
                        
                            self.problem_household_ids.add(household_idx)
                        
                            self.validated_household_ids.discard(household_idx)
                         
                            self.pending_validation_household_ids.discard(household_idx)

                         
                            if did_sample_for_cluster:
                                added = self._infer_similar_outlier_households(
                                    seed_household_idx=int(household_idx),
                                    seed_household=household,
                                    cluster_id=int(cluster_id),
                                    persona_id=int(persona_id),
                                    cluster_households=cluster_households,
                                    exclude_household_ids=cluster_sample_ids,
                                    outlier_dict=outlier_dict,
                                    already_evaluated_in_this_persona=already_evaluated_in_this_persona,
                                    max_additional=5,
                                )
                                if added > 0:
                                    cluster_inferred_outliers += added
                                    total_inferred_outliers += added
                            
            self.persona_trackers[persona_id].compute_reward(
                self.alpha, len(cluster_data)
            )
            
            total_validated += cluster_validated
            total_outliers += cluster_outliers
            
            sample_validation_rate = (
                cluster_validated / len(cluster_sample) if len(cluster_sample) > 0 else 0
            )
        
        # ==================== Stage 2: Recluster outliers and generate new personas ====================

        outlier_indices = sorted(set(outlier_dict.keys()) | set(self.problem_household_ids))

        evaluated_count = len(
            set(self.validated_household_ids)
            | set(self.problem_household_ids)
            | set(self.pending_validation_household_ids)
        )
        stage2_trigger_threshold = int(
            math.ceil(float(self.min_problem_rate) * float(max(0, evaluated_count)))
        )
        stage2_trigger_threshold = max(1, stage2_trigger_threshold)
        
        if len(outlier_indices) >= stage2_trigger_threshold:
            rate_suffix = f", rate={self.min_problem_rate:.4g}, evaluated={evaluated_count}"

            stage2_outlier_indices = outlier_indices
            cap = self.stage2_max_outliers
            if cap is not None and len(outlier_indices) > int(cap):
                cap_int = int(cap)
                seed = int(
                    (int(self.random_state) * 1000003 + int(iteration) * 1009)
                    % (2**32 - 1)
                )
                rng = np.random.default_rng(seed)
                current_iter_outlier_ids = set(outlier_dict.keys())
                current = [hid for hid in outlier_indices if hid in current_iter_outlier_ids]
                rest = [hid for hid in outlier_indices if hid not in current_iter_outlier_ids]

                if len(current) >= cap_int:
                    stage2_outlier_indices = sorted(
                        rng.choice(current, size=cap_int, replace=False).tolist()
                    )
                else:
                    remaining = cap_int - len(current)
                    if len(rest) > remaining:
                        chosen_rest = rng.choice(rest, size=remaining, replace=False).tolist()
                    else:
                        chosen_rest = rest
                    stage2_outlier_indices = sorted(current + chosen_rest)
            
            outlier_households_full = cluster_data.loc[stage2_outlier_indices].copy()
            
            available_slots = max(0, self.max_personas - len(self.persona_trackers))
            if available_slots <= 0:
                print("Reached max_personas limit, skipping outlier re-clustering and generating new Persona")
            else:
                continuous_fields, categorical_fields = self._load_selected_fields_for_reclustering(cluster_data)
                feature_fields = continuous_fields + categorical_fields
                
                created = 0
                
                use_recluster = (
                    ClusteringAgent is not None
                    and len(outlier_households_full) >= 4
                    and len(feature_fields) > 0
                )
                
                if use_recluster:
                    max_k_by_n = max(2, min(10, len(outlier_households_full) - 1))
                    max_k = min(max_k_by_n, max(2, available_slots))
                    
                    outlier_clusterer = ClusteringAgent(
                        max_clusters=max_k,
                        min_clusters=2,
                        random_state=self.random_state,
                    )
                    
                    outlier_output_dir = os.path.join(self.output_dir, f"outlier_reclustering_iter_{iteration}")
                    outlier_features_df = outlier_households_full[feature_fields].copy()
                    
                    clustering_results = outlier_clusterer.perform_clustering(
                        df=outlier_features_df,
                        continuous_fields=continuous_fields,
                        categorical_fields=categorical_fields,
                        n_clusters=None,
                        output_dir=outlier_output_dir,
                    )
                    
                    labels = np.array(clustering_results.get('labels'))
                    profiles = clustering_results.get('cluster_profiles')
                    profiles_df = profiles if isinstance(profiles, pd.DataFrame) else pd.DataFrame(profiles)
                    
                    if 'n_samples' in profiles_df.columns:
                        profiles_df = profiles_df.sort_values('n_samples', ascending=False)
                    
                    for _, row in profiles_df.iterrows():
                        if len(self.persona_trackers) >= self.max_personas:
                            break
                        
                        try:
                            local_cluster_id = int(row.get('cluster_id'))
                        except Exception:
                            continue
                        
                        member_indices = outlier_households_full.index[labels == local_cluster_id].tolist()
                        if not member_indices:
                            continue
                        
                        new_id = max(self.persona_trackers.keys()) + 1
                        new_cluster_id = new_id
                        
                        cluster_stats = row.to_dict()
                        cluster_stats['cluster_id'] = new_cluster_id  
                        
                        sample_households = cluster_data.loc[member_indices].copy()
                        new_persona = await self.persona_generator.generate_persona(cluster_stats, sample_households)
                        
                        tracker = PersonaQualityTracker(
                            persona_id=new_id,
                            persona_data=new_persona,
                            fit_score=0.0,
                            num_assigned=0,
                            reward=float('inf'),
                            assigned_households=[]
                        )
                        self.persona_trackers[new_id] = tracker
                        self.cluster_to_persona[new_cluster_id] = new_id
                        
                        # Reassign outlier households to new cluster, and remove from outlier/validated sets (re-validate in next iteration)
                        cluster_data.loc[member_indices, 'cluster'] = new_cluster_id
                        for hid in member_indices:
                            outlier_dict.pop(hid, None)
                            self.problem_household_ids.discard(hid)
                            self.pending_validation_household_ids.add(hid)
                            if hid in self.validated_household_ids:
                                self.validated_household_ids.remove(hid)
                        
                        created += 1
                else:
                    new_id = max(self.persona_trackers.keys()) + 1
                    new_cluster_id = new_id
                    
                    cluster_stats = {
                        'cluster_id': new_cluster_id,
                        'n_samples': len(outlier_households_full),
                        'percentage': 100.0,
                    }
                    for col in continuous_fields:
                        try:
                            vals = pd.to_numeric(outlier_households_full[col], errors='coerce')
                            cluster_stats[f'{col}_mean'] = float(vals.mean())
                        except Exception:
                            continue
                    for col in categorical_fields:
                        try:
                            s = outlier_households_full[col].fillna('Missing').astype(str)
                            m = s.mode()
                            cluster_stats[f'{col}_mode'] = m.iloc[0] if not m.empty else 'Missing'
                        except Exception:
                            continue
                    
                    new_persona = await self.persona_generator.generate_persona(cluster_stats, outlier_households_full)
                    
                    tracker = PersonaQualityTracker(
                        persona_id=new_id,
                        persona_data=new_persona,
                        fit_score=0.0,
                        num_assigned=0,
                        reward=float('inf'),
                        assigned_households=[]
                    )
                    self.persona_trackers[new_id] = tracker
                    self.cluster_to_persona[new_cluster_id] = new_id
                    
                    cluster_data.loc[stage2_outlier_indices, 'cluster'] = new_cluster_id
                    for hid in stage2_outlier_indices:
                        outlier_dict.pop(hid, None)
                        self.problem_household_ids.discard(hid)
                        self.pending_validation_household_ids.add(hid)
                        if hid in self.validated_household_ids:
                            self.validated_household_ids.remove(hid)
                    created = 1
                
                if created > 0:
                    print(f"Stage 2 completed: New Persona={created}, Reassigned outlier households={len(outlier_households_full)}")
                else:
                    print("Outlier re-clustering did not generate new Persona, keeping outlier status")
        else:
            rate_suffix = f", rate={self.min_problem_rate:.4g}, evaluated={evaluated_count}"
            print(
                f"Stage 2: Outlier households insufficient ({len(outlier_indices)} < {stage2_trigger_threshold}), skipping generation"
                f"{rate_suffix}"
            )
        
        # ==================== Stage 3: Evaluate convergence ====================
        # 1. Calculate average fit_score (based on all personas with data)
        trackers_with_data = [t for t in self.persona_trackers.values() if t.num_assigned > 0]
        avg_fit = np.mean([t.fit_score for t in trackers_with_data]) if trackers_with_data else 0.0
        
        # 2. Global statistics (using union of sets to avoid duplicate counting causing abnormal proportions)
        total_households = len(cluster_data)  # Total number of households

        evaluated_set = (
            set(self.validated_household_ids)
            | set(self.problem_household_ids)
            | set(self.pending_validation_household_ids)
        )
        problem_set = set(self.problem_household_ids) | set(self.pending_validation_household_ids)

        total_validated_global = len(self.validated_household_ids)  # Validated households
        total_outliers_global = len(self.problem_household_ids)  # Clearly outliers/problem samples
        total_pending_global = len(self.pending_validation_household_ids)  # Pending validation
        total_evaluated_global = len(evaluated_set)  # Evaluated coverage (validated/outliers/pending validation)
        total_problem_global = len(problem_set)  # Global problem samples (outliers + pending validation)

        evaluated_coverage_rate = total_evaluated_global / total_households if total_households > 0 else 0
        global_coverage_rate = total_validated_global / total_households if total_households > 0 else 0
        # Note: global_outlier_rate represents "problem rate" (outliers + pending validation) / total sample
        global_outlier_rate = total_problem_global / total_households if total_households > 0 else 0
        global_pending_rate = total_pending_global / total_households if total_households > 0 else 0
        
        # 3. This iteration statistics 
        iter_samples = total_validated + total_outliers
        iter_validation_rate = total_validated / iter_samples if iter_samples > 0 else 0
        iter_outlier_rate = total_outliers / iter_samples if iter_samples > 0 else 0

        # 4. Cumulative "evaluated" internal structure 
        evaluated_count = total_evaluated_global
        evaluated_validation_rate = (
            total_validated_global / evaluated_count if evaluated_count > 0 else 0
        )
        evaluated_outlier_rate = (
            total_problem_global / evaluated_count if evaluated_count > 0 else 0
        )
        
        created_in_stage2 = 0
        try:
            created_in_stage2 = int(locals().get("created", 0))
        except Exception:
            created_in_stage2 = 0

        coverage_for_convergence = evaluated_coverage_rate
        outlier_for_convergence = global_outlier_rate
        coverage_label = "Cumulative evaluated coverage rate"
        outlier_label = "Global problem rate (outliers + pending validation)"

        # Need to satisfy:
        # 1) Average fit score >= 0.65
        # 2) Cumulative evaluated coverage rate >= 85% (sampling iteration must cover enough households)
        # 3) Global problem rate < 5% (outliers + pending validation is very low)
        # 4) This iteration has no new Persona, and pending validation = 0 (system stable)

        at_capacity = len(self.persona_trackers) >= self.max_personas
        pending_ok = total_pending_global == 0
        no_new_persona = created_in_stage2 == 0

        # quality_ok = (avg_fit >= 0.65) and (outlier_for_convergence < 0.05)
        quality_ok = outlier_for_convergence < 0.05
        coverage_ok = coverage_for_convergence >= 0.85
        system_stable = pending_ok and (no_new_persona or at_capacity)

        is_converged = quality_ok and coverage_ok and system_stable
        stop_reason: Optional[str] = "criteria" if is_converged else None
        

        # ==================== Stagnation detection: Avoid infinite loop ====================
        # If there are multiple consecutive iterations with "no evaluation + no new persona + no similar outlier expansion", it means the system is stuck (common reasons: max_personas is full and outliers > threshold)
        stalled_this_iter = (
            iter_samples == 0 and created_in_stage2 == 0 and total_inferred_outliers == 0
        )
        if not hasattr(self, "_stall_no_change_iters"):
            self._stall_no_change_iters = 0
            self._last_stall_signature = None

        if stalled_this_iter:
            signature = (
                int(total_validated_global),
                int(total_outliers_global),
                int(total_pending_global),
                int(len(self.persona_trackers)),
                int(self.max_personas),
                bool(at_capacity),
                tuple(sorted(self.cluster_to_persona.items())),
            )
            if signature == getattr(self, "_last_stall_signature", None):
                self._stall_no_change_iters += 1
            else:
                self._stall_no_change_iters = 1
                self._last_stall_signature = signature
        else:
            self._stall_no_change_iters = 0
            self._last_stall_signature = None

        if self._stall_no_change_iters >= 3:
            is_converged = True
            stop_reason = "stalled"
            print(
                f"Detected {self._stall_no_change_iters} consecutive iterations with no change, stopping iteration (stagnation convergence)"
            )
        
        # Record history 
        self.iteration_history.append({
            'iteration': iteration,
            'avg_fit_score': avg_fit,
            'global_coverage_rate': global_coverage_rate,
            'global_outlier_rate': global_outlier_rate,
            'global_pending_rate': global_pending_rate,
            'iter_validation_rate': iter_validation_rate,
            'iter_outlier_rate': iter_outlier_rate,
            'converge_metric': 'sample' if sample_size else 'global',
            'converge_coverage_rate': coverage_for_convergence,
            'converge_outlier_rate': outlier_for_convergence,
            'stage2_trigger_threshold': stage2_trigger_threshold,
            'num_personas_created_stage2': created_in_stage2,
            'num_validated_global': total_validated_global,
            'num_outliers_global': total_outliers_global,
            'num_pending_global': total_pending_global,
            'num_validated_iter': total_validated,
            'num_outliers_iter': total_outliers,
            'num_personas': len(self.persona_trackers),
            'stop_reason': stop_reason,
            'is_converged': is_converged
        })
        self.last_stop_reason = stop_reason
        
        # Return updated Personas
        updated_personas = self._get_current_personas()
        
        
        return is_converged, updated_personas, cluster_data
    
    def _get_current_personas(self) -> List[Dict]:
        """Get current all Personas data"""
        return [tracker.persona_data for tracker in self.persona_trackers.values()]
    
    async def _evaluate_questionnaire_score(
        self,
        household: pd.Series,
        persona: Dict
    ) -> float:
        """
        Evaluate the questionnaire matching score between household and persona
        
        Use HouseholdQuestionEvaluator to let LLM answer 10 questionnaire questions based on persona description + household basic information,
        then compare the answers directly in memory with ground truth, calculate the accuracy
        
        Args:
            household: Household data (needs to contain original fields for generating ground truth)
            persona: Persona data
            
        Returns:
            float: Questionnaire matching score (0.0-1.0).
            If a question is missing ground truth (or is Unknown), it is considered "automatically correct" (no penalty).
        """
        try:
            def _normalize_answer(raw: Any) -> str:
                if raw is None or (isinstance(raw, float) and not math.isfinite(raw)):
                    return ""
                if pd.isna(raw):
                    return ""
                s = str(raw).strip().upper()
                if not s:
                    return ""
                # Common cases: "A", "A.", "A)"...
                if s[0] in {"A", "B", "C", "D", "E"}:
                    return s[0]
                # More tolerant: tokenize and pick standalone A/B/C/D/E.
                cleaned = "".join(ch if ch.isalnum() else " " for ch in s)
                for tok in cleaned.split():
                    if tok in {"A", "B", "C", "D", "E"}:
                        return tok
                return ""

            dataset_lower = (self.dataset or "").strip().lower()
            if dataset_lower == "psid":
                gt_unknown_c_keys = {"q1", "q2", "q4", "q6", "q7", "q8", "q9", "q10"}
            elif dataset_lower == "shed":
                gt_unknown_c_keys = {"q1", "q3", "q4", "q6", "q7", "q8", "q9", "q10"}
            else:
                gt_unknown_c_keys = {f"q{i}" for i in range(1, 11)}

            gt_values: Dict[str, str] = {}
            missing_gt_q_keys: set[str] = set()
            all_q_keys = [f"q{i}" for i in range(1, 11)]
            for q_key in all_q_keys:
                gt_col = f"{q_key}_gt"
                if gt_col not in household.index:
                    missing_gt_q_keys.add(q_key)
                    continue
                gt_raw = household.get(gt_col, None)
                if gt_raw is None or (isinstance(gt_raw, float) and not math.isfinite(gt_raw)):
                    missing_gt_q_keys.add(q_key)
                    continue
                if pd.isna(gt_raw):
                    missing_gt_q_keys.add(q_key)
                    continue
                gt = str(gt_raw).strip().upper()
                if gt not in {"A", "B", "C", "D", "E"}:
                    missing_gt_q_keys.add(q_key)
                    continue
               
                is_unknown = False
                if dataset_lower == "psid":
                   
                    if q_key in {"q3", "q5"}:
                        is_unknown = (gt == "D")
                    else:
                        is_unknown = (gt == "C")
                elif dataset_lower == "shed":
                    #
                    if q_key in {"q2", "q5"}:
                        is_unknown = (gt == "D")
                    else:
                        is_unknown = (gt == "C")
                else:
                    
                    is_unknown = (gt == "C" and q_key in gt_unknown_c_keys)
                
                if is_unknown:
                   
                    missing_gt_q_keys.add(q_key)
                    continue
                gt_values[q_key] = gt

            
            if not gt_values:
                return 1.0
            
            # 1. Build household description text (for LLM inference)
            # Contains: Persona complete description + household basic information (age, household size)
            # Does not contain: Specific financial data, behavior indicators
            household_description = self._build_household_description_for_survey(
                household, persona
            )
            
            # 2. Call LLM to answer questionnaire
            response = await self._call_question_evaluator(household_description, debug=False)
            
            # 3. Parse LLM's response
            predictions = self.question_evaluator.extract_json(response)
            

            if not isinstance(predictions, dict):
                baseline = float(len(missing_gt_q_keys)) / 10.0
                return max(0.5, baseline)

            correct = len(missing_gt_q_keys)
            for q_key, gt in gt_values.items():
                pred = _normalize_answer(predictions.get(q_key, None))
                if pred and pred == gt:
                    correct += 1
            return float(correct) / 10.0
            
        except Exception as e:
            return 0.5
    
    def _build_household_description_for_survey(
        self,
        household: pd.Series,
        persona: Dict
    ) -> str:
        """
        Build household description text for questionnaire evaluation
        
        Args:
            household: Household data
            persona: Persona data

        Returns:
          str: Household description text 
                
        Note:
            - Always contains: Persona description (behavior patterns, decision logic, etc.)
            - Household basic information always comes first (DEMOGRAPHICS)
            - Does not contain: Specific financial data, behavior indicators (these are used for validation ground truth)
        """
        def _format_value(value: Any) -> Optional[str]:
            if pd.isna(value):
                return None
            if isinstance(value, (np.integer, int, bool)):
                return str(int(value))
            if isinstance(value, (np.floating, float)):
                v = float(value)
                if math.isfinite(v) and v.is_integer():
                    return str(int(v))
                return f"{v:.3f}".rstrip("0").rstrip(".")
            if isinstance(value, str):
                s = value.strip()
                return s or None
            return str(value)

        household_key_by_lower = {str(k).strip().lower(): k for k in household.index}

        def _get_first_value(candidates: List[str]) -> Tuple[Optional[str], Any]:
            for cand in candidates:
                if not cand:
                    continue
                key = household_key_by_lower.get(str(cand).strip().lower())
                if key is None:
                    continue
                val = household.get(key)
                if not pd.isna(val):
                    return str(key), val
            return None, None

        description_parts: List[str] = []

        demographics_lines: List[str] = []
        used_keys_lower: set[str] = set()

        # 优先使用标准化字段（如已存在）
        age_key, age_val = _get_first_value(["head_age", "age", "HHLDRAGEP", "AGEP", "TAGE", "ER82018"])
        if age_key is not None:
            age_txt = _format_value(age_val)
            if age_txt is not None:
                demographics_lines.append(f"- Age: {age_txt} years")
                used_keys_lower.add(age_key.lower())

        size_key, size_val = _get_first_value(["family_size", "household_size", "FAM_SIZE", "NP", "NPF", "pphhsize", "ER82017"])
        if size_key is not None:
            size_txt = _format_value(size_val)
            if size_txt is not None:
                demographics_lines.append(f"- Household size: {size_txt}")
                used_keys_lower.add(size_key.lower())

        children_key, children_val = _get_first_value(["num_children", "RHNUMU18", "TCEB", "ppkid017", "HUPAC", "ER82022"])
        if children_key is not None:
            children_txt = _format_value(children_val)
            if children_txt is not None:
                label = "Number of children" if children_key.lower() in {"num_children", "tceb"} else "Children indicator"
                if children_key.lower() == "rhnumu18":
                    label = "Number of people under 18"
                if children_key.lower() == "ppkid017":
                    label = "Children age 0-17 present"
                demographics_lines.append(f"- {label}: {children_txt}")
                used_keys_lower.add(children_key.lower())

        marital_key, marital_val = _get_first_value(["marital_status", "MAR", "MARITAL1", "ppmarit5", "EMS", "ER82026"])
        if marital_key is not None:
            marital_txt = _format_value(marital_val)
            if marital_txt is not None:
                demographics_lines.append(f"- Marital status: {marital_txt}")
                used_keys_lower.add(marital_key.lower())

        edu_key, edu_val = _get_first_value(["education_level", "head_education", "SCHL", "EDUC_REF", "HIGH_EDU", "ppeduc5", "ED0", "EEDUC"])
        if edu_key is not None:
            edu_txt = _format_value(edu_val)
            if edu_txt is not None:
                demographics_lines.append(f"- Education: {edu_txt}")
                used_keys_lower.add(edu_key.lower())

        dataset_key = (self.dataset or "").strip().lower()
        if dataset_key:
            cache = getattr(self, "_survey_demographics_fields_cache", None)
            if not isinstance(cache, dict):
                cache = {}
                setattr(self, "_survey_demographics_fields_cache", cache)

            if dataset_key not in cache:
                folder = "acs" if dataset_key == "acs" else dataset_key
                fields_path = os.path.join("data", folder, "extracted_fields_by_agent.json")
                demo_fields: List[Tuple[str, str]] = []
                try:
                    if os.path.exists(fields_path):
                        with open(fields_path, "r", encoding="utf-8") as f:
                            payload = json.load(f)
                        if isinstance(payload, dict) and isinstance(payload.get("selected_fields"), list):
                            items = payload["selected_fields"]
                        elif isinstance(payload, list):
                            items = payload
                        else:
                            items = []

                        for item in items:
                            if not isinstance(item, dict):
                                continue
                            cat = (item.get("category") or "").strip().lower()
                            if cat != "demographics":
                                continue
                            var = item.get("var_id") or item.get("field_name") or item.get("variable_name")
                            if not isinstance(var, str) or not var.strip():
                                continue
                            label = (
                                item.get("short_label")
                                or item.get("short_description")
                                or item.get("concept_name")
                                or item.get("description")
                                or item.get("variable_label")
                                or item.get("label")
                                or var
                            )
                            demo_fields.append((var.strip(), str(label).strip()))
                except Exception:
                    demo_fields = []

                cache[dataset_key] = demo_fields

            
            for var, label in cache.get(dataset_key, []) or []:
                if len(demographics_lines) >= 15:
                    break
                var_lower = var.lower()
                if var_lower in used_keys_lower:
                    continue
                key = household_key_by_lower.get(var_lower)
                if key is None:
                    continue
                val = household.get(key)
                txt = _format_value(val)
                if txt is None:
                    continue
               
                safe_label = (label or var).strip()
                if len(safe_label) > 80:
                    safe_label = safe_label[:77] + "..."
                demographics_lines.append(f"- {safe_label}: {txt}")
                used_keys_lower.add(var_lower)

        if demographics_lines:
            description_parts.append("DEMOGRAPHICS:")
            description_parts.extend(demographics_lines)
            description_parts.append("")

       
        persona_name = persona.get("persona_name", "Unknown")
        core_characteristics = persona.get("core_characteristics", "")
        typical_profile = persona.get("typical_profile", "")

        description_parts.extend(
            [
                f"PERSONA: {persona_name}",
                "",
                "PERSONA PROFILE:",
                core_characteristics if core_characteristics else typical_profile,
                "",
            ]
        )
        
       
        behavior_patterns = persona.get('behavior_patterns', [])
        if behavior_patterns:
            description_parts.append("BEHAVIOR PATTERNS:")
            for pattern in behavior_patterns[:3]:  
                description_parts.append(f"- {pattern}")
            description_parts.append("")
        
    
        decision_making = persona.get('decision_making_logic', {})
        if decision_making:
            description_parts.append("DECISION-MAKING LOGIC:")
            
            
            financial_dec = decision_making.get('financial_decisions', {})
            if financial_dec:
                core_logic = financial_dec.get('core_logic', '')
                if core_logic:
                    description_parts.append(f"- Financial: {core_logic}")
            
           
            investment_dec = decision_making.get('investment_decisions', {})
            if investment_dec:
                risk_approach = investment_dec.get('risk_approach', '')
                if risk_approach:
                    description_parts.append(f"- Investment: {risk_approach}")
            
            description_parts.append("")
        
        
        behavioral_biases = persona.get('behavioral_biases', {})
        if behavioral_biases:
            description_parts.append("BEHAVIORAL TRAITS:")
            for bias_key in ['loss_aversion', 'present_bias', 'status_quo_bias']:
                if bias_key in behavioral_biases:
                    bias_desc = behavioral_biases[bias_key]
                    if isinstance(bias_desc, str) and len(bias_desc) > 0:
                        description_parts.append(f"- {bias_key.replace('_', ' ').title()}: {bias_desc[:100]}")
            description_parts.append("")
        
        return "\n".join(description_parts)
    
    async def _call_question_evaluator(self, household_description: str, debug: bool = False) -> str:
        """
        Call questionnaire evaluator (handle synchronous/asynchronous model)
        
        Args:
            household_description: Household description text
            debug: Whether to print debug information
            
        Returns:
            str: LLM's response
        """
        try:

            response = await self.question_evaluator.answer_survey(household_description)

            # answer_survey现在直接返回字符串，不需要进一步处理
            if isinstance(response, str):
                return response
            else:
                return str(response)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return "{}"
    
    def _find_similar_personas(
        self,
        household: pd.Series,
        current_persona_id: int,
        top_k: int = 2
    ) -> List[int]:
        """
        Based on statistical feature distance, quickly find the most similar persona to the household

        This method does not call LLM, only calculates distance based on numerical features, and is very fast
        
        Args:
            household: Household data
            current_persona_id: Current persona ID (will be excluded)
            top_k: Return top K most similar personas
            
        Returns:
            List[int]: List of the most similar persona IDs
        """
        def _to_float(val: Any) -> Optional[float]:
            if pd.isna(val):
                return None
            if isinstance(val, bool):
                return float(int(val))
            if isinstance(val, (int, float, np.integer, np.floating)):
                v = float(val)
                return v if math.isfinite(v) else None
            s = str(val).strip()
            if not s:
                return None
            try:
                v = float(s)
                return v if math.isfinite(v) else None
            except Exception:
                return None

        household_key_by_lower = {str(k).strip().lower(): k for k in household.index}

        def _household_value(base: str) -> Any:
            key = household_key_by_lower.get(str(base).strip().lower())
            return household.get(key) if key is not None else pd.NA

      
        distances = []
        skipped_personas = []  

        for persona_id, tracker in self.persona_trackers.items():
            if current_persona_id >= 0 and persona_id == current_persona_id:
                continue  # Skip current persona (if current_persona_id >= 0)

            # Extract persona's statistical features
            stats = tracker.persona_data.get("statistics", {}) or {}
            if not isinstance(stats, dict) or not stats:
                skipped_personas.append((persona_id, tracker.persona_data.get("persona_name", "Unknown")))
                continue

            # Single dataset validation: Use the features in persona.statistics for quick pre-selection
            numeric_sum_sq = 0.0
            numeric_used = 0
            cat_mismatch = 0
            cat_used = 0

            for key, mean_val in stats.items():
                if not isinstance(key, str) or not key.endswith("_mean"):
                    continue
                base = key[: -len("_mean")]
                if base in {"cluster_id", "n_samples", "percentage"}:
                    continue

                h_val = _to_float(_household_value(base))
                m_val = _to_float(mean_val)
                if m_val is None:
                    continue
                if h_val is None:
                    h_val = m_val

                std_val = _to_float(stats.get(f"{base}_std"))
                if std_val is None or std_val <= 0:
                    std_val = 1.0

                numeric_sum_sq += ((h_val - m_val) / std_val) ** 2
                numeric_used += 1

            for key, mode_val in stats.items():
                if not isinstance(key, str) or not key.endswith("_mode"):
                    continue
                base = key[: -len("_mode")]
                if base in {"cluster_id", "n_samples", "percentage"}:
                    continue

                h_raw = _household_value(base)
                if mode_val is None or pd.isna(mode_val):
                    continue

                if pd.isna(h_raw):
                    continue

                cat_used += 1
                if str(h_raw).strip() != str(mode_val).strip():
                    cat_mismatch += 1

            if numeric_used == 0 and cat_used == 0:
                skipped_personas.append((persona_id, tracker.persona_data.get("persona_name", "Unknown")))
                continue

            numeric_distance = math.sqrt(numeric_sum_sq) if numeric_used else 0.0
            distance = numeric_distance + 0.5 * float(cat_mismatch)
            distances.append((persona_id, distance))

        
        # Sort by distance, return top K
        distances.sort(key=lambda x: x[1])
        similar_persona_ids = [pid for pid, _ in distances[:top_k]]
    
        
        return similar_persona_ids

    def _infer_similar_outlier_households(
        self,
        *,
        seed_household_idx: int,
        seed_household: pd.Series,
        cluster_id: int,
        persona_id: int,
        cluster_households: pd.DataFrame,
        exclude_household_ids: set,
        outlier_dict: Dict[int, Dict[str, Any]],
        already_evaluated_in_this_persona: set,
        max_additional: int,
    ) -> int:
        """
        When a sample finds an outlier seed household, calculate the structural feature similarity within the same cluster,
        directly select the Top-K (max_additional) most similar households as "inferred outliers" to supplement the sample for stage 2 re-clustering.
        """
        if max_additional <= 0:
            return 0
        if cluster_households is None or len(cluster_households) == 0:
            return 0

        continuous_fields, categorical_fields = self._load_selected_fields_for_reclustering(
            cluster_households
        )
        if not continuous_fields and not categorical_fields:
            return 0

        std_by_field: Dict[str, float] = {}
        for col in continuous_fields:
            std = 1.0
            try:
                vals = pd.to_numeric(cluster_households[col], errors="coerce")
                std = float(vals.std(ddof=0))
            except Exception:
                std = 1.0
            if not math.isfinite(std) or std <= 1e-12:
                std = 1.0
            std_by_field[str(col)] = std

        def _similarity_score(cand_household: pd.Series) -> Optional[Tuple[float, int, int]]:
            numeric_used = 0
            numeric_sum_sq = 0.0
            for col in continuous_fields:
                a = pd.to_numeric(seed_household.get(col), errors="coerce")
                b = pd.to_numeric(cand_household.get(col), errors="coerce")
                if pd.isna(a) or pd.isna(b):
                    continue
                numeric_used += 1
                std = std_by_field.get(str(col), 1.0)
                z = float(a - b) / float(std)
                numeric_sum_sq += z * z

            cat_used = 0
            cat_mismatch = 0
            for col in categorical_fields:
                a_raw = seed_household.get(col)
                b_raw = cand_household.get(col)
                if pd.isna(a_raw) or pd.isna(b_raw):
                    continue
                cat_used += 1
                if str(a_raw).strip() != str(b_raw).strip():
                    cat_mismatch += 1

            if numeric_used == 0 and cat_used == 0:
                return None

            numeric_dist = (
                math.sqrt(numeric_sum_sq / float(numeric_used))
                if numeric_used
                else 0.0
            )
            cat_dist = (float(cat_mismatch) / float(cat_used)) if cat_used else 0.0
            dist = float(numeric_dist) + 0.5 * float(cat_dist)
            sim = 1.0 / (1.0 + dist)
            return float(sim), int(numeric_used), int(cat_used)

        scored: List[Tuple[float, Any, pd.Series, int, int]] = []
        for cand_idx, cand_household in cluster_households.iterrows():
            try:
                cand_idx_int = int(cand_idx)
            except Exception:
                cand_idx_int = cand_idx

            if cand_idx_int == seed_household_idx:
                continue
            if cand_idx_int in exclude_household_ids:
                continue
            if cand_idx_int in outlier_dict:
                continue
            if cand_idx_int in already_evaluated_in_this_persona:
                continue
            if cand_idx_int in self.validated_household_ids:
                continue
            if cand_idx_int in self.pending_validation_household_ids:
                continue

            score_tuple = _similarity_score(cand_household)
            if not score_tuple:
                continue
            sim, numeric_used, cat_used = score_tuple
            scored.append((sim, cand_idx_int, cand_household, numeric_used, cat_used))

        if not scored:
            return 0

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[: int(max_additional)]

        added = 0
        for sim, cand_idx_int, cand_household, numeric_used, cat_used in top:
            used_fields = int(numeric_used) + int(cat_used)
            outlier_dict[cand_idx_int] = {
                "household_idx": cand_idx_int,
                "household_data": cand_household,
                "original_cluster": int(cluster_id),
                "assigned_persona_id": int(persona_id),
                "fit_score": None,
                "similarity_score": float(sim),
                "reason": (
                    f"Inferred similar-outlier to household#{seed_household_idx} "
                    f"(sim={float(sim):.3f}, fields={used_fields})"
                ),
            }

            self.problem_household_ids.add(cand_idx_int)
            self.pending_validation_household_ids.discard(cand_idx_int)
            already_evaluated_in_this_persona.add(cand_idx_int)
            self.validated_household_ids.discard(cand_idx_int)
            added += 1

        if added > 0:
            best_sim = float(top[0][0])
            worst_sim = float(top[-1][0])

        return int(added)
    
    
    def save_iteration_results(
        self,
        iteration: int,
        personas: List[Dict],
        output_dir: str = "result",
        save_incremental: bool = False
    ):
        """
        Save iteration results
        
        Args:
            iteration: Iteration number
            personas: Current Personas
            output_dir: Output directory
            save_incremental: Whether to save incremental files (save every iteration)
        """
        os.makedirs(output_dir, exist_ok=True)
        
        if save_incremental:
            personas_filename = f"personas_iter_{iteration}.json"
            tracker_filename = f"persona_trackers_iter_{iteration}.json"
        else:
            personas_filename = f"personas_final.json"
            tracker_filename = f"persona_trackers_final.json"
        

        personas_path = os.path.join(output_dir, personas_filename)
        with open(personas_path, 'w', encoding='utf-8') as f:
            json.dump(personas, f, ensure_ascii=False, indent=2)
        
        
        tracker_stats = {
            pid: tracker.to_dict()
            for pid, tracker in self.persona_trackers.items()
        }
        tracker_path = os.path.join(output_dir, tracker_filename)
        with open(tracker_path, 'w', encoding='utf-8') as f:
            json.dump(tracker_stats, f, ensure_ascii=False, indent=2)
    
    def _save_household_persona_mapping(
        self,
        cluster_data: pd.DataFrame,
        iteration: int,
        is_converged: bool,
        output_dir: str = "result",
    ):
        """
        Save the mapping of household (fid) to persona 
        
        Args:
            cluster_data: Household data with cluster labels
            iteration: Current iteration number
        """
        # Build mapping data
        mapping_data = []
        resolved_id_col = self._resolve_household_id_col(cluster_data, requested="fid")
        
        for household_idx, household in cluster_data.iterrows():
            raw_cluster_id = household.get("cluster", -1)
            try:
                cluster_id_current = int(raw_cluster_id)
            except Exception:
                cluster_id_current = -1

            persona_id_current = self.cluster_to_persona.get(cluster_id_current, -1)
            persona_name_current = "Unknown"
            if persona_id_current in self.persona_trackers:
                tracker_current = self.persona_trackers[persona_id_current]
                persona_name_current = tracker_current.persona_data.get(
                    "persona_name", "Unknown"
                )

            is_validated_current = household_idx in self.validated_household_ids

            mapping_source = "current_validated"
            cluster_id_saved = cluster_id_current
            persona_id_saved = persona_id_current

            if not is_validated_current:
                if household_idx in self.household_last_validated_persona_id:
                    persona_id_saved = int(
                        self.household_last_validated_persona_id[household_idx]
                    )
                    cluster_id_saved = int(
                        self.household_last_validated_cluster_id.get(household_idx, -1)
                    )
                    mapping_source = "last_validated"
                else:
                    cluster_id_saved = int(
                        self.household_initial_cluster_id.get(household_idx, cluster_id_current)
                    )
                    persona_id_saved = int(
                        self.household_initial_persona_id.get(household_idx, persona_id_current)
                    )
                    mapping_source = "initial"

            persona_name_saved = "Unknown"
            persona_avg_fit_score = None
            if persona_id_saved in self.persona_trackers:
                tracker_saved = self.persona_trackers[persona_id_saved]
                persona_name_saved = tracker_saved.persona_data.get(
                    "persona_name", "Unknown"
                )
                persona_avg_fit_score = tracker_saved.fit_score

            mapping_data.append({
                'household_idx': household_idx,
                'fid': household.get(resolved_id_col, household_idx),  
                'cluster_id': int(cluster_id_saved),
                'persona_id': int(persona_id_saved),
                'persona_name': persona_name_saved,
                'persona_avg_fit_score': persona_avg_fit_score,
                'cluster_id_current': int(cluster_id_current),
                'persona_id_current': int(persona_id_current),
                'persona_name_current': persona_name_current,
                'mapping_source': mapping_source,
                'is_validated': is_validated_current,
                'iteration': iteration
            })
        
        
        mapping_df = pd.DataFrame(mapping_data)
        
        
        if is_converged:
            output_path = os.path.join(output_dir, "household_persona_mapping_final.csv")
        else:
            output_path = os.path.join(output_dir, f"household_persona_mapping_iter_{iteration}.csv")
        
        mapping_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        

def _resolve_dataset_io(dataset: str) -> Dict[str, str]:
    """
    Resolve input/output paths for iterative refinement based on the same clustering outputs layout.
    """
    dataset_key = (dataset or "").strip().lower()

    if isinstance(CLUSTERING_DATASET_CONFIG, dict) and dataset_key in CLUSTERING_DATASET_CONFIG:
        base = os.path.normpath(CLUSTERING_DATASET_CONFIG[dataset_key]["output_dir"])
    else:
        base = os.path.normpath(os.path.join("result", f"{dataset_key}_clustering_output"))

    cluster_data_path = os.path.join(base, "data_with_clusters.csv")
    personas_path = os.path.join(base, "personas.json")

    return {
        "dataset": dataset_key,
        "output_dir": base,
        "cluster_data_path": cluster_data_path,
        "personas_path": personas_path,
    }


async def main(dataset: str = "psid"):
    """Main function - Demonstrates core algorithm loop (async version)"""
    from llm_config import create_llm_model
    
    
    # Create model
    model = create_llm_model(temperature=0.7, max_tokens=2000)
    
    io_cfg = _resolve_dataset_io(dataset)
    
    # Create engine (similar to algorithm's Update class)
    engine = IterativeRefinementEngine(
        model=model,
        max_iterations=None,     # None/<=0 means until convergence
        alpha=0.2,              # UCB exploration parameter
        fit_threshold=0.6,      # Fitness threshold starting from 0.6
        min_problem_rate=0.05,  # Problem sample rate threshold (stages 1/2/3 all by rate)
        stage2_max_outliers=2000,  # Max outliers for stage 2 reclustering (cost control)
        max_personas=1000000,        # Max number of personas
        dataset=io_cfg["dataset"],
        output_dir=io_cfg["output_dir"],
        random_state=42, 
    )
    
    # Initialize components (Generation, Inference)
    engine.initialize_agents()
    # Optional: provide dataset context so persona_generation can choose better prompts/metadata.
    try:
        if engine.dataset:
            engine.persona_generator.set_dataset_context({"dataset": engine.dataset})
    except Exception:
        pass
    
    # Load data
    print("\n=== Loading Data ===")
    cluster_data = pd.read_csv(io_cfg["cluster_data_path"], encoding="utf-8-sig")
    print(f"✓ Loaded {len(cluster_data)} households")
    
    # Load initial personas (similar to initialize_hypotheses)
    with open(io_cfg["personas_path"], 'r', encoding='utf-8') as f:
        initial_personas = json.load(f)
    print(f"✓ 加载 {len(initial_personas)} 个初始Persona")
    
    # 核心算法循环
    current_personas = initial_personas
    final_iteration = 1
    final_cluster_data = cluster_data  
    did_converge = False
    
    i = 1
    while True:
        if engine.max_iterations is not None and i > engine.max_iterations:
            break
        
        is_converged, updated_personas, updated_cluster_data = await engine.run_iteration(
            personas=current_personas,
            cluster_data=cluster_data,
            sample_size=True,  
            iteration=i
        )
        
    
        current_personas = updated_personas
        cluster_data = updated_cluster_data  
        final_cluster_data = updated_cluster_data
        final_iteration = i
        
       
        if i % 5 == 0:
            engine.save_iteration_results(i, updated_personas, output_dir=io_cfg["output_dir"], save_incremental=True)
            engine._save_household_persona_mapping(final_cluster_data, i, is_converged=False, output_dir=io_cfg["output_dir"])
        
        if is_converged:
            did_converge = True
            break
        
        i += 1
    
    engine.save_iteration_results(final_iteration, current_personas, output_dir=io_cfg["output_dir"], save_incremental=False)
    
    engine._save_household_persona_mapping(
        final_cluster_data,
        final_iteration,
        is_converged=did_converge,
        output_dir=io_cfg["output_dir"],
    )
    
