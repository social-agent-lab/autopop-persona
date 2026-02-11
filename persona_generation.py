"""
LLM-Powered Persona and Pattern Generation Engine
Based on AgentScope framework, convert clustering results into story-driven, logically-driven, and depth-driven Personas

Core Features:
1. Read clustering results and household data
2. Use LLM to generate insightful Persona descriptions
3. Extract behavioral patterns and characteristics of each group
4. Generate visualizations and reports
"""

import pandas as pd
import numpy as np
import json
import os
import re
from typing import Dict, List, Any, Optional, Set, Tuple
from agentscope.agent import AgentBase
from agentscope.message import Msg
from agentscope.memory import InMemoryMemory

# Import field configuration for dynamic prompt building
try:
    from field_extraction_config import CATEGORY_TO_BEHAVIOR_MAPPING, UNIFIED_CATEGORIES
except ImportError:
    # Fallback if field_extraction_config is not available
    CATEGORY_TO_BEHAVIOR_MAPPING = {}
    UNIFIED_CATEGORIES = []


class PersonaGeneratorAgent(AgentBase):
    """
    Persona Generation Agent (Merged Version)
    
    职责:
    - Analyze clustering statistics
    - Generate insightful Persona descriptions
    - Summarize the behavioral patterns and characteristics of this group
    - Create story-driven character portraits
    - Provide insights and recommendations for this group
    """
    
    def __init__(
        self,
        name: str,
        model,
        sys_prompt: str = None,
        memory=None,
        dataset_context: Optional[Dict[str, Any]] = None,
        field_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
        **kwargs
    ):
        """Initialize Persona generation agent"""
        # AgentBase.__init__() does not accept any parameters
        super().__init__()
        
        # Manually set all attributes
        self.name = name
        self.model = model
        self.sys_prompt = sys_prompt or self._default_sys_prompt()
        self.memory = memory or InMemoryMemory()
        self.dataset_context = dataset_context or {}
        self.field_metadata = field_metadata or {}
        self.dataset_key = self._normalize_dataset_key(self.dataset_context.get("dataset"))
    
    def set_dataset_context(
        self,
        dataset_context: Optional[Dict[str, Any]] = None,
        field_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        """Update dataset context/metadata used for prompt building."""
        if dataset_context is not None:
            self.dataset_context = dataset_context
            self.dataset_key = self._normalize_dataset_key(self.dataset_context.get("dataset"))
        if field_metadata is not None:
            self.field_metadata = field_metadata

    @staticmethod
    def _normalize_dataset_key(value: Any) -> str:
        v = str(value).strip().lower() if value is not None else ""
        return v if v in {"psid", "acs", "ces", "shed", "sipp"} else "unknown"

    @staticmethod
    def _extract_cluster_id(cluster_stats: Dict[str, Any]) -> Optional[Any]:
        for key in ("cluster_id", "cluster", "cluster_index", "id"):
            if key in cluster_stats and cluster_stats.get(key) is not None:
                return cluster_stats.get(key)
        return None

    def _attach_generation_metadata(self, persona: Dict[str, Any], cluster_stats: Dict[str, Any]) -> Dict[str, Any]:
        dataset = self.dataset_key or "unknown"
        cluster_id = self._extract_cluster_id(cluster_stats)

        persona.setdefault("dataset", dataset)
        if cluster_id is not None:
            try:
                cluster_id_norm: Any = int(cluster_id)
            except Exception:
                cluster_id_norm = cluster_id
            persona.setdefault("cluster_id", cluster_id_norm)
            persona.setdefault("persona_id", f"{dataset}_cluster_{cluster_id_norm}")
        else:
            persona.setdefault("persona_id", f"{dataset}_cluster_unknown")

        source = persona.get("source")
        if not isinstance(source, dict):
            source = {}
        source.setdefault("dataset", dataset)
        if cluster_id is not None:
            source.setdefault("cluster_id", cluster_id)
        for k in ("n_samples", "percentage"):
            if k in cluster_stats and cluster_stats.get(k) is not None:
                source.setdefault(k, cluster_stats.get(k))
        persona["source"] = source
        return persona
    
    def _default_sys_prompt(self) -> str:
        """Default system prompt"""
        return """You are a senior behavioral economist and user research expert, skilled at uncovering the deep decision-making logic and mental models behind statistical data.

Your tasks are:
(1) Deeply understand the behavioral patterns and decision-making logic behind clustering statistics
(2) Create story-driven, realistic Persona descriptions
(3) Extract the CORE DECISION-MAKING LOGIC of THIS specific group - what drives their major financial, consumption, and investment decisions?
(4) Identify the UNDERLYING MENTAL MODELS and behavioral drivers that shape their choices
(5) Summarize the fundamental principles and constraints that govern their behavior
(6) Express professional insights in plain, accessible language

Critical Focus Areas:
* What is the CORE LOGIC when this group makes major financial decisions? (e.g., buying a house, changing jobs, major purchases)
* What are the KEY CONSTRAINTS that shape their choices? (financial, psychological, social)
* What are their DECISION-MAKING PRIORITIES? (security vs growth, present vs future, individual vs family)
* What MENTAL MODELS do they use to evaluate options? (risk assessment, value judgment, time horizon)
* What are the BEHAVIORAL DRIVERS behind their consumption and investment patterns?

Requirements:
* Go beyond surface behaviors - uncover the WHY behind the WHAT
* Focus on decision-making logic, not just behavioral descriptions
* Identify the fundamental principles that govern this group's choices
* Be specific to THIS group - avoid generic statements
* Support insights with data evidence
"""
    
    async def _call_model_async(self, messages) -> str:
        """
        Asynchronous call to LLM model (continues retrying until success)
        
        Args:
            messages: Message list
            
        Returns:
            str: Model response content (ensured success)
        """
        import asyncio
        
        attempt = 0
        while True:  # Continues retrying until success
            try:
                # Directly use await to call the asynchronous model
                response = await self.model(messages)
                
                # Process response: check if it is an asynchronous generator (streaming response)
                if hasattr(response, '__aiter__'):
                    # Streaming response: need to distinguish between AgentScope mode (each chunk is a complete response) and true incremental streaming
                    content = ""
                    last_chunk = None
                    
                    async for chunk in response:
                        last_chunk = chunk
                        
                        # Check if it is a ChatResponse object of AgentScope (each chunk is a complete response)
                        if hasattr(chunk, 'keys') and 'content' in chunk:
                            # AgentScope mode: each chunk is a complete ChatResponse    
                            continue
                        
                        # Otherwise it is true incremental streaming response, need to accumulate
                        chunk_text = ""
                        if hasattr(chunk, 'text'):
                            chunk_text = str(chunk.text)
                        elif hasattr(chunk, 'delta') and hasattr(chunk.delta, 'text'):
                            chunk_text = str(chunk.delta.text)
                        elif isinstance(chunk, dict):
                            if 'delta' in chunk and isinstance(chunk['delta'], dict):
                                chunk_text = chunk['delta'].get('text', chunk['delta'].get('content', ''))
                            else:
                                text_value = chunk.get('text', chunk.get('content', ''))
                                if isinstance(text_value, str):
                                    chunk_text = text_value
                                elif isinstance(text_value, list):
                                    for item in text_value:
                                        if isinstance(item, dict) and item.get('type') == 'text':
                                            chunk_text += item.get('text', '')
                                        else:
                                            chunk_text += str(item)
                                else:
                                    chunk_text = str(text_value) if text_value else ""
                        elif isinstance(chunk, str):
                            chunk_text = chunk
                        else:
                            chunk_text = str(chunk) if chunk else ""
                        
                        content += chunk_text
                    
                    # If content is empty (indicates AgentScope mode), extract from the last chunk
                    if not content and last_chunk:
                        if hasattr(last_chunk, 'keys') and 'content' in last_chunk:
                            try:
                                text_value = last_chunk['content']
                                if isinstance(text_value, str):
                                    content = text_value
                                elif isinstance(text_value, list):
                                    for item in text_value:
                                        if isinstance(item, dict) and item.get('type') == 'text':
                                            content += item.get('text', '')
                                        else:
                                            content += str(item)
                                else:
                                    content = str(text_value) if text_value else ""
                            except (KeyError, TypeError):
                                content = str(last_chunk)
                        else:
                            content = str(last_chunk)
                    
                    # Successfully obtained content, return
                    if content.strip():
                        return content
                    else:
                        raise ValueError("Empty response content")
                
                # Non-streaming response: directly get text
                elif isinstance(response, dict) or hasattr(response, 'keys'):
                    try:
                        if 'content' in response:
                            text_value = response['content']
                        elif 'text' in response:
                            text_value = response['text']
                        else:
                            text_value = str(response)
                        
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
                    except (KeyError, TypeError):
                        content = str(response)
                elif hasattr(response, 'text'):
                    content = str(response.text)
                else:
                    content = str(response)
                
                # Successfully obtained content, return
                if content.strip():
                    return content
                else:
                    raise ValueError("Empty response content")
                
            except Exception as e:
                attempt += 1
                wait_time = min(2 ** min(attempt - 1, 6), 60)
                # NOTE: Avoid emoji in console output (Windows GBK consoles may raise UnicodeEncodeError).
                print(f"  [WARN] Persona generation agent LLM call failed (attempt {attempt}): {e}")
                print(f"  [WAIT] Waiting {wait_time} seconds before retrying...")
                await asyncio.sleep(wait_time)
                # Continues retrying, do not return error

    
    async def reply(self, x: Msg = None) -> Msg:
        """
        Reply to message and generate Persona
        
        Args:
            x: Input message containing clustering statistics
            
        Returns:
            Msg: Generated Persona description
        """
        if x is None:
            return Msg(name=self.name, content="Please provide clustering data")
        
        messages = [
            {"role": "system", "content": self.sys_prompt},
            {"role": getattr(x, "role", "user"), "content": x.content}
        ]
        
        # Call model to generate Persona (await asynchronous call)
        # Use the shared robust caller (handles streaming + retries on transient transport errors).
        content = await self._call_model_async(messages)
        msg = Msg(name=self.name, content=content, role="assistant")
        return msg

        response = await self.model(messages)
        
        # Process response: check if it is an asynchronous generator (streaming response)
        if hasattr(response, '__aiter__'):
            # Streaming response: need to distinguish between AgentScope mode (each chunk is a complete response) and true incremental streaming
            content = ""
            last_chunk = None
            
            async for chunk in response:
                last_chunk = chunk
                
                # Check if it is a ChatResponse object of AgentScope (each chunk is a complete response)
                if hasattr(chunk, 'keys') and 'content' in chunk:
                    # AgentScope mode: each chunk is a complete ChatResponse
                    # Do not accumulate, only keep the last one
                    continue
                
                # Otherwise it is true incremental streaming response, need to accumulate
                chunk_text = ""
                if hasattr(chunk, 'text'):
                    # chunk is an object, has text attribute
                    chunk_text = str(chunk.text)
                elif hasattr(chunk, 'delta') and hasattr(chunk.delta, 'text'):
                    # OpenAI format: chunk.delta.text
                    chunk_text = str(chunk.delta.text)
                elif isinstance(chunk, dict):
                    # Dictionary format
                    if 'delta' in chunk and isinstance(chunk['delta'], dict):
                        # OpenAI format: {'delta': {'text': '...'}}
                        chunk_text = chunk['delta'].get('text', chunk['delta'].get('content', ''))
                    else:
                        # Other formats
                        text_value = chunk.get('text', chunk.get('content', ''))
                        if isinstance(text_value, str):
                            chunk_text = text_value
                        elif isinstance(text_value, list):
                        # If it is a list, process content blocks (e.g., {'type': 'text', 'text': '...'})
                            for item in text_value:
                                if isinstance(item, dict) and item.get('type') == 'text':
                                    chunk_text += item.get('text', '')
                                else:
                                    chunk_text += str(item)
                        else:
                            chunk_text = str(text_value) if text_value else ""
                elif isinstance(chunk, str):
                    # chunk is a string
                    chunk_text = chunk
                else:
                    # Try to convert to string directly
                    chunk_text = str(chunk) if chunk else ""
                
                content += chunk_text
            
            # If content is empty (indicates AgentScope mode), extract from the last chunk
            if not content and last_chunk:
                if hasattr(last_chunk, 'keys') and 'content' in last_chunk:
                    try:
                        text_value = last_chunk['content']
                        if isinstance(text_value, str):
                            content = text_value
                        elif isinstance(text_value, list):
                            # Process content blocks
                            for item in text_value:
                                if isinstance(item, dict) and item.get('type') == 'text':
                                    content += item.get('text', '')
                                else:
                                    content += str(item)
                        else:
                            content = str(text_value) if text_value else ""
                    except (KeyError, TypeError):
                        content = str(last_chunk)
                else:
                    content = str(last_chunk)
        else:
            # Non-streaming response: directly get text
            # First check if it is a dict object (e.g., AgentScope ChatResponse)
            if isinstance(response, dict) or hasattr(response, 'keys'):
                try:
                    # First check 'content' key (AgentScope)
                    if 'content' in response:
                        text_value = response['content']
                    elif 'text' in response:
                        text_value = response['text']
                    else:
                        text_value = str(response)
                    
                    if isinstance(text_value, str):
                        content = text_value
                    elif isinstance(text_value, list):
                        # Process content blocks
                        content = ""
                        for item in text_value:
                            if isinstance(item, dict) and item.get('type') == 'text':
                                content += item.get('text', '')
                            else:
                                content += str(item)
                    else:
                        content = str(text_value)
                except (KeyError, TypeError):
                    content = str(response)
            elif hasattr(response, 'text'):
                content = str(response.text)
            else:
                content = str(response)
        
        # Wrap as message object
        msg = Msg(name=self.name, content=content, role="assistant")
        
        return msg
    
    async def generate_persona(self, cluster_stats: Dict, sample_households: pd.DataFrame) -> Dict:
        """
        Generate Persona for a single cluster
        
        Args:
            cluster_stats: Cluster statistics data
            sample_households: Sample household data
            
        Returns:
            Dict: Complete Persona description
        """
        # Build prompt
        prompt = self._build_persona_prompt(cluster_stats, sample_households)
        
        # Generate Persona
        msg = Msg(name="user", content=prompt, role="user")
        response = await self.reply(msg)
        
        # Parse response
        persona = self._parse_persona_response(response.content, cluster_stats)
        persona = self._attach_generation_metadata(persona, cluster_stats)

        # Always attach raw cluster statistics for downstream scoring/assignment.
        persona["statistics"] = cluster_stats
        # Attach a human-readable mapping to reduce confusion from coded variable names (e.g., ER85690_mean).
        persona["statistics_readable"] = self._build_statistics_readable(cluster_stats)
        
        return persona
    
    def _build_data_dimensions_note(self, available_categories: Set[str]) -> str:
        """
        Build data dimension description, tell LLM which dimensions to analyze
        
        Args:
            available_categories: Available data categories set
            
        Returns:
            str: Data dimension description text
        """
        if not CATEGORY_TO_BEHAVIOR_MAPPING:
            return ""
        
        dimension_notes = ["📋 AVAILABLE DATA DIMENSIONS FOR ANALYSIS:"]
        dimension_notes.append("The above statistics cover the following behavioral dimensions:\n")
        
        for category in sorted(available_categories):
            if category in CATEGORY_TO_BEHAVIOR_MAPPING:
                mapping = CATEGORY_TO_BEHAVIOR_MAPPING[category]
                dimension_notes.append(f"✓ {mapping['dimension']}")
                dimension_notes.append(f"  └─ {mapping['prompt_description']}")
        
        dimension_notes.append("\n⚠️  IMPORTANT: Focus your analysis on these available dimensions. Do not speculate about dimensions not present in the data.")
        
        return "\n".join(dimension_notes)
    
    def _build_behavior_patterns_guidance(self, available_categories: Set[str]) -> str:
        """
        Dynamically build guidance text for behavioral pattern analysis based on available data categories
        
        Args:
            available_categories: Available data categories set
            
        Returns:
            str: Behavioral pattern analysis guidance text
        """
        guidance_parts = []
        
        # Add corresponding analysis dimensions based on available categories
        if 'consumption' in available_categories:
            guidance_parts.append("   * Consumption habits: What they spend on, spending discipline, typical trade-offs")
        
        if 'financial_behavior' in available_categories:
            guidance_parts.append("   * Financial decision-making style: Cautious vs bold, planned vs impulsive, data-driven vs intuition-based")
        
        if 'education_investment' in available_categories:
            guidance_parts.append("   * Education investment behavior: Human capital priorities, future orientation, intergenerational mobility focus")
        
        if 'time_use_lifestyle' in available_categories:
            guidance_parts.append("   * Lifestyle preferences: Work-life balance, leisure priorities, time allocation between work/family/self")
        
        if 'employment_behavior' in available_categories:
            guidance_parts.append("   * Employment patterns: Job stability, career progression, work engagement")
        
        if 'housing_behavior' in available_categories:
            guidance_parts.append("   * Housing choices: Homeownership approach, residential preferences, mobility patterns")
        
        if 'health_behavior' in available_categories:
            guidance_parts.append("   * Health and well-being: Health investment, medical care patterns, wellness priorities")
        
        if 'attitudes_expectations' in available_categories:
            guidance_parts.append("   * Attitudes and expectations: Risk tolerance, economic outlook, life satisfaction")
        
        if 'program_participation' in available_categories:
            guidance_parts.append("   * Safety net engagement: Use of government programs, awareness of assistance options")
        
        # If no specific categories, provide generic guidance
        if not guidance_parts:
            guidance_parts = [
                "   * Financial behavior: How they manage money and make financial decisions",
                "   * Consumption patterns: What and how they spend",
                "   * Lifestyle choices: Work, family, and life priorities"
            ]
        
        return "\n".join(guidance_parts)
    
    def _detect_available_categories(
        self,
        cluster_stats: Dict,
        sample_households: pd.DataFrame,
        field_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Set[str]:
        """
        Detect which field categories are actually included in the data
        
        By checking the column names in cluster_stats and sample_households,
        infer which UNIFIED_CATEGORIES actually have data
        
        Returns:
            Set[str]: Set of categories actually included in the data
        """
        available_categories: Set[str] = set()
        
        # Check fields in cluster_stats
        stats_keys = set(cluster_stats.keys())
        
        # Check column names in sample_households
        if sample_households is not None and len(sample_households) > 0:
            sample_cols = set(sample_households.columns)
        else:
            sample_cols = set()
        
        all_fields = {str(k) for k in (stats_keys | sample_cols)}

        # Prefer explicit field categories when available (from extracted_fields_by_agent.json)
        if field_metadata:
            base_fields: Set[str] = set()
            for f in all_fields:
                base = f
                for suffix in ("_mean", "_median", "_mode"):
                    if base.endswith(suffix):
                        base = base[: -len(suffix)]
                        break
                base_fields.add(base)

            for base in base_fields:
                meta = field_metadata.get(base)
                cat = meta.get("category") if isinstance(meta, dict) else None
                if isinstance(cat, str) and cat:
                    available_categories.add(cat)
        
        # Infer categories based on field names (simple heuristic rules)
        
        # demographics: age, sex, marital, family, household
        if any(any(token in f.lower() for token in ["age", "sex", "gender", "marit", "race", "ethnic", "hhsize", "family", "child"]) for f in all_fields):
            available_categories.add('demographics')
        
        # income related
        if any(any(token in f.lower() for token in ["income", "inc", "earn", "wage", "salary", "poverty"]) for f in all_fields):
            available_categories.add('income')
        
        # consumption related
        if any('share' in k.lower() or 'consumption' in k.lower() or 'spending' in k.lower() 
               for k in all_fields):
            available_categories.add('consumption')
        
        # education_investment相关
        if any('education' in k.lower() or 'student_debt' in k.lower() or 'tuition' in k.lower()
               for k in all_fields):
            available_categories.add('education_investment')
        
        # assets相关
        if any('wealth' in k.lower() or 'asset' in k.lower() or 'stock' in k.lower() or 'property' in k.lower()
               for k in all_fields):
            available_categories.add('assets')
        
        # debt相关
        if any('debt' in k.lower() or 'mortgage' in k.lower() or 'loan' in k.lower()
               for k in all_fields):
            available_categories.add('debt')
        
        # employment相关
        if any('employ' in k.lower() or 'job' in k.lower() or 'work' in k.lower() or 'occupation' in k.lower()
               for k in all_fields):
            available_categories.add('employment_behavior')
        
        # financial_behavior related (risk preference, savings rate, etc.)
        if any('risk' in k.lower() or 'savings' in k.lower() or 'budget' in k.lower() or 'planning' in k.lower() or 'invest' in k.lower() or 'financial_health' in k.lower() for k in all_fields):
            available_categories.add('financial_behavior')
        
        # housing related
        if any('housing' in k.lower() or 'house' in k.lower() or 'home' in k.lower() or 'rent' in k.lower()
               for k in all_fields):
            available_categories.add('housing_behavior')
        
        # health related
        if any('health' in k.lower() or 'medical' in k.lower() or 'insurance' in k.lower()
               for k in all_fields):
            available_categories.add('health_behavior')
        
        # time_use_lifestyle related
        if any('work_hours' in k.lower() or 'commute' in k.lower() or 'leisure' in k.lower()
               for k in all_fields):
            available_categories.add('time_use_lifestyle')
        
        # attitudes_expectations related
        if any('satisfaction' in k.lower() or 'expectation' in k.lower() or 'attitude' in k.lower()
               for k in all_fields):
            available_categories.add('attitudes_expectations')
        
        # program_participation related
        if any(
            any(token in k.lower() for token in ['snap', 'medicaid', 'medicare', 'ssi', 'tanf', 'wic', 'unemployment', 'program', 'benefit'])
            for k in all_fields
        ):
            available_categories.add('program_participation')

        return available_categories
    
    def _build_dataset_context_note(self) -> str:
        """Dataset-aware guidance so prompts adapt across PSID/ACS/CES/SHED/SIPP datasets  """
        ctx = self.dataset_context or {}
        dataset = ctx.get("dataset") or ctx.get("dataset_name") or ""
        full_name = ctx.get("full_name") or ""
        focus = ctx.get("dataset_focus") or ""
        special = (ctx.get("special_instructions") or "").strip()
        
        if not (dataset or full_name or focus or special):
            return ""
        
        note: List[str] = []
        note.append("===============================================================================")
        note.append("                                DATASET CONTEXT                                ")
        note.append("===============================================================================")
        if dataset or full_name:
            note.append(f"Dataset: {full_name or dataset} ({dataset})".strip())
        if focus:
            note.append(f"Dataset focus: {focus}")
        if special:
            note.append("Dataset-specific guidance:")
            note.append(special)
        note.append(
            "\nIMPORTANT: Keep analysis aligned to this dataset's strengths; do not force PSID-style metrics "
            "onto other datasets."
        )
        return "\n".join(note)
    
    def _build_field_dictionary_note(self, cluster_stats: Dict[str, Any], max_fields: int = 25) -> str:
        """Attach variable descriptions (when available) to help interpret coded columns like I40/B2/etc."""
        if not self.field_metadata:
            return ""
        
        keys = [k for k in cluster_stats.keys() if isinstance(k, str)]
        bases: List[str] = []
        for k in keys:
            base = k
            for suffix in ("_mean", "_median", "_mode"):
                if base.endswith(suffix):
                    base = base[: -len(suffix)]
                    break
            bases.append(base)
        
        seen: Set[str] = set()
        entries: List[str] = []
        for base in bases:
            if base in seen:
                continue
            seen.add(base)
            meta = self.field_metadata.get(base)
            if not isinstance(meta, dict):
                continue
            short = meta.get("short_label") or meta.get("short_description") or ""
            desc = meta.get("description") or meta.get("var_name") or ""
            cat = meta.get("category") or ""
            if not desc and not cat:
                continue
            line = f"- {base}"
            if cat:
                line += f" [{cat}]"
            shown = short or desc
            if shown:
                line += f": {shown}"
            # If we have both, append the longer description as extra context (kept short by max_fields cap).
            if short and desc and str(short).strip().lower() != str(desc).strip().lower():
                line += f" — {desc}"
            entries.append(line)
            if len(entries) >= max_fields:
                break
        
        if not entries:
            return ""
        
        return "\n".join(
            [
                "",
                "Field Dictionary (selected variables present in the cluster profile):",
                *entries,
                "",
                "NOTE: Use the dictionary to interpret coded variables; do not guess meanings beyond what is shown.",
            ]
        )

    def _get_field_meta(self, field: str) -> Optional[Dict[str, Any]]:
        meta = self.field_metadata.get(field) if isinstance(self.field_metadata, dict) else None
        return meta if isinstance(meta, dict) else None

    def _format_field_reference(self, field: str, *, max_label_len: int = 80) -> str:
        """Format a coded field name into '[category] short_label (VAR_ID)' when metadata is available."""
        meta = self._get_field_meta(field)
        if not meta:
            return field

        cat = str(meta.get("category") or "").strip()
        short = str(meta.get("short_label") or meta.get("short_description") or "").strip()
        desc = str(meta.get("description") or meta.get("var_name") or "").strip()

        label = short or desc
        label = re.sub(r"\s+", " ", label).strip()
        if max_label_len and len(label) > max_label_len:
            label = label[: max_label_len - 3] + "..."

        parts: List[str] = []
        if cat:
            parts.append(f"[{cat}]")
        if label:
            parts.append(f"{label} ({field})")
        else:
            parts.append(field)
        return " ".join(parts)

    @staticmethod
    def _extract_feature_metrics(
        cluster_stats: Dict[str, Any],
    ) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
        """Extract per-field metrics from keys like VAR_mean/VAR_std/VAR_mode, preserving first-seen order."""
        order: List[str] = []
        metrics: Dict[str, Dict[str, Any]] = {}
        for key, val in cluster_stats.items():
            if not isinstance(key, str):
                continue
            for suffix in ("_mean", "_std", "_median", "_mode"):
                if key.endswith(suffix):
                    base = key[: -len(suffix)].strip()
                    metric = suffix[1:]  # strip leading underscore
                    if base and base not in metrics:
                        metrics[base] = {}
                        order.append(base)
                    if base:
                        metrics[base][metric] = val
                    break
        return order, metrics

    def _build_statistics_readable(self, cluster_stats: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build a compact, human-readable view of `cluster_stats` using field metadata."""
        order, metrics = self._extract_feature_metrics(cluster_stats)
        out: List[Dict[str, Any]] = []
        for base in order:
            meta = self._get_field_meta(base) or {}
            short = str(meta.get("short_label") or meta.get("short_description") or "").strip()
            desc = str(meta.get("description") or meta.get("var_name") or "").strip()
            cat = str(meta.get("category") or "").strip()

            # Keep descriptions compact to avoid bloating persona JSON.
            desc = re.sub(r"\s+", " ", desc).strip()
            if len(desc) > 240:
                desc = desc[:237] + "..."

            out.append(
                {
                    "field": base,
                    "category": cat,
                    "short_label": short,
                    "description": desc,
                    "metrics": metrics.get(base, {}),
                }
            )
        return out
    
    def _build_generic_cluster_stats_summary(
        self, cluster_stats: Dict[str, Any], max_modes: int = 18, max_cont: int = 10
    ) -> str:
        """Generic summary for non-PSID datasets (e.g., SHED/ACS/CES/SIPP cluster_profiles.csv outputs)."""
        def _fmt(val: Any) -> str:
            try:
                if val is None:
                    return "N/A"
                if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
                    return "N/A"
            except Exception:
                pass

            if isinstance(val, (int, float)):
                try:
                    return f"{float(val):.4g}"
                except Exception:
                    return str(val)
            return str(val)

        lines: List[str] = []
        lines.append("================================================================================")
        lines.append("                         CLUSTER STATISTICS SUMMARY                             ")
        lines.append("================================================================================")
        
        n = cluster_stats.get("n_samples", cluster_stats.get("n_households", cluster_stats.get("n", None)))
        pct = cluster_stats.get("percentage", cluster_stats.get("pct", None))
        if n is not None:
            lines.append(f"Sample Size: {n}")
        if pct is not None:
            try:
                lines.append(f"Cluster Share: {float(pct):.1f}%")
            except Exception:
                lines.append(f"Cluster Share: {pct}")
        
        order, metrics_by_base = self._extract_feature_metrics(cluster_stats)

        mode_bases = [b for b in order if isinstance(metrics_by_base.get(b), dict) and "mode" in metrics_by_base[b]]
        if mode_bases:
            lines.append("\nMost common category values (modes):")
            for base in mode_bases[:max_modes]:
                label = self._format_field_reference(base)
                lines.append(f"- {label}: mode={_fmt(metrics_by_base[base].get('mode'))}")
        
        cont_bases = [
            b
            for b in order
            if isinstance(metrics_by_base.get(b), dict)
            and (("mean" in metrics_by_base[b]) or ("median" in metrics_by_base[b]))
        ]
        if cont_bases:
            lines.append("\nNumeric feature aggregates:")
            for base in cont_bases[:max_cont]:
                label = self._format_field_reference(base)
                parts: List[str] = []
                m = metrics_by_base.get(base) or {}
                if "mean" in m:
                    parts.append(f"mean={_fmt(m.get('mean'))}")
                if "std" in m:
                    parts.append(f"std={_fmt(m.get('std'))}")
                if "median" in m:
                    parts.append(f"median={_fmt(m.get('median'))}")
                if parts:
                    lines.append(f"- {label}: {', '.join(parts)}")
        
        return "\n".join(lines)
    
    def _build_decision_context_guidance(self, available_categories: Set[str]) -> str:
        """Dataset- and category-aware decision contexts to focus the analysis."""
        ctx = self.dataset_context or {}
        dataset_key = (ctx.get("dataset") or ctx.get("dataset_name") or "").lower().strip()
        
        contexts_by_dataset: Dict[str, List[Tuple[str, str]]] = {
            "psid": [
                ("housing_behavior", "Housing: rent vs own, moving, mortgage decisions"),
                ("employment_behavior", "Work: job changes, stability vs opportunity, hours trade-offs"),
                ("education_investment", "Education: human capital investment for self/children"),
                ("assets", "Wealth: saving, portfolio allocation, retirement planning"),
                ("debt", "Debt: leverage tolerance, repayment priorities"),
            ],
            "shed": [
                ("financial_behavior", "Day-to-day money management: budgeting, bill payment, liquidity"),
                ("debt", "Credit/borrowing: credit card behavior, access, repayment vs rollover"),
                ("attitudes_expectations", "Subjective outlook: financial well-being, confidence, expectations"),
                ("income", "Income stability: buffers, volatility coping strategies"),
                ("program_participation", "Safety net: awareness and use of assistance"),
            ],
            "acs": [
                ("housing_behavior", "Housing & location: tenure, cost burden, mobility"),
                ("employment_behavior", "Labor supply: work participation, commuting, occupation/industry"),
                ("demographics", "Household structure: life stage, family composition, migration"),
                ("income", "Economic position: wages/income brackets and constraints"),
            ],
            "ces": [
                ("consumption", "Spending allocation: necessities vs discretionary, category trade-offs"),
                ("income", "Budget constraints: price sensitivity, substitution patterns"),
                ("debt", "Debt/smoothing: using credit to manage consumption"),
                ("assets", "Savings buffers: precautionary saving and liquidity"),
            ],
            "sipp": [
                ("program_participation", "Program take-up: benefit navigation and reliance"),
                ("income", "Income dynamics: shocks, volatility, recovery paths"),
                ("employment_behavior", "Employment transitions: job loss, re-entry, stability"),
                ("health_behavior", "Health coverage: insurance decisions and constraints"),
            ],
        }
        
        candidates = contexts_by_dataset.get(dataset_key, [])
        filtered = [text for cat, text in candidates if not cat or cat in available_categories]
        if not filtered:
            return ""
        
        return "\n".join(
            [
                "",
                "Key decision contexts to focus on (based on dataset & available variables):",
                *[f"- {t}" for t in filtered],
            ]
        )
    
    def _build_dynamic_json_structure(self, available_categories: Set[str]) -> str:
        """
        Dynamically generate JSON structure template based on available data categories
        
        Only include fields supported by data, avoid LLM speculating about missing data
        """
        # Base structure (always included)
        base_structure = {
            "persona_name": "...",
            "core_characteristics": "...",
            "behavior_patterns": ["...", "..."],
            "pain_points": ["...", "..."],
            "opportunities": ["...", "..."],
        }
        
        # Decision logic part: dynamically add based on available categories
        decision_logic = {}
        
        # Financial decisions: need income, assets, debt, etc. categories
        if available_categories & {"income", "assets", "debt", "financial_behavior"}:
            decision_logic["financial_decisions"] = {
                "core_logic": "...",
                "priorities": ["..."],
                "key_constraints": ["..."],
                "mental_models": ["..."]
            }
        
        # Consumption decisions: need consumption, housing_behavior, etc. categories
        if available_categories & {"consumption", "housing_behavior", "income"}:
            decision_logic["consumption_decisions"] = {
                "core_logic": "...",
                "spending_drivers": ["..."],
                "value_evaluation": "...",
                "priorities": ["..."]
            }
        
        # Investment decisions: need assets, investments, etc. categories (ACS does not have these)
        if available_categories & {"assets", "investments", "retirement_savings"}:
            decision_logic["investment_decisions"] = {
                "core_logic": "...",
                "risk_approach": "...",
                "time_horizon": "...",
                "investment_drivers": ["..."]
            }
        
        # Employment decisions: need employment_behavior category
        if "employment_behavior" in available_categories:
            decision_logic["employment_decisions"] = {
                "core_logic": "...",
                "job_priorities": ["..."],
                "trade_offs": ["..."]
            }
        
        # Housing decisions: need housing_behavior category
        if "housing_behavior" in available_categories:
            decision_logic["housing_decisions"] = {
                "core_logic": "...",
                "priorities": ["..."],
                "constraints": ["..."]
            }
        
        # If no decision categories, at least keep one generic
        if not decision_logic:
            decision_logic["general_decisions"] = {
                "core_logic": "...",
                "priorities": ["..."],
                "constraints": ["..."]
            }
        
        base_structure["decision_making_logic"] = decision_logic
        
        # Behavioral drivers (always included)
        base_structure["behavioral_drivers"] = {
            "core_constraints": ["..."],
            "mental_models": ["..."],
            "behavioral_triggers": ["..."]
        }
        
        # Behavioral biases: decide whether to include based on whether there are attitudes/expectations categories
        if available_categories & {"attitudes_expectations", "financial_behavior", "assets", "debt"}:
            base_structure["behavioral_biases"] = {
                "loss_aversion": "...",
                "present_bias": "...",
                "status_quo_bias": "..."
            }
        
        # Format as JSON string
        import json
        return json.dumps(base_structure, indent=4, ensure_ascii=False)
    
    def _build_persona_prompt(self, cluster_stats: Dict, sample_households: pd.DataFrame) -> str:
        """
        Build Persona generation prompt (Dynamic version)
        
        Dynamically build prompt based on the field categories actually included in the data
        """
        
        # Detect available data categories
        available_categories = self._detect_available_categories(
            cluster_stats, sample_households, field_metadata=self.field_metadata
        )

        # Shared dynamic guidance (works across datasets)
        behavior_guidance = self._build_behavior_patterns_guidance(available_categories)
        data_dimensions_note = self._build_data_dimensions_note(available_categories)
        dataset_context_note = self._build_dataset_context_note()
        field_dictionary_note = self._build_field_dictionary_note(cluster_stats)
        decision_context_note = self._build_decision_context_guidance(available_categories)

        # Build prompt purely from the fields present in `cluster_stats` (agent-selected; not fixed).
        stats_summary = self._build_generic_cluster_stats_summary(cluster_stats)

        if len(sample_households) > 0:
            sample_summary = self._summarize_sample_households(sample_households)
            if sample_summary:
                stats_summary += f"\n\nRepresentative Household Characteristics:\n{sample_summary}"

        # Dynamically generate JSON structure based on available categories
        json_structure = self._build_dynamic_json_structure(available_categories)
        
        prompt = f"""{dataset_context_note}
Based on the following cluster statistics, create a vivid user Persona AND deeply analyze the CORE DECISION-MAKING LOGIC and MENTAL MODELS of THIS specific group.

{stats_summary}

{field_dictionary_note}
{decision_context_note}
{data_dimensions_note}

IMPORTANT RULES:
- Use ONLY the information available in the cluster profile (and field dictionary if provided).
- Do NOT speculate or infer information that is not directly supported by the data.
- If a field/section has no supporting data, simply omit it or return empty string/list.
- Keep the persona vivid and realistic, but grounded in evidence.
- Output MUST be a single valid JSON object only (no markdown, no ``` fences, no commentary).

Please generate a comprehensive analysis with the following sections:

Section 1. Persona Name: Give this persona a concise, descriptive name

Section 2. Core Characteristics (2 to 3 sentences): Summarize essential features concisely

Section 3. Behavior Patterns (3 to 5 key points): Typical behavioral characteristics of this persona based on available data
{behavior_guidance}

Section 4. Pain Points and Needs (3 to 4 key points): Main challenges and unmet needs faced by this persona

Section 5. Opportunity Insights (2 to 3 key points): Product or service opportunities targeting this persona

Section 6. Core Decision-Making Logic: Analyze the fundamental decision-making logic ONLY for areas where you have supporting data

Section 7. Behavioral Drivers and Mental Models: Underlying psychological and structural factors that shape behavior

Please output in JSON format with the following structure (ONLY include sections supported by data):
{json_structure}
"""

        return prompt

    def _summarize_sample_households(self, sample_households: pd.DataFrame) -> str:
        """Summarize sample household characteristics"""
        summary_parts = []
        
        # Age distribution
        if 'age_group' in sample_households.columns:
            age_dist = sample_households['age_group'].value_counts()
            summary_parts.append(f"Age Distribution: {age_dist.to_dict()}")
        
        # Marital status
        if 'marital_status' in sample_households.columns:
            marital_dist = sample_households['marital_status'].value_counts()
            summary_parts.append(f"Marital Status: {marital_dist.to_dict()}")
        
        # Education level
        if 'education_level' in sample_households.columns:
            edu_dist = sample_households['education_level'].value_counts()
            summary_parts.append(f"Education Level: {edu_dist.to_dict()}")
        
        # Family type
        if 'family_type' in sample_households.columns:
            family_dist = sample_households['family_type'].value_counts()
            summary_parts.append(f"Family Type: {family_dist.to_dict()}")
        
        return "\n".join(summary_parts)
    
    def _parse_persona_response(self, response: str, cluster_stats: Dict) -> Dict:
        """Parse LLM response and extract Persona information"""
        def _ensure_schema(persona_obj: Dict[str, Any]) -> Dict[str, Any]:
            """
            Only ensure the basic fields exist, do not forcefully complete all decision types
            LLM generates what it wants to keep, no extra filling
            """
            # Basic fields (must exist)
            persona_obj.setdefault("persona_name", "")
            persona_obj.setdefault("core_characteristics", "")
            persona_obj.setdefault("behavior_patterns", [])
            persona_obj.setdefault("pain_points", [])
            persona_obj.setdefault("opportunities", [])
            
            # decision_making_logic: only ensure it is a dictionary, do not forcefully add subtypes
            if "decision_making_logic" not in persona_obj:
                persona_obj["decision_making_logic"] = {}
            elif not isinstance(persona_obj["decision_making_logic"], dict):
                persona_obj["decision_making_logic"] = {}
            
            # behavioral_drivers: only ensure it is a dictionary
            if "behavioral_drivers" not in persona_obj:
                persona_obj["behavioral_drivers"] = {}
            elif not isinstance(persona_obj["behavioral_drivers"], dict):
                persona_obj["behavioral_drivers"] = {}
            
            # behavioral_biases: optional, do not forcefully add
            # If LLM does not generate, do not add
            
            return persona_obj

        def _extract_json_substring(text: str) -> Optional[str]:
            if not isinstance(text, str) or not text.strip():
                return None

            t = text.strip()

            # 1) Prefer fenced code blocks anywhere in the response (```json)  
            fence_match = re.search(r"```(?:json)?\\s*(.*?)\\s*```", t, flags=re.IGNORECASE | re.DOTALL)
            if fence_match:
                t = fence_match.group(1).strip()

            # 2) Extract first balanced JSON object/array from the remaining text
            start_candidates = [i for i in (t.find("{"), t.find("[")) if i != -1]
            if not start_candidates:
                return None

            start = min(start_candidates)
            open_ch = t[start]
            close_ch = "}" if open_ch == "{" else "]"

            depth = 0
            in_str = False
            escape = False

            for idx in range(start, len(t)):
                ch = t[idx]
                if in_str:
                    if escape:
                        escape = False
                        continue
                    if ch == "\\\\":
                        escape = True
                        continue
                    if ch == "\"":
                        in_str = False
                    continue

                if ch == "\"":
                    in_str = True
                    continue

                if ch == open_ch:
                    depth += 1
                elif ch == close_ch:
                    depth -= 1
                    if depth == 0:
                        return t[start : idx + 1].strip()

            return None

        try:
            response_clean = response.strip() if isinstance(response, str) else str(response)

            # Try direct JSON first
            try:
                persona = json.loads(response_clean)
            except json.JSONDecodeError:
                candidate = _extract_json_substring(response_clean)
                if not candidate:
                    raise
                persona = json.loads(candidate)

            if not isinstance(persona, dict):
                raise json.JSONDecodeError("Persona JSON root is not an object", response_clean, 0)

            persona = _ensure_schema(persona)
            
            # Add statistics data
            persona['statistics'] = cluster_stats
            
            return persona
            
        except json.JSONDecodeError:
            # If parsing fails, return original text
            return {
                "persona_name": "Unnamed Group",
                "core_characteristics": response[:200],
                "behavior_patterns": [],
                "pain_points": [],
                "opportunities": [],
                "decision_making_logic": {
                    "financial_decisions": {
                        "core_logic": "",
                        "priorities": [],
                        "key_constraints": [],
                        "mental_models": []
                    },
                    "consumption_decisions": {
                        "core_logic": "",
                        "spending_drivers": [],
                        "value_evaluation": "",
                        "priorities": []
                    },
                    "investment_decisions": {
                        "core_logic": "",
                        "risk_approach": "",
                        "time_horizon": "",
                        "investment_drivers": []
                    }
                },
                "behavioral_drivers": {
                    "core_constraints": [],
                    "mental_models": [],
                    "behavioral_triggers": []
                },
                "behavioral_biases": {
                    "loss_aversion": "",
                    "present_bias": "",
                    "status_quo_bias": "",
                    "overconfidence": "",
                    "mental_accounting": "",
                    "ambiguity_aversion": "",
                    "social_comparison": ""
                },
                "statistics": cluster_stats,
                "parse_error": True
            }


class PersonaGenerationEngine:
    """
    Persona generation engine main class
    
    Integrate all components, coordinate Persona generation process
    Each persona and behavioral pattern is generated by the same agent
    """
    
    def __init__(self, model):
        """
        Initialize Persona generation engine
        
        Args:
            model: LLM model instance (e.g., DashScopeChatModel, OpenAIChatModel)
        """
        self.model = model
        self.persona_generator = None
    
    @staticmethod
    def _infer_dataset_key(*paths: Optional[str]) -> Optional[str]:
        """Infer dataset key from file paths/output dir names."""
        candidates = ["psid", "acs", "ces", "shed", "sipp"]
        hay = " ".join([str(p).lower() for p in paths if p])
        for key in candidates:
            if key in hay:
                return key
        return None
    
    @staticmethod
    def _default_paths_for_dataset(dataset: str) -> Dict[str, str]:
        """Default IO paths matching household_clustering_agent.py outputs (result/{dataset}_clustering_output)"""
        dataset = (dataset or "").lower().strip()
        base = f"result/{dataset}_clustering_output"
        defaults = {
            "psid": {
                "cluster_profiles_path": "result/psid_clustering_output/cluster_profiles.csv",
                "household_data_path": "result/psid_clustering_output/data_with_clusters.csv",
                "output_dir": "result/psid_clustering_output",
            },
            "acs": {
                "cluster_profiles_path": "result/acs_clustering_output/cluster_profiles.csv",
                "household_data_path": "result/acs_clustering_output/data_with_clusters.csv",
                "output_dir": "result/acs_clustering_output",
            },
            "ces": {
                "cluster_profiles_path": "result/ces_clustering_output/cluster_profiles.csv",
                "household_data_path": "result/ces_clustering_output/data_with_clusters.csv",
                "output_dir": "result/ces_clustering_output",
            },
            "shed": {
                "cluster_profiles_path": "result/shed_clustering_output/cluster_profiles.csv",
                "household_data_path": "result/shed_clustering_output/data_with_clusters.csv",
                "output_dir": "result/shed_clustering_output",
            },
            "sipp": {
                "cluster_profiles_path": "result/sipp_clustering_output/cluster_profiles.csv",
                "household_data_path": "result/sipp_clustering_output/data_with_clusters.csv",
                "output_dir": "result/sipp_clustering_output",
            },
        }
        return defaults.get(dataset, {"cluster_profiles_path": f"{base}/cluster_profiles.csv", "household_data_path": f"{base}/data_with_clusters.csv", "output_dir": base})
    
    @staticmethod
    def _load_dataset_context(dataset: str) -> Dict[str, Any]:
        """Load dataset focus/instructions (best-effort) from field_extraction_config_custom.py"""
        dataset = (dataset or "").lower().strip()
        ctx: Dict[str, Any] = {"dataset": dataset}

        try:
            from field_extraction_config_custom import get_custom_config

            config = get_custom_config(dataset)
            ctx.update(
                {
                    "dataset_name": getattr(config, "dataset_name", dataset),
                    "full_name": getattr(config, "full_name", ""),
                    "dataset_focus": getattr(config, "dataset_focus", ""),
                    "special_instructions": getattr(config, "special_instructions", ""),
                }
            )
        except Exception:
            # Minimal fallback
            ctx.setdefault("dataset_name", dataset)
        return ctx
    
    @staticmethod
    def _load_field_metadata(dataset: str, fields_metadata_path: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """Load var_id -> metadata mapping from extracted_fields_by_agent.json (best-effort) from psid_field_extraction_agent.py"""
        dataset = (dataset or "").lower().strip()
        path = fields_metadata_path or f"data/{dataset}/extracted_fields_by_agent.json"
        if not path or not os.path.exists(path):
            return {}

        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            return {}

        # common formats: List[Dict] or {selected_fields:[...]}
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
    
    def initialize_agents(
        self,
        dataset_context: Optional[Dict[str, Any]] = None,
        field_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        """Initialize agents"""
        
        # Create Persona generation agent (including pattern analysis functionality)
        self.persona_generator = PersonaGeneratorAgent(
            name="PersonaGenerator",
            model=self.model,
            dataset_context=dataset_context,
            field_metadata=field_metadata,
        )
        print("✓ Persona generation agent created (including pattern analysis functionality)")
    
    def load_clustering_results(
        self,
        cluster_profiles_path: str,
        household_data_path: str
    ) -> tuple:
        """
        Load clustering results from household_clustering_agent.py
        
        Args:
            cluster_profiles_path: cluster profiles file path from household_clustering_agent.py
            household_data_path: household data file path from household_clustering_agent.py
            
        Returns:
            tuple: (cluster_profiles, household_data)
        """
    
        cluster_profiles = pd.read_csv(cluster_profiles_path)
        print(f"✓ Loaded cluster profiles: {len(cluster profiles)} clusters")
        
        household_data = pd.read_csv(household_data_path)
        print(f"✓ Loaded household data: {len(household_data)} households")
        
        return cluster_profiles, household_data
    
    async def generate_all_personas(
        self,
        cluster_profiles: pd.DataFrame,
        household_data: pd.DataFrame,
        sample_size: Optional[int] = 50
    ) -> List[Dict]:
        """
        Generate all clusters' personas
        
        Args:
            cluster_profiles: cluster profiles data from household_clustering_agent.py
            household_data: household data from household_clustering_agent.py
            sample_size: maximum sample size for each cluster to generate Persona (None/<=0 means no sampling, use all households)
            
        Returns:
            List[Dict]: List of all personas
        """
        
        all_personas = []

        if "cluster" not in household_data.columns:
            raise ValueError("household_data is missing 'cluster' column, cannot generate personas by clusters")

        # Robust matching: try numeric labels first, fall back to string match.
        household_cluster_raw = household_data["cluster"]
        household_cluster_num = pd.to_numeric(household_cluster_raw, errors="coerce")
        has_numeric_cluster_labels = household_cluster_num.notna().any()
        
        for idx, row in cluster_profiles.iterrows():
            cluster_id = row['cluster_id']
            print(f"\nProcessing cluster {cluster_id}...")
            
            # Get cluster statistics
            cluster_stats = row.to_dict()
            
            # Get sample households for this cluster
            cluster_id_num = pd.to_numeric(pd.Series([cluster_id]), errors="coerce").iloc[0]
            if has_numeric_cluster_labels and pd.notna(cluster_id_num):
                cluster_households = household_data[household_cluster_num == cluster_id_num]
            else:
                cluster_households = household_data[household_cluster_raw.astype(str) == str(cluster_id)]

            cluster_total = len(cluster_households)
            
            # Random sampling
            if sample_size is None or (isinstance(sample_size, int) and sample_size <= 0):
                sample_households = cluster_households
            elif cluster_total > sample_size:
                sample_households = cluster_households.sample(int(sample_size), random_state=42)
            else:
                sample_households = cluster_households
            
            # Note: "sample" here is used to generate LLM prompt, not equal to the total number of households in the cluster.
            if sample_size is None or (isinstance(sample_size, int) and sample_size <= 0):
                print(f"  Total households in cluster: {cluster_total} | Sample households for Persona: {len(sample_households)} (no sampling)")
            else:
                print(f"  Total households in cluster: {cluster_total} | Sample households for Persona: {len(sample_households)}/{cluster_total} (sample_size={sample_size})")
            
            # Generate Persona (need await asynchronous call)
            persona = await self.persona_generator.generate_persona(
                cluster_stats=cluster_stats,
                sample_households=sample_households
            )
            
            persona['cluster_id'] = int(cluster_id)
            all_personas.append(persona)
            
            print(f"  ✓ Persona generated: {persona.get('persona_name', 'Unnamed')}")
        
        return all_personas
    
    def save_results(
        self,
        all_personas: List[Dict],
        output_dir: str = "result"
    ):
        """
        Save results
        
        Args:
            all_personas: All personas (including their behavioral patterns)
            output_dir: Output directory
        """
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save Personas JSON (including each cluster's behavioral patterns)
        personas_path = os.path.join(output_dir, "personas.json")
        with open(personas_path, 'w', encoding='utf-8') as f:
            json.dump(all_personas, f, ensure_ascii=False, indent=2)
        print(f"✓ Personas saved: {personas_path}")

    
    
    async def run(
        self,
        cluster_profiles_path: Optional[str] = None,
        household_data_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        dataset: Optional[str] = None,
        fields_metadata_path: Optional[str] = None,
        sample_size: Optional[int] = 50,
    ):
        """
        Run complete Persona generation process
        
        Args:
            cluster_profiles_path: cluster profiles file path from household_clustering_agent.py
            household_data_path: household data file path from household_clustering_agent.py
            output_dir: Output directory
        """
        print("\n" + "="*60)
        print("LLM-powered Persona and behavioral pattern generation engine")
        print("="*60)
        
        # Initialize agents
        dataset_key = (
            dataset
            or self._infer_dataset_key(cluster_profiles_path, household_data_path, output_dir)
            or "psid"
        ).lower().strip()

        defaults = self._default_paths_for_dataset(dataset_key)
        cluster_profiles_path = cluster_profiles_path or defaults["cluster_profiles_path"]
        household_data_path = household_data_path or defaults["household_data_path"]
        output_dir = output_dir or defaults["output_dir"]

        dataset_context = self._load_dataset_context(dataset_key)
        field_metadata = self._load_field_metadata(dataset_key, fields_metadata_path)
        self.initialize_agents(dataset_context=dataset_context, field_metadata=field_metadata)
        
        # Load data
        cluster_profiles, household_data = self.load_clustering_results(
            cluster_profiles_path,
            household_data_path
        )
        
        # Generate Personas (including their behavioral patterns) (need await asynchronous call)
        all_personas = await self.generate_all_personas(
            cluster_profiles,
            household_data,
            sample_size=sample_size,
        )
        
        # Save results
        self.save_results(
            all_personas,
            output_dir
        )
        
        print("\n" + "="*60)
        print("✓ Persona generation completed!")
        print("="*60)


def main(dataset: str = "psid", sample_size: Optional[int] = 50):
    
    import asyncio
    from llm_config import create_llm_model
    
  
    model = create_llm_model(temperature=0.7, max_tokens=2000)
    
    # Create engine using DashScopeChatModel
    engine = PersonaGenerationEngine(model=model)
    
     
    asyncio.run(engine.run(dataset=dataset, sample_size=sample_size))




