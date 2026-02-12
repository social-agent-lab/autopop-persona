"""
Using an LLM to automatically read and interpret codebook files, 
and extract the variables required for constructing personalized personas.
"""

import os
import re
import json
import pandas as pd
import asyncio
from typing import Dict, List, Tuple, Any, Optional
from agentscope.agent import AgentBase
from agentscope.message import Msg
from agentscope.memory import InMemoryMemory
from codebook_parser import CodebookParser


class PSIDFieldExtractionAgent(AgentBase):
    """
    General-Purpose Field Extraction Agent (Applicable to Multiple Survey Datasets)

    Responsibilities:
    - Read and interpret codebook files from diverse survey datasets
    - Intelligently identify fields relevant for persona construction and behavioral characterization
    - [If available] Analyze value distributions and filter out invalid fields where over 99% of values are 0 or Inap
    - [If distribution information is unavailable] Perform intelligent screening based on field names, descriptions, and categories
    - Extract fields related to basic demographics, economic structure, and behavioral characteristics
    - Categorize each field and assess its relevance
    
    """
    
    # ===================== Test/Validation Field Exclusion Lists =====================
    # These fields are used for testing and validation, and should not be extracted into persona construction to prevent data leakage
    # PSID Test Fields (10)
    PSID_TEST_FIELDS = {
        "ER84008",   
        "ER82904",   
        "ER82886",   
        "ER82459",   
        "ER84134",   
        "ER82052",   
        "ER82479",   
        "ER82444",   
        "ER84114",  
        "ER82147",   
        #Validation
        "ER84043","ER82885","ER82887","ER82890","ER82442","ER84048","ER82069","ER82160","ER82692","ER82070",
        "ER84034","ER84150","ER82958","ER85222","ER83971","ER85208","ER83872","ER82148","ER84024","ER85243"
    }
    
    # SHED 测试问题字段 (10个)
    SHED_TEST_FIELDS = {
        "K21_c",     
        "ppfs0001",  
        "E1_e",      
        "E1_d",      
        "EF6C_b",    
        "K21_f",     
        "A0",        
        "D44_c",     
        "E1_b",      
        "R11",       
        #Validation
        "EF1","pay_casheqv","K5A_b","EF5C","K5A_c","S16_a","EF3_a","DC4","INF3_f","A0B",
        "BNPL1","C3P","D44_d","D44_a","D44_e","ND4_c","K0","E1_a","E1_c","R1_c"
    }
    # ====================================================================
    
    def __init__(
        self,
        name: str,
        model,
        sys_prompt: str = None,
        memory=None,
        usability_threshold: float = 1.0,  
        category_thresholds: Dict[str, float] = None,  
        category_examples: Dict[str, List[str]] = None, 
        enable_quality_filter: bool = True,  
        dataset_focus: str = "", 
        special_instructions: str = "",  
        **kwargs
    ):
        
        super().__init__()
        
        self.name = name
        self.model = model
        self.memory = memory or InMemoryMemory()
        self.usability_threshold = usability_threshold
        self.category_examples = category_examples or {}
        self.enable_quality_filter = enable_quality_filter  
        self.dataset_focus = dataset_focus  
        self.special_instructions = special_instructions  
        
        

        # Prompt/context length safeguards (ACS has many high-cardinality code lists)
        self.max_chunk_chars = max(2000, int(kwargs.pop("max_chunk_chars", 9000)))
        self.max_code_values_head = max(0, int(kwargs.pop("max_code_values_head", 12)))
        self.max_code_values_tail = max(0, int(kwargs.pop("max_code_values_tail", 8)))
        self.max_code_value_text_chars = max(20, int(kwargs.pop("max_code_value_text_chars", 120)))
        self.max_label_chars = max(40, int(kwargs.pop("max_label_chars", 200)))
        self.max_description_chars = max(60, int(kwargs.pop("max_description_chars", 400)))
        self.include_code_values = bool(kwargs.pop("include_code_values", True))

        self.sys_prompt = sys_prompt or self._generate_sys_prompt()
    
    def _normalize_whitespace(self, text: Any) -> str:
        return re.sub(r"\s+", " ", str(text or "")).strip()

    def _truncate_text(self, text: Any, max_chars: int) -> str:
        if max_chars <= 0:
            return ""
        normalized = self._normalize_whitespace(text)
        if len(normalized) <= max_chars:
            return normalized
        if max_chars <= 3:
            return normalized[:max_chars]
        return normalized[: max_chars - 3].rstrip() + "..."

    def _format_code_values_for_prompt(self, code_values: Any) -> str:
        """Format code_values into a compact text block to avoid oversized prompts."""
        if not self.include_code_values or not code_values:
            return ""
        if not isinstance(code_values, list):
            return ""

        total = len(code_values)
        if total == 0:
            return ""

        head_n = self.max_code_values_head
        tail_n = self.max_code_values_tail
        show_n = max(0, head_n + tail_n)

        if show_n <= 0:
            return f"**Code Values**: {total} codes (omitted)\n"

        if total <= show_n:
            indices = list(range(total))
        else:
            head_end = min(head_n, total)
            tail_start = max(head_end, total - tail_n)
            indices = list(range(head_end)) + list(range(tail_start, total))

        lines: List[str] = []
        for idx in indices:
            code = code_values[idx]
            if not isinstance(code, dict):
                continue
            value = self._truncate_text(code.get("value", ""), 60)
            text = self._truncate_text(code.get("text", ""), self.max_code_value_text_chars)
            if value and text:
                lines.append(f"- {value}: {text}")
            elif value:
                lines.append(f"- {value}")
            elif text:
                lines.append(f"- {text}")

        shown = len(lines)
        omitted = max(0, total - shown)

        header = (
            f"**Code Values** ({total} total; showing {shown}):\n"
            if omitted > 0
            else f"**Code Values** ({total}):\n"
        )
        if omitted > 0:
            lines.append(f"- ... ({omitted} more omitted)")

        return header + "\n".join(lines) + "\n"

    def _generate_sys_prompt(self) -> str:
        """
        Generate system prompt dynamically based on configuration (English, adaptable to different data sources)
        """
        # Build category descriptions and examples
        category_descriptions = []
        for cat_name, threshold in sorted(self.category_thresholds.items()):
            max_fields = self.max_fields_per_category.get(cat_name, 10)
            examples = self.category_examples.get(cat_name, [])

            desc = f"- **{cat_name}** (max {max_fields} fields"
            if self.enable_quality_filter:
                desc += f", quality threshold ≥{threshold}%"
            desc += ")"

            if examples:
                examples_str = ", ".join(examples[:5])  
                desc += f"\n  Examples: {examples_str}"
            category_descriptions.append(desc)

        categories_section = "\n".join(category_descriptions)

        dataset_context = ""
        if self.dataset_focus:
            dataset_context = f"\n**Dataset Context**: {self.dataset_focus}\n"

        if self.special_instructions:
            dataset_context += f"\n{self.special_instructions}\n"

    
        if self.enable_quality_filter:
            quality_section = """
**Quality Filtering** (from JSON codebook availability_rate):
   - Each category has a **quality threshold** (minimum usable_percentage)
   - Fields below threshold will be rejected
   - Use the "availability_rate" from codebook as usable_percentage
   - availability_rate shows the percentage of valid (non-missing) responses

**How to determine usable_percentage**:
   - Check the **Availability** field in the variable description
   - This shows percentage like "99.77%" or "100.0%"
   - Use this value directly as usable_percentage (remove % symbol, keep number only)
   - If availability_rate is missing, use 100.0

   **Example from JSON codebook**:
   ```
   ## ER82003
   **Label**: PSID STATE OF RESIDENCE CODE
   **Description**: State code for residence
   **Availability**: 99.77%

   → usable_percentage = 99.77
   ```"""
        else:
            quality_section = """
**Relevance-Based Filtering** (no quality metrics needed):
   - Since no quality metrics are available, focus on field **relevance** and **importance**
   - Prioritize fields that are commonly used in behavioral/demographic analysis
   - Set usable_percentage = 100.0 for all fields (no quality filtering)
   - Filter based on field importance only"""

 
        prompt = f"""You are a behavioral economist and data analyst specialized in extracting fields that reveal household decision-making logic and behavioral patterns.

**Core Mission**: Extract fields that help us understand HOW and WHY households make decisions, not just demographic facts.

**What Makes a Good Field for Behavioral Analysis?**
Think: "Can this field help me understand this household's decision-making logic, constraints, priorities, or behavioral patterns?"

✓ Good examples:
  - Education spending → reveals future orientation and human capital investment mindset
  - Work hours + commute time → reveals time preferences and work-life trade-offs
  - Multiple debt types → reveals leverage strategy and financial planning approach
  - Health insurance coverage → reveals risk management and security mindset
  - Asset portfolio composition → reveals risk appetite and investment philosophy

✗ Poor examples (unless no better options exist):
  - Administrative codes, sample weights, technical flags
  - Redundant variations of the same concept
  - Overly granular details that don't add behavioral insight

**Task**: Extract fields suitable for understanding household behavioral patterns and decision-making logic.
{dataset_context}
**Available Categories and Constraints**:

{categories_section}

**CRITICAL - "Quality Over Quantity" Principle**:
- The `max_fields` numbers are UPPER LIMITS, not targets or goals
- **DO NOT extract low-quality fields just to reach the limit**
- If a category has NO relevant fields in this dataset, return EMPTY array for that category
- If a category has only 3 high-quality fields, extract those 3 ONLY (don't add low-quality ones to reach max)
- If a category has fewer fields than the max, extract ALL relevant HIGH-QUALITY fields only
- If a category has MORE fields than the max, select the most behaviorally insightful ones up to the limit
- **Quality Threshold**: Only extract fields that genuinely reveal behavioral patterns or decision-making logic
- **Example 1**: If ACS has NO consumption data, return [] for consumption - don't try to force it
- **Example 2**: If a category has max_fields=10 but only 4 fields are truly insightful, return 4 fields ONLY

**Extraction Principles (CRITICAL - Avoid Redundancy & Control Context Length)**:

1. **Prioritize Summary/Aggregate Fields**: Extract total income rather than all income subcategories; extract total assets rather than every asset type.

2. **Avoid Duplicate Characterization**:
   - If "TOTAL HOUSEHOLD INCOME" exists, skip individual income sources unless they provide unique insights
   - If "AGE OF REFERENCE PERSON" exists, skip "AGE OF HOUSEHOLD HEAD" unless clearly different
   - If "# IN FU (family size)" exists, skip "HOUSEHOLD SIZE" if redundant
   - Extract reference person fields first; spouse/other member fields are optional

3. **Field Relevance and Importance** (critically important - BEHAVIORAL LENS):
   - Only extract fields that **reveal behavioral patterns, decision-making logic, or key constraints**
   - Skip technical/administrative fields (sample weights, flags, internal codes)
   - Ask yourself: "Does this field help me understand HOW this household thinks and decides?"
   - **Assign importance_score (1-10)** based on **BEHAVIORAL INSIGHT POTENTIAL**:
     * **10**: Irreplaceable for understanding core behavior (e.g., risk-taking, time preferences, values)
     * **8-9**: Reveals major constraints or priorities that shape decisions (income, debt burden, family obligations)
     * **6-7**: Shows specific behavioral patterns or choices (consumption priorities, housing decisions)
     * **4-5**: Provides context for behavior but doesn't directly reveal decision logic
     * **1-3**: Demographic facts with limited behavioral insight
   - **Behavioral Insight Criteria**:
     * Does it reveal trade-offs they make? (time, money, risk)
     * Does it show their priorities? (present vs future, security vs growth)
     * Does it indicate their constraints? (financial, time, knowledge, social)
     * Does it reflect their mental models? (how they think about money, risk, life)
   - **Important**: Behavioral relevance > data quality. A lower-quality field that reveals decision logic beats a high-quality demographic field

4. **Quantity Limits** (upper bounds, NOT targets - Quality First!):
   - Each category has a **maximum field count** (this is a CEILING, not a goal)
   - **QUALITY > QUANTITY**: Better to return 3 excellent fields than 10 mediocre ones
   - If no relevant HIGH-QUALITY fields exist for a category, return [] (empty)
   - If only 5 high-quality fields exist and max=10, return those 5 ONLY (don't pad with low-quality fields)
   - If more high-quality fields exist than the limit, rank by behavioral insight and select top N
   - **Critical**: A field must justify its inclusion by providing unique behavioral insight
   - **Ask yourself**: "Does this field help me understand HOW they decide?" If not, skip it even if below the limit

{quality_section}

**Output Format (JSON ONLY, NO MARKDOWN)**:

```json
{{
  "fields": [
    {{
      "var_id": "ER82002",
      "var_name": "2023 FAMILY INTERVIEW (ID) NUMBER",
      "description": "2023 Interview Number",
      "short_label": "family interview id",
      "category": "identifier",
      "data_type": "identifier",
      "usable_percentage": 100.0,
      "importance_score": 10,
      "reasoning": "Household unique identifier - MUST extract"
    }},
    {{
      "var_id": "ER82018",
      "var_name": "AGE OF REFERENCE PERSON",
      "description": "Age of 2023 Reference Person",
      "short_label": "head age",
      "category": "demographics",
      "data_type": "continuous",
      "usable_percentage": 100.0,
      "importance_score": 9,
      "reasoning": "Reference person age - key demographic variable"
    }}
  ]
}}
```

**Required JSON Fields**:
- **var_id**: Variable identifier (code/name)
- **var_name**: Variable name (human-readable)
- **description**: MUST copy the codebook **Description** verbatim (no paraphrase)
- **short_label**: 2-4 words, lowercase; extracted/summarized from description (no year/codes)
- **category**: Must be one of: {', '.join(self.category_thresholds.keys())}
- **data_type**: Must be one of: identifier, continuous, count, binary, categorical, ordinal
- **usable_percentage**: Number (0-100), NO % symbol{' (calculate from distribution if available, else use 100.0)' if not self.enable_quality_filter else ''}
- **importance_score**: Integer (1-10), where 10=critical, 8-9=very important, 6-7=important, 4-5=useful, 1-3=optional
  * **10**: Identifier fields, unique demographic/economic indicators
  * **8-9**: Core demographics (age, gender, marital status), total income/assets/debt
  * **6-7**: Subcategory income/expenditure, employment status, education
  * **4-5**: Detailed breakdowns, secondary behaviors
  * **1-3**: Supplementary information
- **reasoning**: Explain the **BEHAVIORAL INSIGHT** this field provides - how does it help understand decision-making logic or behavioral patterns?

**CRITICAL - Importance Score Guidelines (BEHAVIORAL FOCUS)**:
- **Think Behavioral First**: "What decision-making insight does this field provide?"
- **Examples by Score**:
  * **10**: Education spending (reveals future orientation), Asset portfolio composition (reveals risk appetite)
  * **9**: Total income (constraint on choices), Debt-to-income ratio (leverage strategy)
  * **8**: Work hours (time-money trade-off), Health insurance (risk management)
  * **7**: Housing type (stability preference), Job changes (career mobility)
  * **5**: Age (life stage context), Family size (household constraint)
  * **3**: Marital status code, State of residence (limited behavioral insight alone)
- **Prioritize fields that reveal**:
  * Trade-offs they make (time vs money, present vs future, risk vs safety)
  * How they allocate scarce resources (money, time, attention)
  * What they prioritize (security, growth, family, self, status)
  * How they manage risk and uncertainty

**Data Type Guidelines**:
- **identifier**: Unique ID (exclude from clustering)
- **continuous**: Numeric values with meaningful magnitude (age, income, amount)
- **count**: Discrete counts (number of children, vehicles)
- **binary**: Yes/No, True/False, Male/Female (needs dummy coding)
- **categorical**: Multiple unordered categories (occupation, vehicle type) (needs one-hot encoding)
- **ordinal**: Ordered categories (education level, income bracket) (needs label encoding)

**Critical Rules**:
- Return ONLY pure JSON, NO markdown code blocks, NO explanatory text
- If no qualifying fields found, return {{"fields": []}}
- Strictly respect quantity limits for each category
- Prioritize non-redundant, high-impact fields
- Skip administrative/technical fields that don't characterize personas"""
        
        return prompt
    
    async def _call_model_async(self, messages: List[Dict]) -> str:
        """
        Asynchronously call the LLM model (continues retrying until successful)
    
        """
        attempt = 0
        while True:
            try:
                response = await self.model(messages)
                
                if hasattr(response, '__aiter__'):
                    content = ""
                    last_chunk = None
                    
                    async for chunk in response:
                        last_chunk = chunk
                        
                        if hasattr(chunk, 'keys') and 'content' in chunk:
                            continue
                        
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
                    
                    if content.strip():
                        return content
                    else:
                        raise ValueError("Empty response content")
                
            
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
                
                if content.strip():
                    return content
                else:
                    raise ValueError("Empty response content")
            except Exception as e:
                    attempt += 1
                    wait_time = min(2 ** min(attempt - 1, 6), 60)
                    print(f"  ⚠️ Field extraction LLM call failed (attempt {attempt}): {e}")
                    await asyncio.sleep(wait_time)
                
    
    async def reply(self, x: Msg = None) -> Msg:
        """
        Process message and extract fields
        
        """
        if x is None:
            return Msg(name=self.name, content='{"fields": []}', role="assistant")
        
        user_prompt = f"""Analyze the following variable definitions from the survey codebook and extract qualifying fields.

**CRITICAL Instructions**:
1. **"Extract What Exists" Rule**: Only extract fields that ACTUALLY EXIST and are relevant
   - If a category has NO relevant fields, return empty array for that category
   - Don't try to force extraction to meet the max_fields limit
   - max_fields is an UPPER BOUND, not a target
2. Extract only fields that meet the quality threshold for their category
3. Prioritize aggregate/summary fields over detailed subcategories
4. Avoid redundant fields that characterize the same dimension
5. Calculate usable_percentage correctly (exclude Inap, DK, NA, refused)
6. Assign correct data_type AND importance_score (1-10) for each field
7. For each extracted field, keep **description** EXACTLY the same as the codebook **Description** text (copy it verbatim, no paraphrase) and generate **short_label** (2-4 words, lowercase; summarize/extract from description only; remove years/codes)
8. Return pure JSON format with "fields" array, NO markdown, NO other text
9. If no qualifying fields found in this chunk, return {{"fields": []}}

**Variable definitions**:

{x.content}

**Response (JSON only)**:"""
        
        messages = [
            {"role": "system", "content": self.sys_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response_content = await self._call_model_async(messages)
    
        return Msg(
            name=self.name,
            content=response_content,
            role="assistant"
        )
    
    def _get_excluded_fields(self, dataset_name: str) -> set:
       
        dataset_lower = dataset_name.lower()
        if dataset_lower == "psid":
            return self.PSID_TEST_FIELDS
        elif dataset_lower == "shed":
            return self.SHED_TEST_FIELDS
        else:
            return set()
    
    def read_codebook_from_json(self, dataset_name: str, chunk_size: int = 30) -> List[str]:
        
        parser = CodebookParser()
        codebook = parser.load_dataset(dataset_name)
        
        
        excluded_fields = self._get_excluded_fields(dataset_name)
        if excluded_fields:
            print(f"   ⚠️ Exclude {len(excluded_fields)} test/validation fields to prevent data leakage")
        
       
        chunks = []
        current_chunk = []
        current_count = 0
        current_chars = 0
        excluded_count = 0
        
        for var_id, var_info in codebook.items():
            
            if var_id in excluded_fields:
                excluded_count += 1
                continue
           
            label = self._truncate_text(var_info.get("label", "N/A"), self.max_label_chars)
            description = self._truncate_text(var_info.get("description", "N/A"), self.max_description_chars)

            var_text = f"\n## {var_id}\n"
            var_text += f"**Label**: {label}\n"
            if description and self._normalize_whitespace(description) != self._normalize_whitespace(label):
                var_text += f"**Description**: {description}\n"
           
            code_values_block = self._format_code_values_for_prompt(var_info.get("code_values"))
            if code_values_block:
                var_text += "\n" + code_values_block
            
            
            if 'availability_rate' in var_info:
                var_text += f"\n**Availability**: {var_info['availability_rate']}\n"
            
            if len(var_text) > self.max_chunk_chars:
                label_min = self._truncate_text(var_info.get("label", "N/A"), min(self.max_label_chars, 120))
                desc_min = self._truncate_text(var_info.get("description", "N/A"), min(self.max_description_chars, 180))
                var_text = f"\n## {var_id}\n**Label**: {label_min}\n"
                if desc_min and self._normalize_whitespace(desc_min) != self._normalize_whitespace(label_min):
                    var_text += f"**Description**: {desc_min}\n"
                if 'availability_rate' in var_info:
                    var_text += f"\n**Availability**: {var_info['availability_rate']}\n"

            sep = 1 if current_chunk else 0
            if current_chunk and (current_chars + sep + len(var_text) > self.max_chunk_chars):
                chunks.append("\n".join(current_chunk))
                current_chunk = []
                current_count = 0
                current_chars = 0

            sep = 1 if current_chunk else 0
            current_chunk.append(var_text)
            current_count += 1
            current_chars += len(var_text) + sep
            
        
            if current_count >= chunk_size:
                chunks.append("\n".join(current_chunk))
                current_chunk = []
                current_count = 0
                current_chars = 0
        
        
        if current_chunk:
            chunks.append("\n".join(current_chunk))
        
        return chunks
    
    def extract_json_from_text(self, text: str) -> dict:
       
        
        try:
            return json.loads(text)
        except:
            pass
        
        json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except:
                pass
        
        code_match = re.search(r'```\s*(.*?)\s*```', text, re.DOTALL)
        if code_match:
            try:
                return json.loads(code_match.group(1))
            except:
                pass
        
        json_obj_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_obj_match:
            try:
                return json.loads(json_obj_match.group(0))
            except:
                pass
        
        return {"fields": []}
    
    async def extract_fields_from_chunk(self, chunk: str, chunk_index: int, total_chunks: int) -> List[Dict]:
        """
        Extract fields from a chunk (maximum 10 retries, ensuring all fields have importance_score)
    
        """
        print(f"   Processing chunk {chunk_index + 1}/{total_chunks}...", end=" ", flush=True)
        
        retry_count = 0
        max_retries = 5  # Maximum number of retries
        
        while retry_count < max_retries:
           
            msg = Msg(
                name="user",
                content=chunk,
                role="user"
            )
            
            response = await self.reply(msg)
            
            try:
                result = self.extract_json_from_text(response.content)
                
                if isinstance(result, dict) and 'fields' in result:
                    fields = result['fields']
                elif isinstance(result, list):
                    fields = result
                else:
                    fields = []
                
                if not fields:
                    retry_count += 1
                    if retry_count >= max_retries:
                        print(f"❌ Reached maximum retries ({max_retries} times), this chunk may have formatting issues, skip")
                        print(f"      💡 Suggestions: Check codebook format or adjust chunk_size")
                        return []
                    print(f"⚠️ No valid fields -> Retry {retry_count}/{max_retries} times")
                    continue
                
                missing_importance = []
                for field in fields:
                    var_id = field.get('var_id', 'unknown')
                    importance = field.get('importance_score', None)
                    if importance is None:
                        missing_importance.append(var_id)
                
                if missing_importance:
                    retry_count += 1
                    if retry_count >= max_retries:
                        valid_fields = [f for f in fields if f.get('importance_score') is not None]
                        if valid_fields:
                            return valid_fields
                        else:
                            return []
                    continue
                
                if retry_count > 0:
                    print(f"✅ Retry successful! Extracted {len(fields)} complete fields")
                else:
                    print(f"✅ Extracted {len(fields)} fields")
                
                return fields if isinstance(fields, list) else []
                
            except Exception as e:
                retry_count += 1
                if retry_count >= max_retries:
                    print(f"      💡 Suggestions: Check codebook format, LLM configuration or chunk content")
                    return []
                print(f"❌ Error: {type(e).__name__}: {str(e)}")
                print(f"      -> Retry {retry_count}/{max_retries} times")
                continue
        
        print(f"❌ Unknown reason exit retry loop, skip this chunk")
        return []
    
    async def process_codebook_dataset(self, dataset_name: str, chunk_size: int = 30) -> List[Dict]:
        """
        Process a complete dataset codebook
        
        Args:
            dataset_name: Dataset name (psid, pums, ces, shed, sipp)
            chunk_size: Number of variables per chunk
            
        Returns:
            List[Dict]: All extracted fields
        """
        chunks = self.read_codebook_from_json(dataset_name, chunk_size)
        
        all_fields = []
        for i, chunk in enumerate(chunks):
            fields = await self.extract_fields_from_chunk(chunk, i, len(chunks))
            all_fields.extend(fields)
        
        return all_fields
    
    def adaptive_allocate_fields(
        self, 
        fields: List[Dict], 
        total_target: int = 75,
        priority_weights: Dict[str, int] = None
    ) -> Tuple[List[Dict], Dict[str, int]]:
        """
        Adaptive field allocation based on Agent extraction results
        
        Strategy:
        1. First group by category, count the number of high-quality fields actually extracted for each category
        2. Allocate名额 based on the number, quality, and priority of fields in each category
        3. Keep the most important fields in each category
        
        Args:
            fields: List of original fields extracted by Agent
            total_target: Total number of fields target
            priority_weights: Category priority (1-10)
            
        Returns:
            Tuple[List[Dict], Dict[str, int]]: 
                - List of fields after adaptive allocation
                - Allocation statistics
        """
        import math

        # Default priority
        if priority_weights is None:
            priority_weights = {
                'identifier': 10,
                'demographics': 9,
                'income': 8,
                'consumption': 8,
                'assets': 8,
                'debt': 7,
                'employment_behavior': 7,
                'financial_behavior': 7,
                'housing_behavior': 6,
                'health_behavior': 6,
                'education_investment': 7,
                'time_use_lifestyle': 5,
                'attitudes_expectations': 7,
                'program_participation': 6,
                'other': 3
            }
        
        # Deduplicate and group by category
        seen_ids = set()
        categorized = {}
        
        for field in fields:
            var_id = field.get('var_id', '')
            if not var_id or var_id in seen_ids:
                continue
            
            usable_pct = field.get('usable_percentage', 100.0)
            if isinstance(usable_pct, str):
                usable_pct = float(re.sub(r'[^\d.]', '', usable_pct))
                field['usable_percentage'] = usable_pct
            
            importance = field.get('importance_score', 5)
            if isinstance(importance, str):
                try:
                    importance = float(importance)
                    field['importance_score'] = importance
                except ValueError:
                    continue
            
            if self.enable_quality_filter:
                importance_normalized = importance * 10
                composite_score = importance_normalized * 0.7 + usable_pct * 0.3
            else:
                composite_score = importance * 10
            field['composite_score'] = composite_score

            category_norm = str(field.get("category") or "").lower().strip()
            dtype_norm = str(field.get("data_type") or "").lower().strip()
            if category_norm == "identifier":
                field["data_type"] = "identifier"
            elif dtype_norm == "identifier":
                field["data_type"] = "categorical"

            category = field.get('category', 'other')
            if category not in categorized:
                categorized[category] = []
            categorized[category].append(field)
            seen_ids.add(var_id)
        
        category_stats = {}
        for category, cat_fields in categorized.items():
            count = len(cat_fields)
            avg_quality = sum(f.get('usable_percentage', 0) for f in cat_fields) / count
            avg_importance = sum(f.get('importance_score', 0) for f in cat_fields) / count
            category_stats[category] = {
                'count': count,
                'avg_quality': avg_quality,
                'avg_importance': avg_importance,
                'fields': sorted(cat_fields, key=lambda x: x.get('composite_score', 0), reverse=True)
            }
            print(f"{category:<25} | {count:<8} | {avg_quality:<10.1f}% | {avg_importance:<10.1f}")
        
        # Adaptive allocation strategy
        allocation_scores = {}
        for category, stats in category_stats.items():
            priority = priority_weights.get(category, 5)
            count = stats['count']
            avg_importance = stats['avg_importance']
            
            if category == 'identifier':
                allocation_scores[category] = 1
            else:
                score = priority * math.log(count + 1) * (avg_importance / 5)
                allocation_scores[category] = score
        
        total_score = sum(allocation_scores.values())
        allocated = {}
        remaining = total_target - 1 
        
        for category, score in allocation_scores.items():
            if category == 'identifier':
                allocated[category] = 1
            else:
                proportion = score / total_score
                alloc_count = int(remaining * proportion)
                
                max_available = category_stats[category]['count']
                alloc_count = min(alloc_count, max_available)

                if max_available > 0:
                    alloc_count = max(alloc_count, 1)

                allocated[category] = alloc_count
    
        current_total = sum(allocated.values())
        diff = total_target - current_total
        
        if diff != 0:
            adjustable = [
                (cat, priority_weights.get(cat, 5), category_stats.get(cat, {}).get("count", 0) - allocated[cat])
                for cat in allocated.keys()
                if cat != "identifier"
            ]
            
            if diff > 0:
                adjustable = [(c, p, r) for c, p, r in adjustable if r > 0]
                adjustable.sort(key=lambda x: x[1], reverse=True)
                
                idx = 0
                while diff > 0 and adjustable:
                    cat, _, remaining_capacity = adjustable[idx % len(adjustable)]
                    if remaining_capacity > 0:
                        allocated[cat] += 1
                        diff -= 1
                        adjustable[idx % len(adjustable)] = (cat, adjustable[idx % len(adjustable)][1], remaining_capacity - 1)
                    idx += 1
                    adjustable = [(c, p, r) for c, p, r in adjustable if r > 0]
            else:
                adjustable = [(c, p, allocated[c]) for c, p, _ in adjustable if allocated[c] > 0]
                adjustable.sort(key=lambda x: x[1])
                
                idx = 0
                while diff < 0 and adjustable:
                    cat, _, can_reduce = adjustable[idx % len(adjustable)]
                    if can_reduce > 0:
                        allocated[cat] -= 1
                        diff += 1
                        adjustable[idx % len(adjustable)] = (cat, adjustable[idx % len(adjustable)][1], can_reduce - 1)
                    idx += 1
                    adjustable = [(c, p, r) for c, p, r in adjustable if r > 0]
        
        allocation_info = {}
        for category in sorted(allocated.keys()):
            available = category_stats.get(category, {}).get('count', 0)
            alloc_count = allocated[category]
            priority = priority_weights.get(category, 5)
            utilization = (alloc_count / available * 100) if available > 0 else 0
            
            if alloc_count > 0:
                print(f"{category:<25} | {available:<6} | {alloc_count:<6} | {priority:<6} | {utilization:<7.1f}%")
                allocation_info[f"{category}_allocated"] = alloc_count
        
        
        final_fields = []
        for category, alloc_count in allocated.items():
            if category in category_stats:
                top_fields = category_stats[category]['fields'][:alloc_count]
                final_fields.extend(top_fields)
                
                # Record the number of filtered fields
                filtered_count = len(category_stats[category]['fields']) - alloc_count
                if filtered_count > 0:
                    allocation_info[f"{category}_filtered"] = filtered_count
        
        return final_fields, allocation_info

    async def synthesize_behavioral_insights(self, fields: List[Dict], dataset_name: str = "psid") -> str:
        """
        Synthesize the extracted fields and generate a behavioral analysis report

        Args:
            fields: List of extracted fields

        Returns:
            str: Behavioral analysis report (Markdown format)
        """
        categorized = {}
        for field in fields:
            category = field.get('category', 'other')
            if category not in categorized:
                categorized[category] = []
            categorized[category].append(field)

        summary = "# Extracted Fields Summary\n\n"
        for category in sorted(categorized.keys()):
            cat_fields = categorized[category]
            summary += f"## {category} ({len(cat_fields)} fields)\n"
            for field in cat_fields[:10]:
                summary += f"- {field['var_id']}: {field.get('var_name', '')} (usable: {field.get('usable_percentage', 0):.1f}%)\n"
            if len(cat_fields) > 10:
                summary += f"- ... and {len(cat_fields) - 10} more fields\n"
            summary += "\n"

        dataset_label = (dataset_name or "").upper() if isinstance(dataset_name, str) else ""
        fields_label = f"{dataset_label} fields" if dataset_label else "fields"

        prompt = f"""Based on the extracted {fields_label}, please summarize how to use these fields to characterize household behaviors.

{summary}

Please provide specific analytical approaches for the following dimensions:

1. **Economic Behavior Characterization**:
   - How to analyze income source diversity and stability using income fields
   - How to calculate financial health from asset and debt fields
   - How to identify saving and investment propensities

2. **Consumption Behavior Characterization**:
   - How to analyze consumption structure and priorities from expenditure fields
   - How to identify spending habits (necessities vs. luxuries)
   - How to calculate marginal propensity to consume (MPC)

3. **Employment and Work Behavior**:
   - How to assess job stability
   - How to analyze work hour preferences
   - How to identify career development trajectories

4. **Housing and Residential Behavior**:
   - How to distinguish rent/own decision patterns
   - How to assess housing investment propensity
   - How to identify moving and migration behaviors

5. **Health and Lifestyle**:
   - How to assess health awareness and behaviors
   - How to analyze health investment
   - How to identify lifestyle characteristics

6. **Comprehensive Behavioral Indicator Construction**:
   - Propose 3-5 key composite behavioral indicators
   - Explain how to calculate these indicators from multiple fields
   - Explain how to transform numbers into meaningful behavioral labels

Please provide specific calculation formulas and classification methods. Answer in Chinese with clear structure."""

        messages = [
            {"role": "system", "content": "You are an expert in sociology and economics, skilled at extracting behavioral insights from data."},
            {"role": "user", "content": prompt}
        ]

        try:
            response = await self._call_model_async(messages)
            return response
        except Exception as e:
            return "Failed to generate behavioral insights analysis"

async def run_extraction_pipeline(
    dataset_name: str,
    model,
    output_dir: str = None,
    chunk_size: int = 30,
    config=None,  # DatasetConfig object (from field_extraction_config_custom.py)
    adaptive_target: int = 75  # Total number of fields target for adaptive allocation
):
    """
    Run the complete field extraction pipeline (using JSON codebook)
    
    Args:
        dataset_name: Dataset name (psid, acs, ces, shed, sipp)
        model: AgentScope model object
        output_dir: Output directory (if None, automatically set to data/{dataset_name})
        chunk_size: Number of variables per chunk
        config: DatasetConfig object (from field_extraction_config_custom.py)
        adaptive_target: Total number of fields target for adaptive allocation
    """
    
    # Set output directory
    if output_dir is None:
        output_dir = f'data/{dataset_name}'
    
    if config:
        
        category_thresholds = config.get_quality_thresholds()
        category_examples = config.get_category_examples()
        enable_quality_filter = config.enable_quality_filter
        dataset_focus = config.dataset_focus
        special_instructions = config.special_instructions
    else:
        category_examples = None
        enable_quality_filter = True
        dataset_focus = ""
        special_instructions = ""
    
    # Determine if quality filtering is needed
    filter_mode = "Quality-based" if enable_quality_filter else "Relevance-based"
    
    # Create Agent
    agent_kwargs = {
        'name': "FieldExtractor",
        'model': model,
        'usability_threshold': 1.0,
        'enable_quality_filter': enable_quality_filter,
        'dataset_focus': dataset_focus,
        'special_instructions': special_instructions
    }
    if category_thresholds:
        agent_kwargs['category_thresholds'] = category_thresholds
    if category_examples:
        agent_kwargs['category_examples'] = category_examples
    
    agent = PSIDFieldExtractionAgent(**agent_kwargs)

    # 处理数据集
    fields = await agent.process_codebook_dataset(dataset_name, chunk_size)
    print(f"✅ Extracted {len(fields)} fields from {dataset_name.upper()}")
    
    # Fixed use adaptive allocation (based on Agent extraction results)
    priority_weights = config.get_priority_weights() if config else None
    unique_fields, _ = agent.adaptive_allocate_fields(
        fields,
        total_target=adaptive_target,
        priority_weights=priority_weights,
    )
    print(f"✅ Final count: {len(unique_fields)} fields (selected & deduplicated)")
    
    # Save results
    # Ensure saved `description` matches codebook exactly (avoid LLM paraphrasing/truncation)
    try:
        codebook = CodebookParser().load_dataset(dataset_name)
        for field in unique_fields:
            var_id = str(field.get("var_id", "")).strip()
            if not var_id:
                continue
            var_info = codebook.get(var_id)
            if isinstance(var_info, dict):
                if var_info.get("label"):
                    field["var_name"] = var_info["label"]
                if var_info.get("description"):
                    field["description"] = var_info["description"]
    except Exception as e:
        print(f"⚠️ Failed to align fields with codebook: {type(e).__name__}: {e}")

    os.makedirs(output_dir, exist_ok=True)
    
    json_file = os.path.join(output_dir, 'extracted_fields_by_agent.json')
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(unique_fields, f, ensure_ascii=False, indent=2)
    print(f"💾 Detailed information saved: {json_file}")
    
    
    # Rebuild categorized dictionary to ensure no duplicates
    final_categorized = {}
    for field in unique_fields:
        category = field.get('category', 'other')
        if category not in final_categorized:
            final_categorized[category] = []
        final_categorized[category].append(field)
    
    for category in sorted(final_categorized.keys()):
        count = len(final_categorized[category])
        avg_usable = sum(f.get('usable_percentage', 0) for f in final_categorized[category]) / count if count > 0 else 0
        avg_importance = sum(f.get('importance_score', 0) for f in final_categorized[category]) / count if count > 0 else 0
        print(f"{category:.<30} {count:>4} fields (avg quality: {avg_usable:.1f}%, avg importance: {avg_importance:.1f})")
    
    return unique_fields


def main(dataset: str = "psid", adaptive_target_override: Optional[int] = None):
  
    dataset = (dataset or "psid").lower()
    chunk_size = {"psid": 25, "acs": 25, "ces": 30, "shed": 30, "sipp": 30}.get(dataset, 20)
    config = get_custom_config(dataset)

    adaptive_target = adaptive_target_override if adaptive_target_override is not None else config.default_total_fields


     # Create model instance
    from llm_config import create_llm_model
    model = create_llm_model(temperature=0.7, max_tokens=2000)
    

    asyncio.run(
        run_extraction_pipeline(
            dataset_name=dataset,
            model=model,
            output_dir=f"data/{dataset}",
            chunk_size=chunk_size,
            config=config,
            adaptive_target=adaptive_target,
        )
    )
