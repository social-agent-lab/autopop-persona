"""
Customized Field Extraction Configuration
Works with codebook_parser.py and psid_field_extraction_agent.py

Core Strategy (Adaptive Allocation):
1. Agent first extracts all relevant fields based on category definitions and examples
2. Statistics on actual field count, quality, and importance per category
3. Adaptively allocate final quotas based on category priority weights and actual distribution
4. Ensure total field count reaches target value (e.g., 75)

Configuration File Purpose:
- Define category priority weights (affects adaptive allocation)
- Define quality thresholds (filter low-quality fields)
- Provide category example keywords (help Agent understand and classify)
- Provide dataset-specific instructions (guide Agent extraction)
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from codebook_parser import CodebookParser


# ============================================================================
# Unified Field Classification System (consistent with field_extraction_config.py)
# ============================================================================

UNIFIED_CATEGORIES = [
    'identifier',              # Unique identifier
    'demographics',            # Demographics
    'income',                  # Income
    'consumption',             # Consumption/Expenditure
    'assets',                  # Assets
    'debt',                    # Debt
    'employment_behavior',     # Employment behavior
    'financial_behavior',      # Financial behavior
    'housing_behavior',        # Housing behavior
    'health_behavior',         # Health behavior
    'education_investment',    # Education investment
    'time_use_lifestyle',      # Time use/Lifestyle
    'attitudes_expectations',  # Attitudes/Expectations
    'program_participation',   # Government programs
    'other'                    # Other
]


@dataclass
class CategoryConfig:
    """Category configuration (for adaptive allocation)"""
    name: str
    priority: int  # 1-10, 10 is highest (affects adaptive allocation weight)
    quality_threshold: float = 0.0  # Quality threshold (%) - set to 0 in adaptive mode, filtered by composite score
    description: str = ""
    examples: List[str] = field(default_factory=list)  # Help Agent understand this category
    
    # Field count configuration
    fixed_fields: int = 0  # Fixed field count, if > 0 then use this fixed count (e.g., identifier fixed at 1), otherwise adaptive
    # NOTE: No longer using suggested_max_fields; final field count determined by adaptive allocation


@dataclass
class DatasetConfig:
    """Dataset configuration (adaptive mode)"""
    dataset_name: str
    full_name: str
    default_total_fields: int = 75  # Default target field count
    category_configs: Dict[str, CategoryConfig] = None  # Category configs (priority, threshold, examples)
    dataset_focus: str = ""
    special_instructions: str = ""
    enable_quality_filter: bool = True
    
    def __post_init__(self):
        if self.category_configs is None:
            self.category_configs = {}
    
    def get_priority_weights(self) -> Dict[str, int]:
        """Get category priority weight dictionary"""
        return {name: cat.priority for name, cat in self.category_configs.items()}
    
    def get_quality_thresholds(self) -> Dict[str, float]:
        """Get category quality threshold dictionary"""
        return {name: cat.quality_threshold for name, cat in self.category_configs.items()}
    
    def get_category_examples(self) -> Dict[str, List[str]]:
        """Get category example keywords dictionary"""
        return {name: cat.examples for name, cat in self.category_configs.items()}
    
    def get_fixed_fields(self) -> Dict[str, int]:
        """Get fixed field count dictionary (categories with fixed field count, e.g., identifier fixed at 1)"""
        return {name: cat.fixed_fields for name, cat in self.category_configs.items() if cat.fixed_fields > 0}
    
    

# ============================================================================
# Category Descriptions (unified usage)
# ============================================================================

CATEGORY_DESCRIPTIONS = {
    'identifier': 'Unique household/person identifier',
    'demographics': 'Basic demographic information (age, gender, marital status, family size, race, ethnicity)',
    'income': 'Income from various sources (wages, business, investments, government transfers)',
    'consumption': 'Daily consumption expenditure (food, transportation, entertainment, goods)',
    'assets': 'Household assets and wealth (home equity, financial assets, business value)',
    'debt': 'Household debts and liabilities (mortgage, credit card debt, other loans)',
    'employment_behavior': 'Employment patterns and work behavior (employment status, industry, occupation, job transitions)',
    'financial_behavior': 'Financial decision-making behavior (saving, investment, financial planning, portfolio allocation)',
    'housing_behavior': 'Housing and residential behavior (own/rent, housing choice, relocation)',
    'health_behavior': 'Health-related behavior (medical spending, insurance, health status)',
    'education_investment': 'Education investment behavior (education spending, student loans, human capital investment)',
    'time_use_lifestyle': 'Time use and lifestyle (work hours, commute time, leisure, volunteer work, work-life balance)',
    'attitudes_expectations': 'Attitudes and expectations (risk attitude, economic expectations, life satisfaction, financial stress)',
    'program_participation': 'Government program participation (SNAP, Medicaid, unemployment insurance)',
    'other': 'Other relevant fields'
}


# ============================================================================
# Dataset Configuration: PSID (Panel Study of Income Dynamics)
# Characteristics: Comprehensive longitudinal household survey, broad coverage
# ============================================================================

PSID_CONFIG = DatasetConfig(
    dataset_name='psid',
    full_name='Panel Study of Income Dynamics',
    default_total_fields=75,
    dataset_focus='Comprehensive longitudinal household panel data',
    special_instructions="""
PSID is the most comprehensive household survey data with broad coverage.
Focus:
- Longitudinal tracking fields (changes, transitions)
- Household structure and dynamics
- Consider both reference person and spouse information
""",
    enable_quality_filter=True,
    category_configs={
        'identifier': CategoryConfig('identifier', priority=10, quality_threshold=0, fixed_fields=1),
        'demographics': CategoryConfig('demographics', priority=9, quality_threshold=0,
            examples=['AGE', 'SEX', 'MARITAL STATUS', 'FAMILY SIZE', 'STATE', 'EDUCATION', 'RACE']),
        'income': CategoryConfig('income', priority=8, quality_threshold=0,
            examples=['WAGES', 'BUSINESS INCOME', 'INVESTMENT INCOME', 'TOTAL INCOME']),
        'consumption': CategoryConfig('consumption', priority=7, quality_threshold=0,
            examples=['FOOD', 'HOUSING', 'MEDICAL', 'TRANSPORTATION', 'ENTERTAINMENT']),
        'assets': CategoryConfig('assets', priority=8, quality_threshold=0,
            examples=['HOME VALUE', 'SAVINGS', 'STOCKS', 'VEHICLE VALUE', 'NET WORTH']),
        'debt': CategoryConfig('debt', priority=7, quality_threshold=0,
            examples=['MORTGAGE', 'CAR LOAN', 'CREDIT CARD DEBT', 'STUDENT LOAN']),
        'employment_behavior': CategoryConfig('employment_behavior', priority=6, quality_threshold=0,
            examples=['WORK HOURS', 'OCCUPATION', 'EMPLOYMENT STATUS', 'JOB CHANGE']),
        'financial_behavior': CategoryConfig('financial_behavior', priority=6, quality_threshold=0,
            examples=['SAVING RATE', 'INVESTMENT', 'RISK ATTITUDE', 'FINANCIAL PLANNING']),
        'housing_behavior': CategoryConfig('housing_behavior', priority=5, quality_threshold=0,
            examples=['RENT/OWN', 'MOVING', 'HOUSING INVESTMENT']),
        'health_behavior': CategoryConfig('health_behavior', priority=5, quality_threshold=0,
            examples=['HEALTH INSURANCE', 'MEDICAL VISITS', 'HEALTH STATUS']),
        'education_investment': CategoryConfig('education_investment', priority=7, quality_threshold=0,
            examples=['EDUCATION EXPENDITURE', 'TUITION', 'STUDENT LOAN']),
        'time_use_lifestyle': CategoryConfig('time_use_lifestyle', priority=4, quality_threshold=0,
            examples=['WORK HOURS', 'COMMUTE TIME', 'LEISURE']),
        'attitudes_expectations': CategoryConfig('attitudes_expectations', priority=6, quality_threshold=0,
            examples=['RISK ATTITUDE', 'LIFE SATISFACTION', 'ECONOMIC EXPECTATIONS']),
        'program_participation': CategoryConfig('program_participation', priority=5, quality_threshold=0,
            examples=['SNAP', 'MEDICAID', 'UNEMPLOYMENT']),
        'other': CategoryConfig('other', priority=3, quality_threshold=0)
    }
)


# ============================================================================
# 数据集配置：ACS (American Community Survey)
# ============================================================================

ACS_CONFIG = DatasetConfig(
    dataset_name='acs',
    full_name='American Community Survey (acs)',
    default_total_fields=75,
    dataset_focus='Large-scale demographic and geographic data',
    special_instructions="""
""",
    enable_quality_filter=True,
    category_configs={
        'identifier': CategoryConfig('identifier', priority=10, quality_threshold=0, fixed_fields=1),
        'demographics': CategoryConfig('demographics', priority=10, quality_threshold=0,
            examples=['AGE', 'SEX', 'RACE', 'HISPANIC', 'CITIZENSHIP', 'NATIVITY', 'LANGUAGE', 'EDUCATION']),
        'income': CategoryConfig('income', priority=8, quality_threshold=0,
            examples=['WAGES', 'SELF-EMPLOYMENT', 'INTEREST', 'SOCIAL SECURITY', 'PUBLIC ASSISTANCE']),
        'consumption': CategoryConfig('consumption', priority=5, quality_threshold=0,
            examples=['FOOD STAMPS VALUE', 'RENT', 'UTILITIES']),
        'assets': CategoryConfig('assets', priority=6, quality_threshold=0,
            examples=['HOME VALUE', 'PROPERTY VALUE']),
        'debt': CategoryConfig('debt', priority=5, quality_threshold=0,
            examples=['MORTGAGE', 'PROPERTY TAX']),
        'employment_behavior': CategoryConfig('employment_behavior', priority=8, quality_threshold=0,
            examples=['EMPLOYMENT STATUS', 'INDUSTRY', 'OCCUPATION', 'CLASS OF WORKER', 'WORK HOURS']),
        'financial_behavior': CategoryConfig('financial_behavior', priority=4, quality_threshold=0,
            examples=['RETIREMENT INCOME', 'INVESTMENT INCOME']),
        'housing_behavior': CategoryConfig('housing_behavior', priority=9, quality_threshold=0,
            examples=['TENURE', 'HOUSING TYPE', 'ROOMS', 'YEAR BUILT', 'MOBILITY', 'MORTGAGE STATUS']),
        'health_behavior': CategoryConfig('health_behavior', priority=6, quality_threshold=0,
            examples=['HEALTH INSURANCE', 'DISABILITY', 'VETERAN STATUS']),
        'education_investment': CategoryConfig('education_investment', priority=6, quality_threshold=0,
            examples=['SCHOOL ENROLLMENT', 'EDUCATIONAL ATTAINMENT']),
        'time_use_lifestyle': CategoryConfig('time_use_lifestyle', priority=4, quality_threshold=0,
            examples=['COMMUTE TIME', 'MEANS OF TRANSPORTATION', 'WORK HOURS']),
        'attitudes_expectations': CategoryConfig('attitudes_expectations', priority=3, quality_threshold=0),
        'program_participation': CategoryConfig('program_participation', priority=6, quality_threshold=0,
            examples=['SNAP', 'MEDICAID', 'MEDICARE', 'SOCIAL SECURITY']),
        'other': CategoryConfig('other', priority=3, quality_threshold=0)
    }
)


# ============================================================================
# Dataset Configuration: CES (Consumer Expenditure Survey)
# Characteristics: Extremely detailed consumption expenditure data, core data source for consumption research
# ============================================================================

CES_CONFIG = DatasetConfig(
    dataset_name='ces',
    full_name='Consumer Expenditure Survey',
    default_total_fields=75,
    dataset_focus='Detailed consumer expenditure patterns and spending behavior',
    special_instructions="""
CES is the gold standard for consumption expenditure research with extremely detailed consumption categories.
Focus:
- Detailed consumption expenditure categories (food, housing, transportation, entertainment, etc.)
- Quarterly consumption patterns
- Consumption priorities and trade-offs
- Relationship between income and expenditure
""",
    enable_quality_filter=True,
    category_configs={
        'identifier': CategoryConfig('identifier', priority=10, quality_threshold=0, fixed_fields=1),
        'demographics': CategoryConfig('demographics', priority=8, quality_threshold=0,
            examples=['AGE', 'SEX', 'MARITAL STATUS', 'FAMILY SIZE', 'EDUCATION', 'RACE']),
        'income': CategoryConfig('income', priority=8, quality_threshold=0,
            examples=['WAGES', 'SELF-EMPLOYMENT', 'SOCIAL SECURITY', 'TOTAL INCOME']),
        'consumption': CategoryConfig('consumption', priority=10, quality_threshold=0,
            examples=['FOOD AT HOME', 'FOOD AWAY', 'HOUSING', 'UTILITIES', 'APPAREL', 'TRANSPORTATION',
                     'HEALTHCARE', 'ENTERTAINMENT', 'PERSONAL CARE', 'EDUCATION', 'TOBACCO', 'ALCOHOL']),
        'assets': CategoryConfig('assets', priority=6, quality_threshold=0,
            examples=['HOME VALUE', 'VEHICLE VALUE', 'FINANCIAL ASSETS']),
        'debt': CategoryConfig('debt', priority=5, quality_threshold=0,
            examples=['MORTGAGE', 'VEHICLE LOAN', 'CREDIT CARD']),
        'employment_behavior': CategoryConfig('employment_behavior', priority=6, quality_threshold=0,
            examples=['EMPLOYMENT STATUS', 'OCCUPATION', 'INDUSTRY', 'WORK HOURS']),
        'financial_behavior': CategoryConfig('financial_behavior', priority=5, quality_threshold=0,
            examples=['SAVINGS', 'INSURANCE PAYMENTS', 'PENSION CONTRIBUTIONS']),
        'housing_behavior': CategoryConfig('housing_behavior', priority=7, quality_threshold=0,
            examples=['TENURE', 'HOUSING TYPE', 'RENT/MORTGAGE PAYMENT']),
        'health_behavior': CategoryConfig('health_behavior', priority=5, quality_threshold=0,
            examples=['HEALTH INSURANCE', 'MEDICAL SPENDING']),
        'education_investment': CategoryConfig('education_investment', priority=5, quality_threshold=0,
            examples=['EDUCATION EXPENDITURE', 'TUITION']),
        'time_use_lifestyle': CategoryConfig('time_use_lifestyle', priority=4, quality_threshold=0,
            examples=['WORK HOURS', 'COMMUTE']),
        'attitudes_expectations': CategoryConfig('attitudes_expectations', priority=4, quality_threshold=0,
            examples=['FINANCIAL SITUATION', 'ECONOMIC OUTLOOK']),
        'program_participation': CategoryConfig('program_participation', priority=3, quality_threshold=0),
        'other': CategoryConfig('other', priority=3, quality_threshold=0)
    }
)


# ============================================================================
# Dataset Configuration: SHED (Survey of Household Economics and Decisionmaking)
# Characteristics: Focus on household economic decision-making, attitudes, expectations, and subjective experiences
# ============================================================================

SHED_CONFIG = DatasetConfig(
    dataset_name='shed',
    full_name='Survey of Household Economics and Decisionmaking',
    default_total_fields=75,
    dataset_focus='Household economic decision-making, attitudes, and subjective well-being',
    special_instructions="""
SHED focuses on household economic decision-making and subjective experiences, an important data source for studying attitudes and expectations.
Focus:
- Economic security and financial stress
- Risk attitudes and decision preferences
- Expectations for the future
- Life satisfaction and well-being
- Financial behavior and decision-making
""",
    enable_quality_filter=True,
    category_configs={
        'identifier': CategoryConfig('identifier', priority=10, quality_threshold=0, fixed_fields=1),
        'demographics': CategoryConfig('demographics', priority=8, quality_threshold=0,
            examples=['AGE', 'SEX', 'MARITAL STATUS', 'FAMILY SIZE', 'EDUCATION', 'RACE']),
        'income': CategoryConfig('income', priority=7, quality_threshold=0,
            examples=['HOUSEHOLD INCOME', 'INCOME SOURCES', 'INCOME STABILITY']),
        'consumption': CategoryConfig('consumption', priority=6, quality_threshold=0,
            examples=['SPENDING PATTERNS', 'FINANCIAL STRAIN', 'EXPENSE MANAGEMENT']),
        'assets': CategoryConfig('assets', priority=6, quality_threshold=0,
            examples=['SAVINGS', 'RETIREMENT ACCOUNTS', 'HOME OWNERSHIP']),
        'debt': CategoryConfig('debt', priority=7, quality_threshold=0,
            examples=['CREDIT CARD DEBT', 'STUDENT LOANS', 'DEBT BURDEN']),
        'employment_behavior': CategoryConfig('employment_behavior', priority=6, quality_threshold=0,
            examples=['EMPLOYMENT STATUS', 'JOB SECURITY', 'GIG WORK']),
        'financial_behavior': CategoryConfig('financial_behavior', priority=9, quality_threshold=0,
            examples=['SAVINGS BEHAVIOR', 'INVESTMENT', 'FINANCIAL PLANNING', 'EMERGENCY FUND', 'BUDGETING']),
        'housing_behavior': CategoryConfig('housing_behavior', priority=5, quality_threshold=0,
            examples=['HOUSING TENURE', 'HOUSING COSTS', 'HOUSING SECURITY']),
        'health_behavior': CategoryConfig('health_behavior', priority=5, quality_threshold=0,
            examples=['HEALTH INSURANCE', 'MEDICAL COSTS', 'HEALTH STATUS']),
        'education_investment': CategoryConfig('education_investment', priority=6, quality_threshold=0,
            examples=['EDUCATION DEBT', 'EDUCATION VALUE', 'CHILD EDUCATION']),
        'time_use_lifestyle': CategoryConfig('time_use_lifestyle', priority=4, quality_threshold=0,
            examples=['WORK-LIFE BALANCE', 'LEISURE']),
        'attitudes_expectations': CategoryConfig('attitudes_expectations', priority=10, quality_threshold=0,
            examples=['FINANCIAL WELL-BEING', 'ECONOMIC OUTLOOK', 'RISK ATTITUDE', 'LIFE SATISFACTION',
                     'FINANCIAL STRESS', 'CONFIDENCE', 'OPTIMISM', 'RETIREMENT CONFIDENCE']),
        'program_participation': CategoryConfig('program_participation', priority=6, quality_threshold=0,
            examples=['SNAP', 'MEDICAID', 'UNEMPLOYMENT', 'STIMULUS PAYMENTS']),
        'other': CategoryConfig('other', priority=3, quality_threshold=0)
    }
)


# ============================================================================
# Dataset Configuration: SIPP (Survey of Income and Program Participation)
# Characteristics: Detailed tracking of government program participation and income dynamics
# ============================================================================

SIPP_CONFIG = DatasetConfig(
    dataset_name='sipp',
    full_name='Survey of Income and Program Participation',
    default_total_fields=75,
    dataset_focus='Income dynamics and government program participation',
    special_instructions="""
SIPP focuses on income dynamics and government program participation, important data for studying the social safety net.
Focus:
- Detailed government program participation information
- Dynamic changes in income sources
- Usage patterns of welfare programs
- Employment and income transitions
""",
    enable_quality_filter=True,
    category_configs={
        'identifier': CategoryConfig('identifier', priority=10, quality_threshold=0, fixed_fields=1),
        'demographics': CategoryConfig('demographics', priority=8, quality_threshold=0,
            examples=['AGE', 'SEX', 'MARITAL STATUS', 'FAMILY SIZE', 'EDUCATION', 'RACE', 'CITIZENSHIP']),
        'income': CategoryConfig('income', priority=9, quality_threshold=0,
            examples=['WAGES', 'SELF-EMPLOYMENT', 'SOCIAL SECURITY', 'SSI', 'UNEMPLOYMENT', 'WELFARE']),
        'consumption': CategoryConfig('consumption', priority=5, quality_threshold=0,
            examples=['FOOD SPENDING', 'HOUSING COSTS', 'UTILITIES']),
        'assets': CategoryConfig('assets', priority=7, quality_threshold=0,
            examples=['BANK ACCOUNTS', 'STOCKS', 'HOME VALUE', 'VEHICLE VALUE']),
        'debt': CategoryConfig('debt', priority=6, quality_threshold=0,
            examples=['MORTGAGE', 'CREDIT CARD', 'STUDENT LOAN', 'MEDICAL DEBT']),
        'employment_behavior': CategoryConfig('employment_behavior', priority=8, quality_threshold=0,
            examples=['EMPLOYMENT STATUS', 'JOB SEARCH', 'OCCUPATION', 'INDUSTRY', 'WORK HOURS', 'JOB TRANSITIONS']),
        'financial_behavior': CategoryConfig('financial_behavior', priority=6, quality_threshold=0,
            examples=['SAVINGS', 'BANK ACCOUNT', 'FINANCIAL ASSETS']),
        'housing_behavior': CategoryConfig('housing_behavior', priority=6, quality_threshold=0,
            examples=['TENURE', 'RENT/MORTGAGE', 'HOUSING ASSISTANCE', 'MOBILITY']),
        'health_behavior': CategoryConfig('health_behavior', priority=7, quality_threshold=0,
            examples=['HEALTH INSURANCE', 'MEDICAID', 'MEDICARE', 'DISABILITY', 'MEDICAL COSTS']),
        'education_investment': CategoryConfig('education_investment', priority=5, quality_threshold=0,
            examples=['SCHOOL ENROLLMENT', 'EDUCATION ASSISTANCE', 'STUDENT LOANS']),
        'time_use_lifestyle': CategoryConfig('time_use_lifestyle', priority=4, quality_threshold=0,
            examples=['WORK HOURS', 'CHILD CARE']),
        'attitudes_expectations': CategoryConfig('attitudes_expectations', priority=5, quality_threshold=0,
            examples=['JOB SATISFACTION', 'ECONOMIC EXPECTATIONS']),
        'program_participation': CategoryConfig('program_participation', priority=10, quality_threshold=0,
            examples=['SNAP', 'TANF', 'MEDICAID', 'MEDICARE', 'SSI', 'SOCIAL SECURITY', 'UNEMPLOYMENT',
                     'WIC', 'HOUSING ASSISTANCE', 'ENERGY ASSISTANCE', 'SCHOOL LUNCH']),
        'other': CategoryConfig('other', priority=3, quality_threshold=0)
    }
)


# ============================================================================
# Configuration Dictionary
# ============================================================================

CUSTOM_CONFIGS = {
    'psid': PSID_CONFIG,
    'acs': ACS_CONFIG,
    'ces': CES_CONFIG,
    'shed': SHED_CONFIG,
    'sipp': SIPP_CONFIG
}


# ============================================================================
# Configuration Management Class
# ============================================================================

class CustomConfigManager:
    """Custom configuration manager"""
    
    def __init__(self):
        self.configs = CUSTOM_CONFIGS
        self.parser = CodebookParser()
    
    def get_config(self, dataset_name: str) -> DatasetConfig:
        """
        Get dataset configuration
        
        Args:
            dataset_name: Dataset name (psid, pums, ces, shed, sipp)
            
        Returns:
            DatasetConfig: Dataset configuration
        """
        dataset_name = dataset_name.lower()
        if dataset_name not in self.configs:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. "
                f"Available datasets: {', '.join(self.configs.keys())}"
            )
        return self.configs[dataset_name]
    
    def validate_all_configs(self) -> Dict[str, Tuple[bool, str]]:
        """Validate all configurations"""
        results = {}
        for dataset_name, config in self.configs.items():
            results[dataset_name] = config.validate()
        return results
    
    def print_all_configs(self):
        """Print configuration plans for all datasets"""
        print("\n" + "=" * 70)
        print("Custom Field Extraction Configuration - All Dataset Configurations (Adaptive Mode)")
        print("=" * 70)
        
        for dataset_name, config in self.configs.items():
            print(config.get_config_summary())
    
    def compare_allocations(self, category: str):
        """
        Compare field allocations for a specific category across different datasets
        
        Args:
            category: Category name
        """
        print(f"\nCategory '{category}' allocation comparison across datasets:")
        print("-" * 70)
        
        for dataset_name, config in self.configs.items():
            if category in config.category_configs:
                cat_config = config.category_configs[category]
                print(f"{dataset_name.upper():8s} | "
                      f"Priority: {cat_config.priority:2d} | "
                      f"Quality Threshold: {cat_config.quality_threshold:5.1f}%")
    
    def get_dataset_statistics(self, dataset_name: str) -> Dict:
        """
        Get dataset statistics (combined with codebook data)
        
        Args:
            dataset_name: Dataset name
            
        Returns:
            Dict: Statistics information
        """
        dataset_name = dataset_name.lower()
        config = self.get_config(dataset_name)
        
        # Load codebook data
        try:
            codebook = self.parser.load_dataset(dataset_name)
            total_variables = len(codebook)
        except Exception as e:
            total_variables = "N/A"
        
        # Statistics for configuration
        categories_total = len(config.category_configs)
        categories_with_examples = sum(1 for cat in config.category_configs.values() if cat.examples)
        
        return {
            'dataset_name': dataset_name,
            'full_name': config.full_name,
            'total_variables_in_codebook': total_variables,
            'default_target_fields': config.default_total_fields,
            'categories_total': categories_total,
            'categories_with_examples': categories_with_examples,
            'enable_quality_filter': config.enable_quality_filter
        }
    
    def export_config_to_dict(self, dataset_name: str) -> Dict:
        """
        Export configuration as dictionary format (for passing to field extraction agent)
        
        Args:
            dataset_name: Dataset name
            
        Returns:
            Dict: Configuration dictionary
        """
        config = self.get_config(dataset_name)
        
        return {
            'config_name': config.dataset_name,
            'dataset_focus': config.dataset_focus,
            'special_instructions': config.special_instructions,
            'enable_quality_filter': config.enable_quality_filter,
            'default_total_fields': config.default_total_fields,
            'categories': {
                cat_name: {
                    'name': cat_config.name,
                    'description': CATEGORY_DESCRIPTIONS.get(cat_name, ''),
                    'fixed_fields': cat_config.fixed_fields,
                    'priority': cat_config.priority,
                    'quality_threshold': cat_config.quality_threshold,
                    'examples': cat_config.examples
                }
                for cat_name, cat_config in config.category_configs.items()
            }
        }


# ============================================================================
# Utility Functions
# ============================================================================

def get_custom_config(dataset_name: str) -> DatasetConfig:
    """
    Get custom configuration for dataset
    
    Args:
        dataset_name: Dataset name
        
    Returns:
        DatasetConfig: Dataset configuration
    """
    manager = CustomConfigManager()
    return manager.get_config(dataset_name)


def print_config_summary(dataset_name: str):
    """
    Print dataset configuration summary
    
    Args:
        dataset_name: Dataset name
    """
    manager = CustomConfigManager()
    config = manager.get_config(dataset_name)
    print(config.get_allocation_summary())
    
    # Validate
    is_valid, message = config.validate()
    if is_valid:
        print(f"✓ Configuration is valid")
    else:
        print(f"✗ Configuration error: {message}")
    
    # Statistics
    stats = manager.get_dataset_statistics(dataset_name)
    print(f"\nDataset Statistics:")
    print(f"  Total variables in codebook: {stats['total_variables_in_codebook']}")
    print(f"  Default target field count: {stats['default_target_fields']}")
    print(f"  Total categories: {stats['categories_total']}")
    print(f"  Categories with examples: {stats['categories_with_examples']}")


def compare_all_datasets():
    """Compare configurations for all datasets"""
    manager = CustomConfigManager()
    manager.print_all_configs()
    
    print("\n" + "=" * 70)
    print("Dataset Feature Comparison")
    print("=" * 70)
    
    # Find top 3 categories with highest priority for each dataset
    for dataset_name, config in manager.configs.items():
        sorted_cats = sorted(
            [(name, cat_config.priority) for name, cat_config in config.category_configs.items()],
            key=lambda x: x[1],
            reverse=True
        )
        top_3 = [f"{name}({count})" for name, count in sorted_cats[:3] if count > 0]
        print(f"{config.full_name:50s} | Features: {', '.join(top_3)}")

