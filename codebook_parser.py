"""
Unified Codebook Data Reader
Supports reading codebook files for all datasets: PSID, ACS, CES, SHED, SIPP
"""

import json
from typing import Dict, Optional
from pathlib import Path


class CodebookParser:
    """Unified Codebook Parser - for reading codebook JSON files"""
    
    # Dataset configuration
    DATASETS = {
        'psid': {
            'name': 'Panel Study of Income Dynamics',
            'file': 'data/psid/PSID_codebook_wiki.json'
        },
        'acs': {
            'name': 'American Community Survey',
            'file': 'data/acs/ACS_codebook_wiki.json'
        },
        'ces': {
            'name': 'Consumer Expenditure Survey',
            'file': 'data/ces/CES_codebook_wiki.json'
        },
        'shed': {
            'name': 'Survey of Household Economics and Decisionmaking',
            'file': 'data/shed/SHED_codebook_wiki.json'
        },
        'sipp': {
            'name': 'Survey of Income and Program Participation',
            'file': 'data/sipp/sipp_codebook_wiki.json'
        }
    }
    
    def __init__(self, base_path: Optional[str] = None):
        """
        Initialize CodebookParser
        
        Args:
            base_path: Project root directory path, defaults to the directory where this file is located
        """
        if base_path is None:
            self.base_path = Path(__file__).parent
        else:
            self.base_path = Path(base_path)
        
        self.codebooks: Dict[str, Dict] = {}
        
    def load_dataset(self, dataset_name: str) -> Dict:
        """
        Load codebook for the specified dataset
        
        Args:
            dataset_name: Dataset name (psid, pums, ces, shed, sipp)
            
        Returns:
            Dict: Codebook data dictionary, format: {variable_name: {label, description, code_values, ...}}
            
        Raises:
            ValueError: If dataset name does not exist
            FileNotFoundError: If file does not exist
        """
        dataset_name = dataset_name.lower()
        
        if dataset_name not in self.DATASETS:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. "
                f"Available datasets: {', '.join(self.DATASETS.keys())}"
            )
        
        # If already loaded, return directly
        if dataset_name in self.codebooks:
            return self.codebooks[dataset_name]
        
        # Build file path
        file_path = self.base_path / self.DATASETS[dataset_name]['file']
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Load JSON file
        with open(file_path, 'r', encoding='utf-8') as f:
            self.codebooks[dataset_name] = json.load(f)
        
        print(f"✓ Loaded {self.DATASETS[dataset_name]['name']} "
              f"({len(self.codebooks[dataset_name])} variables)")
        
        return self.codebooks[dataset_name]
    
    def load_all_datasets(self) -> Dict[str, Dict]:
        """
        Load codebooks for all datasets
        
        Returns:
            Dict[str, Dict]: Dictionary of codebooks for all datasets
        """
        for dataset_name in self.DATASETS.keys():
            try:
                self.load_dataset(dataset_name)
            except FileNotFoundError as e:
                print(f"⚠ Skipped {dataset_name}: {e}")
        
        return self.codebooks
    
    def get_variable(self, dataset_name: str, variable_name: str) -> Optional[Dict]:
        """
        Get complete information for the specified variable
        
        Args:
            dataset_name: Dataset name
            variable_name: Variable name
            
        Returns:
            Dict: Variable information dictionary, returns None if it doesn't exist
        """
        dataset_name = dataset_name.lower()
        
        # Ensure dataset is loaded
        if dataset_name not in self.codebooks:
            self.load_dataset(dataset_name)
        
        return self.codebooks[dataset_name].get(variable_name)
    
    def get_all_variables(self, dataset_name: str) -> Dict[str, Dict]:
        """
        Get all variables in the dataset
        
        Args:
            dataset_name: Dataset name
            
        Returns:
            Dict[str, Dict]: Dictionary of all variables
        """
        dataset_name = dataset_name.lower()
        
        # Ensure dataset is loaded
        if dataset_name not in self.codebooks:
            self.load_dataset(dataset_name)
        
        return self.codebooks[dataset_name]
    
    def get_variable_names(self, dataset_name: str) -> list:
        """
        Get all variable names in the dataset
        
        Args:
            dataset_name: Dataset name
            
        Returns:
            list: List of variable names
        """
        dataset_name = dataset_name.lower()
        
        # Ensure dataset is loaded
        if dataset_name not in self.codebooks:
            self.load_dataset(dataset_name)
        
        return list(self.codebooks[dataset_name].keys())
