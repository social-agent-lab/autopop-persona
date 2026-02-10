# autopop-persona
Code for reproducing the experiments in Automated Population Modeling for LLM-driven Social Simulation via Multi-source Data Integration
An LLM-powered automated pipeline for extracting survey data fields, clustering households, and generating rich personas from multiple survey datasets.

## Overview

AutoPop uses AI agents built on the AgentScope framework to automatically process survey codebooks, intelligently extract relevant fields, cluster households into behavioral groups, and generate insightful personas with behavioral patterns. The system supports multiple survey datasets including PSID, ACS/PUMS, CES, SHED, and SIPP.

## Features

- **Intelligent Field Extraction**: Automatically parses codebook files and identifies relevant demographic, economic, and behavioral variables
- **Multi-Dataset Support**: Works with PSID, ACS, CES, SHED, and SIPP survey data
- **Household Clustering**: Uses K-Prototypes algorithm to cluster households based on mixed categorical and numerical features
- **Persona Generation**: Creates rich, story-driven personas with behavioral insights for each cluster
- **Persona calibration**: simulation-driven calibration mechanism that evaluates and iteratively refines personas using held-out questionnaire responses to reproduce observed empirical distributions.
- **Question Design**: Automatically designs survey questions for validation and testing

## Project Structure

```
AutoPop/
├── data/                              # Survey data and codebooks
│   ├── psid/                         # PSID dataset
│   ├── acs/                          # ACS/PUMS dataset
│   ├── ces/                          # Consumer Expenditure Survey
│   ├── shed/                         # Survey of Household Economics
│   └── sipp/                         # Survey of Income and Program Participation
├── result/                           # Output results
│   └── psid_clustering_output/       # PSID Clustering results and personas
│   └── acs_clustering_output/        # ACS Clustering results and personas
│   └── ces_clustering_output/        # CES Clustering results and personas
│   └── shed_clustering_output/       # SHED Clustering results and personas
│   └── sipp_clustering_output/       # SIPP Clustering results and personas
├── psid_field_extraction_agent.py    # Field extraction agent
├── household_clustering_agent.py     # Household clustering system
├── persona_generation.py             # Persona generation engine
├── persona_iterative_refinement.py   # Persona refinement calibration
├── question_design.py                # Survey question design
├── codebook_parser.py                # Codebook parsing utilities
├── field_extraction_config_custom.py # Field extraction configuration
└── llm_config.py                     # LLM model configuration
```

## Prerequisites

- Python 3.8+
- OpenAI API key or compatible LLM service
- Required Python packages (install via pip)

## Installation

```bash
# Clone the repository
git clone https://github.com/social-agent-lab/autopop-persona/
cd AutoPop

# Install required packages
pip install agentscope pandas numpy scikit-learn kmodes matplotlib
```

## Data Setup

Place your survey data files in the appropriate subdirectories under `data/`:

- **PSID**: Place codebook and data files in `data/psid/` URL：https://psidonline.isr.umich.edu/
- **ACS**: Place files in `data/acs/csv_hus/` and `data/acs/csv_pus/` URL：https://www.census.gov/programs-surveys/acs
- **CES**: Place consumer expenditure data in `data/ces/cesdata/` URL：https://www.bls.gov/cex/
- **SHED**: Place files in `data/shed/SHED_public_use_data_2023/` URL：https://www.federalreserve.gov/consumerscommunities/shed.htm
- **SIPP**: Place files in `data/sipp/pu2023_dta/` URL：https://www.census.gov/programs-surveys/sipp.html

Each dataset directory should contain:
- Codebook files (JSON format: `*_codebook_wiki.json`)
- Survey response data (CSV or other formats)

## Usage

### 1. Configure LLM Settings

Edit `llm_config.py` to configure your LLM model settings:

```python
# Set your API key and model preferences
```

### 2. Extract Fields from Codebook

```python
from psid_field_extraction_agent import PSIDFieldExtractionAgent
# Run field extraction
```

### 3. Cluster Households

```python
from household_clustering_agent import FieldSelectionAgent, ClusteringAgent
# Perform household clustering
```

### 4. Generate Personas

```python
from persona_generation import PersonaGeneratorAgent
# Generate personas for each cluster
```

## Key Components

### Field Extraction Agent
Intelligently parses survey codebooks and extracts relevant variables for persona construction while filtering out test fields and low-quality data.

### Household Clustering System
Uses K-Prototypes algorithm to cluster households based on mixed-type features (categorical and numerical), with automatic feature selection and optimization.

### Persona Generator
Transforms clustering results into rich, narrative personas with behavioral insights, patterns, and actionable recommendations.

### Persona refinement
Evaluates and refines personas through behavioral consistency checks and simulation feedback.

### Question Designer
Automatically designs validation questions based on extracted fields and generated personas.

## Output

Results are saved in the `result/` directory, including:
- Clustering assignments
- Cluster statistics
- Generated personas (JSON format)
- persona Fusion
- Visualization plots

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.


