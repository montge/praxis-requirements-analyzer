# Praxis Requirements Analyzer

A system for analyzing software requirements using LLMs, vector similarity, and graph databases.

## Key Components

### Research Notebooks (`/notebooks`)

| Notebook | Purpose |
|----------|---------|
| `01_Test_Multi_LLMs.ipynb` | Testing multiple LLM models for requirements analysis workflows using Neo4j data |
| `02_Get_Neo4j_Schema.ipynb` | Extracting and documenting Neo4j database schema using APOC procedures with JSON export |
| `03_Requirements_Judging.ipynb` | LLM-based requirements traceability evaluation using actor-judge pattern with batch processing |
| `04_SiFP_Function_Point_Estimation_Target.ipynb` | Automated Software Function Point (SiFP) estimation using LLM analysis with UGEP/UGDG identification |
| `05_Analysis_Notebook.ipynb` | General requirements analysis and evaluation workflows |
| `06_Sentence_Transformer_Analysis_Notebook.ipynb` | Sentence transformer model analysis for requirements similarity matching |
| `07_Meta_Judge_Analysis_Notebook.ipynb` | Meta-judge analysis for improving LLM-based requirements evaluation accuracy |
| `08_SIFP_Analysis_Notebook.ipynb` | SiFP (Software Intelligence Function Points) analysis and validation |
| `09_Traceability_Analysis_Notebook.ipynb` | Requirements traceability analysis using graph database relationships |
| `10_Hypothesis_1_Sentence_Transformer_Meta_Judge_Comparison.ipynb` | Hypothesis testing: Comparing sentence transformers vs meta-judge for hallucination reduction |
| `11_Hypothesis_2_Tracability_SiFP.ipynb` | Hypothesis testing: Analyzing relationship between requirements traceability and SiFP metrics |
| `12_Hypothesis_3_SiFP_COSMIC_Estimation.ipynb` | Hypothesis testing: Comparing SiFP estimation methods with COSMIC function point standards |

### Code (`/src`)

- Core requirements analysis workflows and prompt management
- LLM model management and integration  
- Neo4j graph database client and operations
- Redis caching layer
- Data models and utility functions

## Installation

### From PyPI (once published)

```bash
pip install praxis-requirements-analyzer
```

### From Source

```bash
# Clone the repository
git clone https://github.com/yourusername/praxis-requirements-analyzer.git
cd praxis-requirements-analyzer

# Install in development mode
pip install -e .
```

## Usage

```python
# Import the main components
from praxis_requirements_analyzer.llm import LLMManager
from praxis_requirements_analyzer.requirements_analyzer import RequirementsWorkflow

# Initialize the LLM manager
llm_manager = LLMManager()
await llm_manager.initialize_models()

# Set up and run the requirements workflow
workflow = RequirementsWorkflow(llm_manager, ...)
matches = await workflow.process_requirements_batch(source_reqs, target_reqs)
```

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
python -m pytest
```

## Attribution

This project includes code that was generated or assisted by [Cursor AI](https://cursor.ai/) tools.