# ClueMeisterAgent

## Introduction
`ClueMeisterAgent` is an AI agent designed for **Search & Rescue (SAR) missions**, specifically for **analyzing and prioritizing clues** to assist command personnel in decision-making.

## Key Features
- **Clue Management**
  - Receive and store SAR-related clues
  - Assign priority levels to clues based on urgency
- **Pattern Recognition**
  - Identify keyword patterns in clue sets
- **Relevance Analysis**
  - Compute similarity between new clues and existing ones

## Code Example
```python
from sar_project.agents.clue_meister_agent import ClueMeisterAgent

agent = ClueMeisterAgent()
agent.add_clue("Found footprints near river")
print(agent.get_sorted_clues())
