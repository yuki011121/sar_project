# ClueMeister Agent

**ClueMeisterAgent** is an intelligent Search and Rescue (SAR) assistant designed to analyze and prioritize clues during missions. It provides automated clue processing, pattern recognition, and similarity analysis to help SAR teams efficiently identify relevant clues.

---

## Features

- **Clue Prioritization**  
  Assigns priority scores (0-10) to clues based on urgency using Google Gemini API.

- **Pattern Recognition**  
  Analyzes the most frequent keywords within clues to detect common patterns.

- **Similarity Analysis**  
  Utilizes TF-IDF vectorization and cosine similarity to find related clues from past observations.

- **Inquiry Initiation**  
  Suggests possible follow-up actions based on clue patterns.

---

## Implementation Details

- **Natural Language Processing (NLP)**  
  Uses TF-IDF vectorization for analyzing relationships between clues.

- **AI-Driven Prioritization**  
  Leverages Google Gemini to dynamically determine the urgency of clues.

- **Data Handling**  
  Real-time storage, sorting, and analysis of clues to assist rescue operations.

- **Integration**  
  Works within the `sar_project` framework, extending the `SARBaseAgent` class.

---

## Example Usage

```python
from sar_project.agents.clue_meister_agent import ClueMeisterAgent

agent = ClueMeisterAgent()
agent.add_clue("Footprints found near riverbank")
print(agent.get_sorted_clues())
