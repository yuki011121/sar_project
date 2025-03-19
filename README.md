# ClueMeister Agent

**ClueMeisterAgent** is an intelligent Search and Rescue (SAR) assistant designed to analyze and prioritize clues during missions. It provides automated clue processing, pattern recognition, and similarity analysis to help SAR teams efficiently identify relevant clues.

### **Changes in ClueMeisterAgent**  

This update improves the functionality of `ClueMeisterAgent` by optimizing three key functions:  

1. **`analyze_patterns`**  
   - Added **lemmatization** to normalize words (e.g., "shouting" → "shout").  
   - Removed **common stopwords** (e.g., "the", "in", "near") for better keyword extraction.  
   - Filtered **non-alphabetic characters** and ignored words with fewer than three letters.  

2. **`calculate_priority`**  
   - Improved **error handling** when retrieving priority from Google Gemini API.  
   - Used **regular expressions** to safely extract numeric values.  
   - Enforced **priority range (0-10)** and set a **default fallback value (5)**.  

3. **`find_related_clues`**  
   - Enhanced **TF-IDF similarity calculation** by adding **bigram support (`ngram_range=(1,2)`)**.  
   - Introduced a **minimum similarity threshold (0.2)** to filter out weak matches. 



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


## Installation & Setup

**Clone or fork** this repository and navigate to the project directory:
   ```bash
   git clone https://github.com/yourname/sar_project.git
   cd sar_project
```
   
## Example Usage

### Basic Code Example

```python
from sar_project.agents.clue_meister_agent import ClueMeisterAgent

agent = ClueMeisterAgent()

# Add clues
agent.add_clue("Footprints found near riverbank")
agent.add_clue("Bloodstains discovered in the forest")

# Get clues sorted by priority
sorted_clues = agent.get_sorted_clues()
print("Sorted Clues:", sorted_clues)

# Analyze patterns in existing clues
patterns = agent.analyze_patterns()
print("Common Patterns:", patterns)

# Find the most related clue to a new observation
related = agent.find_related_clues("Suspicious clothing found near river")
print("Most Related Clue:", related)
```
### Process Request Interface

```python
request_add = {
    "add_clue": True,
    "clue_text": "Shouting heard near mountain trail"
}
response_add = agent.process_request(request_add)
print("Add Clue Response:", response_add)

request_get = {"get_clues": True}
response_get = agent.process_request(request_get)
print("Get Clues Response:", response_get)

request_patterns = {"analyze_patterns": True}
response_patterns = agent.process_request(request_patterns)
print("Analyze Patterns Response:", response_patterns)
```

### Running Tests
```python
pytest tests/test_clue_meister_agent.py
```


