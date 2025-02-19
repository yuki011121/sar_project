from sar_project.agents.base_agent import SARBaseAgent
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ClueMeisterAgent(SARBaseAgent):
    def __init__(self, name="clue_meister"):
        super().__init__(
            name=name,
            role="Clue Analysis Specialist",
            system_message="""You analyze and prioritize clues in SAR missions. Your tasks include:
            1. Sorting and prioritizing clues
            2. Identifying patterns in clues
            3. Initiating further inquiries based on clues
            4. Sharing relevant clue information with other agents"""
        )
        self.clues = []

    def process_request(self, message):
        """Process requests related to clue analysis"""
        try:
            if "add_clue" in message:
                return self.add_clue(message["clue_text"])
            elif "get_clues" in message:
                return self.get_sorted_clues()
            elif "analyze_patterns" in message:
                return self.analyze_patterns()
            else:
                return {"error": "Unknown request type"}
        except Exception as e:
            return {"error": str(e)}

    def add_clue(self, clue_text):
        """Add a new clue and assign a priority"""
        priority = self.calculate_priority(clue_text)
        self.clues.append({"text": clue_text, "priority": priority})
        return {"message": "Clue added", "clue": clue_text, "priority": priority}

    def get_sorted_clues(self):
        """Return clues sorted by priority"""
        sorted_clues = sorted(self.clues, key=lambda x: x["priority"], reverse=True)
        return {"clues": sorted_clues}

    def analyze_patterns(self):
        """Identify potential patterns in clues"""
        keyword_counts = {}
        for clue in self.clues:
            words = clue["text"].lower().split()
            for word in words:
                keyword_counts[word] = keyword_counts.get(word, 0) + 1
        
        frequent_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)
        return {"common_keywords": frequent_keywords[:5]}  

    def calculate_priority(self, clue_text):
        """Use Google Gemini to determine clue priority"""
        prompt = f"Assign a priority (0-10) to the following search and rescue clue based on urgency: '{clue_text}'"
        response = self.query_gemini(prompt)
        try:
            priority = int(response.strip())
            return min(max(priority, 0), 10)  
        except ValueError:
            return 5 

    def find_related_clues(self, new_clue_text):
        """Find the most related clue using TF-IDF similarity."""
        if not self.clues:
            return {"message": "No related clues yet."}
        
        clue_texts = [clue["text"] for clue in self.clues]
        clue_texts.append(new_clue_text)
        
        vectorizer = TfidfVectorizer(stop_words="english")  
        tfidf_matrix = vectorizer.fit_transform(clue_texts)
        similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]

        most_similar_index = similarity_scores.argmax()  
        return {"related_clue": self.clues[most_similar_index]["text"], "similarity": similarity_scores[most_similar_index]}
