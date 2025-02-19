import pytest
from sar_project.agents.clue_meister_agent import ClueMeisterAgent

@pytest.fixture
def clue_meister():
    return ClueMeisterAgent()

def test_add_clue(clue_meister):
    response = clue_meister.add_clue("Found blood stains near riverbank")
    assert response["message"] == "Clue added"
    assert response["clue"] == "Found blood stains near riverbank"
    assert isinstance(response["priority"], int)  

def test_get_sorted_clues(clue_meister):
    clue_meister.add_clue("Heard shouting in the forest")
    clue_meister.add_clue("Found footprints near cave entrance")
    response = clue_meister.get_sorted_clues()
    assert len(response["clues"]) > 0
    assert response["clues"][0]["priority"] >= response["clues"][-1]["priority"]

def test_analyze_patterns(clue_meister):
    clue_meister.add_clue("Heard shouting in the woods")
    clue_meister.add_clue("Shouting reported near the mountain")
    response = clue_meister.analyze_patterns()
    assert "shouting" in [kw[0] for kw in response["common_keywords"]]

def test_find_related_clues(clue_meister):
    clue_meister.add_clue("Bloodstains found near river")
    clue_meister.add_clue("Clothing found near river")
    response = clue_meister.find_related_clues("Footprints near river")
    assert response["similarity"] > 0.4