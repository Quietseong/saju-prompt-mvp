# tests/test_prompt_manager.py

from src.prompt_manager import PromptManager

def test_summary_template_format():
    pm = PromptManager()
    template = pm.get_template("summary")
    prompt = template.format(
        birth_date="1985-03-15",
        birth_time="08:30",
        birth_location="Seoul, South Korea",
        gender="female"
    )
    assert "PERSONALITY OVERVIEW:" in prompt
    assert "SCORES:" in prompt