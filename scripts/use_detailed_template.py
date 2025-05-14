from src.prompt_manager import PromptManager
from datetime import datetime

pm = PromptManager()
template = pm.get_template("detailed")
prompt_text = template.format(
    birth_date="1990-11-22",
    birth_time="23:45",
    birth_location="Busan, South Korea",
    gender="male",
    current_year=datetime.now().year
)
print(prompt_text)