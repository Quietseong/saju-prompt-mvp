from src.prompt_manager import PromptManager

pm = PromptManager()
template = pm.get_template("feedback")
prompt_text = template.format(
    saju_reading="(여기에 실제 사주 점괘 결과 텍스트)",
    birth_date="1985-03-15",
    birth_time="08:30",
    birth_location="Seoul, South Korea",
    gender="female"
)
print(prompt_text)