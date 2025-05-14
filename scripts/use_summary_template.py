# scripts/use_summary_template.py

from src.prompt_manager import PromptManager

def main():
    pm = PromptManager()
    template = pm.get_template("summary")
    prompt_text = template.format(
        birth_date="1985-03-15",
        birth_time="08:30",
        birth_location="Seoul, South Korea",
        gender="female"
    )
    print(prompt_text)

if __name__ == "__main__":
    main()