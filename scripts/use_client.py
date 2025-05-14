from src.gpt_client import get_llm_client

llm = get_llm_client()  # 환경변수 LLM_TYPE에 따라 자동 선택

result = llm.generate_completion(
    prompt="1988년 5월 15일생 여성의 사주를 해석해줘.",
    temperature=0.7,
    max_tokens=1024
)
print(result["text"])