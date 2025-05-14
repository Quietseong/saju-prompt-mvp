import os
import json
import logging
from typing import List, Dict, Any
from src.gpt_client import get_llm_client
from src.models import SajuInput, Gender

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_personas(filepath: str) -> List[Dict[str, Any]]:
    """
    페르소나(입력 데이터셋) 파일을 로드한다.

    Args:
        filepath (str): JSON 파일 경로
    Returns:
        List[Dict[str, Any]]: 페르소나 리스트
    """
    with open(filepath, encoding="utf-8") as f:
        return json.load(f)


def load_prompt_template(filepath: str) -> str:
    """
    프롬프트 템플릿 파일을 로드한다.
    (Phi-3 등 고급 LLM 실험에서는 'template_text' 키를 사용)

    Args:
        filepath (str): JSON 파일 경로
    Returns:
        str: 프롬프트 템플릿 문자열
    """
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)
        return data["template_text"]


def fill_prompt(template: str, saju: SajuInput) -> str:
    """
    SajuInput 객체를 프롬프트 템플릿에 채워넣는다.

    Args:
        template (str): 프롬프트 템플릿
        saju (SajuInput): 입력 데이터
    Returns:
        str: 완성된 프롬프트
    """
    return template.format(
        birth_date=saju.birth_date.strftime("%Y-%m-%d"),
        birth_time=saju.birth_time,
        birth_location=saju.birth_location,
        gender=saju.gender.value
    )


def run_experiment(
    personas_path: str = "data/sample_personas.json",
    prompt_path: str = "prompts/summary_v1.json",
    output_path: str = "results/summary_results.json",
    llm_type: str = None
) -> None:
    """
    실험 자동화: 입력 데이터셋, 프롬프트 템플릿, LLM을 이용해 결과를 생성하고 저장한다.

    Args:
        personas_path (str): 페르소나 데이터셋 경로
        prompt_path (str): 프롬프트 템플릿 경로
        output_path (str): 결과 저장 경로
        llm_type (str): 사용할 LLM 종류 ('openai', 'sllm', 'phi3')
    """
    logger.info(f"실험 시작: {personas_path}, {prompt_path}, LLM={llm_type}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    personas = load_personas(personas_path)
    template = load_prompt_template(prompt_path)
    llm = get_llm_client(llm_type)
    results = []

    for idx, persona in enumerate(personas):
        try:
            # SajuInput 데이터 검증
            saju = SajuInput(**persona)
            if not saju.validate():
                logger.warning(f"{idx}번 페르소나: 입력값 검증 실패")
                continue
            prompt = fill_prompt(template, saju)
            logger.info(f"{idx}번 프롬프트 생성: {prompt}")
            response = llm.generate_completion(prompt)
            results.append({
                "persona": persona,
                "prompt": prompt,
                "response": response["text"],
                "meta": {
                    "model": response["model"],
                    "usage": response.get("usage", {}),
                    "timestamp": response.get("timestamp")
                }
            })
        except Exception as e:
            logger.error(f"{idx}번 페르소나 처리 중 오류: {str(e)}")

    # 결과 저장
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"실험 결과 저장 완료: {output_path}")


if __name__ == "__main__":
    """
    실험 자동화 스크립트 실행 예시
    """
    run_experiment(
        personas_path="data/sample_personas.json",
        prompt_path="prompts/summary_v1.json",
        output_path="results/summary_results.json",
        llm_type="phi3"  # 또는 os.getenv("LLM_TYPE", "phi3")
    ) 