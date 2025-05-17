import os
import json
import time
import logging
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

from src.models import SajuInput, parse_saju_input
from src.prompt_manager import PromptManager
from src.llm_client import LLMClient, ModelType, save_response

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ExperimentRunner:
    """
    다양한 프롬프트 템플릿과 입력 조합으로 실험을 자동화하는 모듈
    
    이 클래스는 템플릿, 페르소나, LLM 모델 등 다양한 조합으로 실험을 실행하고
    결과를 구조화된 형식으로 저장합니다.
    """
    
    def __init__(self, 
                output_dir: str = "results/experiments",
                provider_type: Union[ModelType, str] = ModelType.LLAMA,
                model: Optional[str] = None,
                api_key: Optional[str] = None,
                model_path: Optional[str] = None,
                host: str = "http://localhost:8000"):
        """
        ExperimentRunner 초기화
        
        Args:
            output_dir: 결과를 저장할 디렉토리
            provider_type: LLM 제공자 타입 ('openai' 또는 'llama')
            model: 사용할 모델 이름/ID
            api_key: OpenAI API 키 (OpenAI 모델용)
            model_path: 로컬 모델 경로 (Llama 모델용)
            host: 모델 서버 주소 (Llama 모델용)
        """
        self.prompt_manager = PromptManager()
        
        # LLM 클라이언트 초기화
        self.llm_client = LLMClient(
            provider_type=provider_type,
            model=model or ("gpt-3.5-turbo" if provider_type == ModelType.OPENAI else "llama-3-8b-instruct"),
            api_key=api_key,
            model_path=model_path,
            host=host
        )
        
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def load_personas(self, 
                     file_path: str = "data/sample_personas.json", 
                     limit: Optional[int] = None,
                     filter_func = None) -> List[Dict[str, Any]]:
        """
        페르소나 데이터 로드
        
        Args:
            file_path: 페르소나 데이터 파일 경로
            limit: 로드할 최대 페르소나 수
            filter_func: 페르소나 필터링 함수
            
        Returns:
            페르소나 데이터 목록
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                personas = json.load(f)
            
            logger.info(f"{len(personas)} 개의 페르소나 로드됨: {file_path}")
            
            # 필터 적용
            if filter_func:
                personas = [p for p in personas if filter_func(p)]
                logger.info(f"필터 적용 후 {len(personas)} 개의 페르소나 남음")
                
            # 페르소나 수 제한
            if limit and limit > 0:
                personas = personas[:limit]
                logger.info(f"제한 적용: {len(personas)} 개의 페르소나 사용")
                
            return personas
        except Exception as e:
            logger.error(f"페르소나 로드 실패: {e}")
            raise
    
    def run_experiment(self, 
                      experiment_name: str, 
                      template_ids: List[str], 
                      personas: List[Dict[str, Any]],
                      temperature: float = 0.7,
                      max_tokens: int = 2000) -> tuple:
        """
        여러 템플릿과 페르소나로 실험 실행
        
        Args:
            experiment_name: 실험 이름
            template_ids: 사용할 템플릿 ID 목록
            personas: 사용할 페르소나 목록
            temperature: 생성 다양성 (0~2)
            max_tokens: 생성할 최대 토큰 수
            
        Returns:
            실험 디렉토리 경로와 결과 목록의 튜플
        """
        # 실험 ID 및 디렉토리 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_id = f"{experiment_name}_{timestamp}"
        experiment_dir = os.path.join(self.output_dir, experiment_id)
        os.makedirs(experiment_dir, exist_ok=True)
        
        logger.info(f"실험 시작: {experiment_id}")
        logger.info(f"템플릿: {', '.join(template_ids)}")
        logger.info(f"페르소나 수: {len(personas)}")
        
        # 결과 저장용 리스트
        results = []
        
        # 템플릿 및 페르소나 루프
        for template_id in template_ids:
            try:
                template = self.prompt_manager.get_template(template_id)
                logger.info(f"템플릿 사용 중: {template_id} v{template.version}")
                
                for i, persona in enumerate(personas):
                    persona_id = persona.get('id', f"unknown_{i}")
                    logger.info(f"페르소나 처리 중: {persona_id} ({i+1}/{len(personas)})")
                    
                    try:
                        # 기본 데이터 준비
                        current_year = datetime.now().year
                        template_data = {
                            "birth_date": persona.get('birth_date', ''),
                            "birth_time": persona.get('birth_time', ''),
                            "birth_location": persona.get('birth_location', ''),
                            "gender": persona.get('gender', ''),
                            "current_year": current_year
                        }
                        
                        # 프롬프트 포맷팅
                        formatted_prompt = template.format(**template_data)
                        
                        # 완성 생성
                        start_time = time.time()
                        response = self.llm_client.generate_saju_reading(
                            prompt=formatted_prompt,
                            temperature=temperature,
                            max_tokens=max_tokens
                        )
                        end_time = time.time()
                        
                        # 결과 기록
                        result = {
                            "experiment_id": experiment_id,
                            "template_id": template_id,
                            "template_version": template.version,
                            "persona_id": persona_id,
                            "birth_date": template_data["birth_date"],
                            "birth_time": template_data["birth_time"],
                            "birth_location": template_data["birth_location"],
                            "gender": template_data["gender"],
                            "prompt": formatted_prompt,
                            "response": response["text"],
                            "model": response["model"],
                            "provider": response.get("provider", "unknown"),
                            "prompt_tokens": response["usage"]["prompt_tokens"],
                            "completion_tokens": response["usage"]["completion_tokens"],
                            "total_tokens": response["usage"]["total_tokens"],
                            "finish_reason": response["finish_reason"],
                            "response_time": end_time - start_time,
                            "timestamp": datetime.now().isoformat()
                        }
                        
                        results.append(result)
                        
                        # 개별 결과 저장
                        result_filename = f"{template_id}_{persona_id}_{timestamp}.json"
                        result_path = os.path.join(experiment_dir, result_filename)
                        
                        with open(result_path, 'w', encoding='utf-8') as f:
                            json.dump(result, f, indent=2, ensure_ascii=False)
                            
                        logger.info(f"결과 저장됨: {result_path}")
                        
                    except Exception as e:
                        logger.error(f"페르소나 {persona_id} 처리 중 오류: {e}")
                        # 계속 진행
                
            except Exception as e:
                logger.error(f"템플릿 {template_id} 처리 중 오류: {e}")
                # 다음 템플릿으로 계속
        
        # 전체 결과가 없으면 실패
        if not results:
            logger.error("실험 실패: 결과 없음")
            return experiment_dir, []
        
        # 모든 결과를 CSV로 저장
        try:
            results_df = pd.DataFrame(results)
            csv_path = os.path.join(experiment_dir, "all_results.csv")
            results_df.to_csv(csv_path, index=False)
            logger.info(f"CSV 결과 저장됨: {csv_path}")
        except Exception as e:
            logger.error(f"CSV 저장 실패: {e}")
        
        # 실험 요약 저장
        try:
            summary = {
                "experiment_id": experiment_id,
                "experiment_name": experiment_name,
                "templates_used": template_ids,
                "personas_count": len(personas),
                "total_completions": len(results),
                "total_tokens": sum(r["total_tokens"] for r in results),
                "average_response_time": sum(r["response_time"] for r in results) / max(1, len(results)),
                "timestamp": datetime.now().isoformat(),
                "provider": self.llm_client.provider_type,
                "model": self.llm_client.model
            }
            
            summary_path = os.path.join(experiment_dir, "experiment_summary.json")
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
                
            logger.info(f"실험 요약 저장됨: {summary_path}")
        except Exception as e:
            logger.error(f"요약 저장 실패: {e}")
        
        logger.info(f"실험 {experiment_id} 완료. 결과가 {experiment_dir}에 저장됨")
        logger.info(f"완성 생성 수: {len(results)}")
        logger.info(f"총 토큰 사용량: {sum(r['total_tokens'] for r in results)}")
        
        return experiment_dir, results
    
    def run_feedback_experiment(self, 
                               original_experiment_dir: str, 
                               template_id: str = "feedback",
                               temperature: float = 0.7,
                               max_tokens: int = 2000) -> tuple:
        """
        이전 실험 결과에 대한 피드백 평가 실행
        
        Args:
            original_experiment_dir: 원본 실험 디렉토리 경로
            template_id: 피드백 템플릿 ID
            temperature: 생성 다양성 (0~2)
            max_tokens: 생성할 최대 토큰 수
            
        Returns:
            피드백 실험 디렉토리 경로와 결과 목록의 튜플
        """
        # 원본 실험 디렉토리 확인
        if not os.path.exists(original_experiment_dir):
            raise FileNotFoundError(f"원본 실험 디렉토리를 찾을 수 없음: {original_experiment_dir}")
        
        # 원본 실험 요약 로드
        summary_path = os.path.join(original_experiment_dir, "experiment_summary.json")
        if not os.path.exists(summary_path):
            raise FileNotFoundError(f"실험 요약 파일을 찾을 수 없음: {summary_path}")
            
        with open(summary_path, 'r', encoding='utf-8') as f:
            original_summary = json.load(f)
            
        original_id = original_summary.get("experiment_id", "unknown")
        
        # 피드백 실험 디렉토리 설정
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        feedback_id = f"feedback_{original_id}_{timestamp}"
        feedback_dir = os.path.join(self.output_dir, feedback_id)
        os.makedirs(feedback_dir, exist_ok=True)
        
        logger.info(f"피드백 실험 시작: {feedback_id}")
        logger.info(f"원본 실험: {original_id}")
        
        # 피드백 템플릿 로드
        try:
            template = self.prompt_manager.get_template(template_id)
            logger.info(f"피드백 템플릿 로드됨: {template_id} v{template.version}")
        except Exception as e:
            logger.error(f"피드백 템플릿 로드 실패: {e}")
            raise
        
        # 원본 실험 결과 파일 찾기
        result_files = [f for f in os.listdir(original_experiment_dir) 
                        if f.endswith('.json') and f != "experiment_summary.json"]
        
        if not result_files:
            raise ValueError(f"원본 실험 디렉토리에 결과 파일이 없음: {original_experiment_dir}")
            
        logger.info(f"{len(result_files)} 개의 결과 파일 발견됨")
        
        # 피드백 결과 저장용 리스트
        feedback_results = []
        
        # 각 결과 파일에 대해 피드백 생성
        for result_file in result_files:
            result_path = os.path.join(original_experiment_dir, result_file)
            
            try:
                # 원본 결과 로드
                with open(result_path, 'r', encoding='utf-8') as f:
                    original_result = json.load(f)
                
                persona_id = original_result.get("persona_id", "unknown")
                original_template_id = original_result.get("template_id", "unknown")
                
                logger.info(f"피드백 생성 중: {result_file} (템플릿: {original_template_id}, 페르소나: {persona_id})")
                
                # 피드백 템플릿 데이터 준비
                template_data = {
                    "saju_reading": original_result.get("response", ""),
                    "birth_date": original_result.get("birth_date", ""),
                    "birth_time": original_result.get("birth_time", ""),
                    "birth_location": original_result.get("birth_location", ""),
                    "gender": original_result.get("gender", "")
                }
                
                # 피드백 프롬프트 포맷팅
                formatted_prompt = template.format(**template_data)
                
                # 피드백 생성
                start_time = time.time()
                response = self.llm_client.generate_saju_reading(
                    prompt=formatted_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                end_time = time.time()
                
                # 결과 기록
                feedback_result = {
                    "feedback_id": feedback_id,
                    "original_experiment_id": original_id,
                    "original_result_file": result_file,
                    "template_id": template_id,
                    "template_version": template.version,
                    "persona_id": persona_id,
                    "original_template_id": original_template_id,
                    "birth_date": template_data["birth_date"],
                    "birth_time": template_data["birth_time"],
                    "birth_location": template_data["birth_location"],
                    "gender": template_data["gender"],
                    "original_response": template_data["saju_reading"],
                    "feedback_prompt": formatted_prompt,
                    "feedback_response": response["text"],
                    "model": response["model"],
                    "provider": response.get("provider", "unknown"),
                    "prompt_tokens": response["usage"]["prompt_tokens"],
                    "completion_tokens": response["usage"]["completion_tokens"],
                    "total_tokens": response["usage"]["total_tokens"],
                    "finish_reason": response["finish_reason"],
                    "response_time": end_time - start_time,
                    "timestamp": datetime.now().isoformat()
                }
                
                feedback_results.append(feedback_result)
                
                # 개별 피드백 결과 저장
                feedback_filename = f"feedback_{original_template_id}_{persona_id}_{timestamp}.json"
                feedback_path = os.path.join(feedback_dir, feedback_filename)
                
                with open(feedback_path, 'w', encoding='utf-8') as f:
                    json.dump(feedback_result, f, indent=2, ensure_ascii=False)
                    
                logger.info(f"피드백 저장됨: {feedback_path}")
                
            except Exception as e:
                logger.error(f"파일 {result_file}에 대한 피드백 생성 중 오류: {e}")
                # 계속 진행
        
        # 전체 피드백 결과가 없으면 실패
        if not feedback_results:
            logger.error("피드백 실험 실패: 결과 없음")
            return feedback_dir, []
        
        # 모든 피드백 결과를 CSV로 저장
        try:
            feedback_df = pd.DataFrame(feedback_results)
            csv_path = os.path.join(feedback_dir, "all_feedback.csv")
            feedback_df.to_csv(csv_path, index=False)
            logger.info(f"CSV 피드백 결과 저장됨: {csv_path}")
        except Exception as e:
            logger.error(f"CSV 저장 실패: {e}")
        
        # 피드백 실험 요약 저장
        try:
            feedback_summary = {
                "feedback_id": feedback_id,
                "original_experiment_id": original_id,
                "template_id": template_id,
                "feedback_count": len(feedback_results),
                "total_tokens": sum(r["total_tokens"] for r in feedback_results),
                "average_response_time": sum(r["response_time"] for r in feedback_results) / max(1, len(feedback_results)),
                "timestamp": datetime.now().isoformat(),
                "provider": self.llm_client.provider_type,
                "model": self.llm_client.model
            }
            
            summary_path = os.path.join(feedback_dir, "feedback_summary.json")
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(feedback_summary, f, indent=2, ensure_ascii=False)
                
            logger.info(f"피드백 요약 저장됨: {summary_path}")
        except Exception as e:
            logger.error(f"요약 저장 실패: {e}")
        
        logger.info(f"피드백 실험 {feedback_id} 완료. 결과가 {feedback_dir}에 저장됨")
        
        return feedback_dir, feedback_results


# 기본 사용 예시
if __name__ == "__main__":
    # ExperimentRunner 초기화
    runner = ExperimentRunner(
        provider_type=ModelType.LLAMA,
        host="http://localhost:8000"
    )
    
    # 샘플 페르소나 로드 (10개만)
    personas = runner.load_personas(limit=10)
    
    # 기본 실험 실행
    experiment_dir, results = runner.run_experiment(
        experiment_name="basic_test",
        template_ids=["summary", "summary_ko"],
        personas=personas[:3]  # 첫 3개만 사용
    )
    
    # 피드백 실험 실행
    feedback_dir, feedback_results = runner.run_feedback_experiment(
        original_experiment_dir=experiment_dir,
        template_id="feedback_ko"  # 한국어 피드백 템플릿 사용
    ) 