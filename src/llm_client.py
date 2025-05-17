import os
import time
import json
import logging
import asyncio
import requests
from typing import Dict, Any, Optional, Union, List, Literal
from datetime import datetime
from pathlib import Path
from abc import ABC, abstractmethod
from enum import Enum

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 환경변수 로드 (옵션)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    logger.warning("dotenv 패키지가 설치되지 않았습니다. 환경변수를 직접 설정하세요.")

# PyTorch 및 변환기 임포트 (Llama 로컬 양자화에 필요)
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
except ImportError:
    logger.warning("PyTorch 또는 transformers 패키지가 설치되지 않았습니다. LlamaProvider(로컬 모드)를 사용할 수 없습니다.")
    torch = None  # Prevent NameError

class ModelType(Enum):
    """LLM 모델 타입 Enum"""
    OPENAI = "openai"
    LLAMA = "llama"
    HUGGINGFACE = "huggingface"

class LLMProvider(ABC):
    """LLM 프로바이더 인터페이스"""
    
    @abstractmethod
    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1000, **kwargs) -> Dict[str, Any]:
        """텍스트 생성 메서드"""
        pass
    
    @abstractmethod
    async def agenerate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1000, **kwargs) -> Dict[str, Any]:
        """비동기 텍스트 생성 메서드"""
        pass
    
    @abstractmethod
    def get_token_limit(self) -> int:
        """모델의 최대 토큰 수 반환"""
        pass

class OpenAIProvider(LLMProvider):
    """OpenAI API 프로바이더"""
    def __init__(self, model: str = "gpt-4-turbo", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        # API 키 검증
        if not self.api_key:
            raise ValueError("OpenAI API 키가 필요합니다. API_KEY 매개변수를 제공하거나 OPENAI_API_KEY 환경변수를 설정하세요.")
        
        try:
            import openai
            self.client = openai.OpenAI(api_key=self.api_key)
            self.aclient = openai.AsyncOpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("openai 패키지가 설치되지 않았습니다. 'pip install openai' 명령으로 설치하세요.")
    
    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1000, **kwargs) -> Dict[str, Any]:
        start_time = time.time()
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            result = {
                "text": response.choices[0].message.content.strip(),
                "model": self.model,
                "latency": time.time() - start_time,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
            
            return result
        except Exception as e:
            logger.error(f"OpenAI API 오류: {e}")
            raise
    
    async def agenerate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1000, **kwargs) -> Dict[str, Any]:
        start_time = time.time()
        
        try:
            response = await self.aclient.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            result = {
                "text": response.choices[0].message.content.strip(),
                "model": self.model,
                "latency": time.time() - start_time,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
            
            return result
        except Exception as e:
            logger.error(f"OpenAI API 오류: {e}")
            raise
    
    def get_token_limit(self) -> int:
        """모델의 최대 토큰 수 반환"""
        model_limits = {
            "gpt-3.5-turbo": 4096,
            "gpt-3.5-turbo-16k": 16384,
            "gpt-4": 8192,
            "gpt-4-32k": 32768,
            "gpt-4-turbo": 128000,
            "gpt-4o": 128000
        }
        
        for model_prefix, limit in model_limits.items():
            if self.model.startswith(model_prefix):
                return limit
        
        # 기본값
        return 4096

class LlamaProvider(LLMProvider):
    """Llama 모델 프로바이더 (로컬 또는 API)"""
    def __init__(
        self, 
        model: str = "llama-3-8b-instruct", 
        model_path: Optional[str] = None, 
        host: Optional[str] = None,
        load_in_4bit: bool = False,
        device: str = "auto"
    ):
        self.model = model
        self.model_path = model_path or os.getenv("LLAMA_MODEL_PATH")
        self.host = host or os.getenv("LLAMA_HOST")
        self.load_in_4bit = load_in_4bit
        
        # 디바이스 설정
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # API vs 로컬 결정
        self.use_api = bool(self.host and not self.model_path)
        
        if not self.use_api and not self.model_path:
            logger.warning("모델 경로가 제공되지 않아 호스트가 지정된 경우 API 모드로 설정됩니다.")
            self.use_api = bool(self.host)
        
        if not self.use_api and not self.model_path:
            raise ValueError("Llama 사용을 위해서는 model_path 또는 host가 필요합니다.")
        
        # 로컬 모델 로드 (필요한 경우)
        self.local_model = None
        self.local_tokenizer = None
        
        if not self.use_api and self.model_path:
            self._load_local_model()
    
    def _load_local_model(self):
        """로컬 모델 로드"""
        try:
            logger.info(f"Llama 모델 로드 중: {self.model_path}, 디바이스: {self.device}")
            
            # 4비트 양자화 설정
            if self.load_in_4bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                
                self.local_model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    device_map=self.device,
                    trust_remote_code=True,
                    quantization_config=quantization_config
                )
            else:
                self.local_model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    device_map=self.device,
                    trust_remote_code=True
                )
            
            self.local_tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            logger.info("Llama 모델 로드 완료")
        except Exception as e:
            logger.error(f"모델 로드 오류: {e}")
            raise
    
    def _api_inference(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1000, **kwargs) -> Dict[str, Any]:
        """API를 통한 추론"""
        start_time = time.time()
        
        payload = {
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }
        
        url = f"{self.host.rstrip('/')}/generate"
        
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            
            result = response.json()
            result["latency"] = time.time() - start_time
            result["model"] = self.model
            
            # 표준화된 응답 형식으로 변환
            if "text" not in result and "generation" in result:
                result["text"] = result.pop("generation")
            
            # 토큰 사용량 추가 (있는 경우)
            if "usage" not in result:
                result["usage"] = {}
            
            return result
        except requests.RequestException as e:
            logger.error(f"API 호출 오류: {e}")
            raise
    
    def _local_inference(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1000, **kwargs) -> Dict[str, Any]:
        """로컬 모델을 사용한 추론"""
        start_time = time.time()
        
        if not self.local_model or not self.local_tokenizer:
            self._load_local_model()
        
        try:
            # 토큰화 및 토큰 수 계산
            input_tokens = self.local_tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            input_token_count = input_tokens.shape[1]
            
            # 파이프라인 설정
            generation_config = {
                "temperature": temperature,
                "max_new_tokens": max_tokens,
                "top_p": kwargs.get("top_p", 0.9),
                "do_sample": temperature > 0.0,
                **{k: v for k, v in kwargs.items() if k not in ["top_p"]}
            }
            
            # 추론 실행
            with torch.no_grad():
                output = self.local_model.generate(
                    input_tokens,
                    **generation_config
                )
            
            # 결과 디코딩 및 프롬프트 제거
            full_output = self.local_tokenizer.decode(output[0], skip_special_tokens=True)
            result_text = full_output[len(prompt):].strip()
            
            # 출력 토큰 추정
            output_token_count = len(self.local_tokenizer.encode(result_text))
            
            result = {
                "text": result_text,
                "model": self.model,
                "latency": time.time() - start_time,
                "usage": {
                    "prompt_tokens": input_token_count,
                    "completion_tokens": output_token_count,
                    "total_tokens": input_token_count + output_token_count
                }
            }
            
            return result
        except Exception as e:
            logger.error(f"로컬 추론 오류: {e}")
            raise
    
    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1000, **kwargs) -> Dict[str, Any]:
        """텍스트 생성 메서드"""
        if self.use_api:
            return self._api_inference(prompt, temperature, max_tokens, **kwargs)
        else:
            return self._local_inference(prompt, temperature, max_tokens, **kwargs)
    
    async def agenerate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1000, **kwargs) -> Dict[str, Any]:
        """비동기 텍스트 생성 메서드"""
        # API 요청은 비동기로 처리하고, 로컬 모델 사용은 스레드 풀에서 오프로드
        if self.use_api:
            return await asyncio.to_thread(self._api_inference, prompt, temperature, max_tokens, **kwargs)
        else:
            return await asyncio.to_thread(self._local_inference, prompt, temperature, max_tokens, **kwargs)
    
    def get_token_limit(self) -> int:
        """모델의 최대 토큰 수 반환"""
        model_limits = {
            "llama-3-70b": 8192,
            "llama-3-8b": 8192,
            "llama-2-70b": 4096,
            "llama-2-13b": 4096,
            "llama-2-7b": 4096,
        }
        
        for model_prefix, limit in model_limits.items():
            if self.model.startswith(model_prefix):
                return limit
        
        # 기본값
        return 4096

class HuggingFaceProvider(LLMProvider):
    """허깅페이스 API 프로바이더"""
    def __init__(self, model: str = "meta-llama/Llama-3-8b-instruct", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.getenv("HUGGINGFACE_API_KEY")
        
        # API 키 검증
        if not self.api_key:
            raise ValueError("HuggingFace API 키가 필요합니다. api_key 매개변수를 제공하거나 HUGGINGFACE_API_KEY 환경변수를 설정하세요.")
            
        # 모델 이름에서 태스크 감지 (추론 API에서 중요)
        self.is_chat_model = any(kw in model.lower() for kw in ["chat", "instruct"])
        
        # API 엔드포인트 설정
        self.api_url = f"https://api-inference.huggingface.co/models/{model}"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def _format_prompt_for_chat(self, prompt: str) -> Dict[str, Any]:
        """채팅 모델용 프롬프트 포맷팅"""
        if self.is_chat_model:
            return {
                "inputs": {
                    "messages": [
                        {"role": "user", "content": prompt}
                    ]
                }
            }
        else:
            return {"inputs": prompt}
    
    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1000, **kwargs) -> Dict[str, Any]:
        """텍스트 생성 메서드"""
        start_time = time.time()
        
        payload = self._format_prompt_for_chat(prompt)
        # 매개변수 추가
        payload["parameters"] = {
            "temperature": temperature,
            "max_new_tokens": max_tokens,
            "return_full_text": False,
            **kwargs
        }
        
        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            response.raise_for_status()
            data = response.json()
            
            # 응답 파싱 (채팅 모델과 일반 모델 응답 형식이 다름)
            result_text = ""
            if isinstance(data, list) and len(data) > 0:
                if self.is_chat_model and "generated_text" in data[0]:
                    result_text = data[0]["generated_text"]
                elif "message" in data[0] and "content" in data[0]["message"]:
                    result_text = data[0]["message"]["content"]
                elif "generated_text" in data[0]:
                    result_text = data[0]["generated_text"]
                else:
                    result_text = str(data[0])
            
            # 예상 토큰 수 계산 (정확하지 않지만 대략적인 추정)
            # 영어 기준으로 단어 하나가 약 1.3 토큰이라고 가정
            prompt_tokens = len(prompt.split()) * 1.3
            completion_tokens = len(result_text.split()) * 1.3
            
            result = {
                "text": result_text.strip(),
                "model": self.model,
                "latency": time.time() - start_time,
                "usage": {
                    "prompt_tokens": int(prompt_tokens),
                    "completion_tokens": int(completion_tokens),
                    "total_tokens": int(prompt_tokens + completion_tokens)
                }
            }
            
            return result
        except requests.RequestException as e:
            logger.error(f"HuggingFace API 오류: {e}")
            raise
    
    async def agenerate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1000, **kwargs) -> Dict[str, Any]:
        """비동기 텍스트 생성 메서드"""
        # 비동기 구현 (requests를 이용한 동기 함수를 스레드 풀에서 실행)
        return await asyncio.to_thread(self.generate, prompt, temperature, max_tokens, **kwargs)
    
    def get_token_limit(self) -> int:
        """모델의 최대 토큰 수 반환"""
        # 허깅페이스 모델별 토큰 한도 (대략적인 값)
        model_limits = {
            "llama-3-70b": 8192,
            "llama-3-8b": 8192, 
            "llama-2-70b": 4096,
            "llama-2-13b": 4096,
            "llama-2-7b": 4096,
            "mixtral-8x7b": 32768,
            "mistral-7b": 8192
        }
        
        # 모델 이름에서 키워드 확인
        for model_keyword, limit in model_limits.items():
            if model_keyword.lower() in self.model.lower():
                return limit
        
        # 기본값
        return 2048

class LLMClient:
    """LLM 클라이언트 - 다양한 프로바이더를 관리하는 통합 인터페이스"""
    
    def __init__(
        self,
        provider_type: ModelType,
        model: str,
        api_key: Optional[str] = None,
        model_path: Optional[str] = None,
        host: Optional[str] = None,
        load_in_4bit: bool = False,
        device: str = "auto"
    ):
        """
        LLM 클라이언트 초기화
        
        Args:
            provider_type: 사용할 프로바이더 타입 (ModelType)
            model: 모델 이름
            api_key: API 키 (필요한 경우)
            model_path: 로컬 모델 경로 (Llama 로컬 용)
            host: API 호스트 (Llama API 용)
            load_in_4bit: 4비트 양자화 사용 여부 (Llama 로컬 용)
            device: 사용할 디바이스 (Llama 로컬 용)
        """
        self.provider_type = provider_type
        self.model = model
        
        # 프로바이더 초기화
        if provider_type == ModelType.OPENAI:
            self.provider = OpenAIProvider(model=model, api_key=api_key)
        elif provider_type == ModelType.LLAMA:
            self.provider = LlamaProvider(
                model=model,
                model_path=model_path,
                host=host,
                load_in_4bit=load_in_4bit,
                device=device
            )
        elif provider_type == ModelType.HUGGINGFACE:
            self.provider = HuggingFaceProvider(model=model, api_key=api_key)
        else:
            raise ValueError(f"지원되지 않는 프로바이더 타입: {provider_type}")
    
    def generate_saju_reading(
        self, 
        prompt: str, 
        temperature: float = 0.7, 
        max_tokens: int = 1000, 
        **kwargs
    ) -> Dict[str, Any]:
        """
        사주 해석 생성
        
        Args:
            prompt: 프롬프트 내용
            temperature: 온도 (크리에이티브 정도)
            max_tokens: 최대 토큰 수
            **kwargs: 부가 옵션 (프로바이더마다 다름)
            
        Returns:
            생성 결과를 포함한 딕셔너리
        """
        try:
            return self.provider.generate(prompt, temperature, max_tokens, **kwargs)
        except Exception as e:
            logger.error(f"사주 해석 생성 오류: {e}")
            
            # 오류 결과 반환
            return {
                "text": f"사주 해석 생성 중 오류가 발생했습니다: {str(e)}",
                "model": self.model,
                "error": str(e)
            }
    
    async def agenerate_saju_reading(
        self, 
        prompt: str, 
        temperature: float = 0.7, 
        max_tokens: int = 1000, 
        **kwargs
    ) -> Dict[str, Any]:
        """
        비동기 사주 해석 생성
        
        Args:
            prompt: 프롬프트 내용
            temperature: 온도 (크리에이티브 정도)
            max_tokens: 최대 토큰 수
            **kwargs: 부가 옵션 (프로바이더마다 다름)
            
        Returns:
            생성 결과를 포함한 딕셔너리
        """
        try:
            return await self.provider.agenerate(prompt, temperature, max_tokens, **kwargs)
        except Exception as e:
            logger.error(f"비동기 사주 해석 생성 오류: {e}")
            
            # 오류 결과 반환
            return {
                "text": f"사주 해석 생성 중 오류가 발생했습니다: {str(e)}",
                "model": self.model,
                "error": str(e)
            }
    
    def get_token_limit(self) -> int:
        """현재 모델의 최대 토큰 수 반환"""
        return self.provider.get_token_limit()

# 유틸리티 함수
def save_response(response: Dict[str, Any], output_dir: str = "results/responses") -> str:
    """
    LLM 응답을 저장하는 유틸리티 함수
    
    Args:
        response: LLM 응답 딕셔너리
        output_dir: 저장할 디렉토리
        
    Returns:
        저장된 파일 경로
    """
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    provider = response.get("provider", "unknown")
    model = response.get("model", "unknown").split("/")[-1]  # 경로에서 모델 이름만 추출
    
    filename = f"{provider}_{model}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(response, f, ensure_ascii=False, indent=2)
    
    logger.info(f"응답이 '{filepath}'에 저장되었습니다.")
    return filepath

# 기본 사용 예시
if __name__ == "__main__":
    # Llama 모델 사용 (기본값)
    llm_client = LLMClient(
         provider_type=ModelType.LLAMA,
         model="llama-3-8b-instruct",
         host="http://localhost:8000",  # Llama 서버 주소
     )
    
    # 간단한 테스트
    sample_prompt = "1990년 3월 15일 오전 9시에 태어난 남성의 사주를 간략하게 해석해주세요."
    
    try:
        response = llm_client.generate_saju_reading(sample_prompt)
        print(f"응답: {response['text'][:100]}...")  # 응답 일부 출력
        print(f"토큰 사용량: {response['usage']}")
        print(f"응답 시간: {response.get('latency', 'N/A')}초")
        
        # 응답 저장
        save_response(response)
    except Exception as e:
        logger.error(f"테스트 실패: {str(e)}")
        
    # 비동기 사용 예시는 생략 (실제 사용 시 구현 필요) 