import os
import time
import logging
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
import torch

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BaseLLMClient(ABC):
    """LLM 클라이언트 추상 클래스. 모든 LLM 클라이언트는 이 인터페이스를 따라야 함."""

    @abstractmethod
    def generate_completion(self, prompt: str, temperature: float = 0.7, max_tokens: int = 2000) -> Dict[str, Any]:
        """
        LLM에 프롬프트를 입력해 응답을 생성한다.

        Args:
            prompt (str): 입력 프롬프트
            temperature (float): 창의성 조절 파라미터
            max_tokens (int): 최대 토큰 수
        Returns:
            Dict[str, Any]: 응답 텍스트, 모델명, 토큰 사용량 등
        """
        pass

class GPTClient(BaseLLMClient):
    """OpenAI GPT API용 클라이언트"""
    def __init__(self, model: str = None, max_retries: int = 3, retry_delay: int = 2):
        import openai
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY 환경변수가 설정되어 있지 않습니다.")
        self.client = openai.OpenAI(api_key=self.api_key)
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4-turbo")
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def generate_completion(self, prompt: str, temperature: float = 0.7, max_tokens: int = 2000) -> Dict[str, Any]:
        """
        OpenAI GPT API를 사용해 프롬프트에 대한 응답을 생성한다.
        """
        attempts = 0
        while attempts < self.max_retries:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a Saju fortune telling expert."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return {
                    "text": response.choices[0].message.content,
                    "model": self.model,
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    },
                    "finish_reason": response.choices[0].finish_reason,
                    "timestamp": time.time()
                }
            except Exception as e:
                attempts += 1
                logger.warning(f"OpenAI API 오류: {str(e)} (시도 {attempts}/{self.max_retries})")
                if attempts < self.max_retries:
                    time.sleep(self.retry_delay * attempts)
                else:
                    logger.error(f"OpenAI API {self.max_retries}회 실패: {str(e)}")
                    raise

    def generate_chat_completion(self, messages: list[dict[str, str]], temperature: float = 0.7, max_tokens: int = 2000) -> Dict[str, Any]:
        """
        OpenAI GPT API를 사용해 대화 히스토리(messages) 기반으로 응답을 생성한다.

        Args:
            messages (list[dict[str, str]]): 대화 히스토리 (role: user/assistant/system, content)
            temperature (float): 창의성 조절 파라미터
            max_tokens (int): 최대 토큰 수
        Returns:
            Dict[str, Any]: 응답 텍스트, 모델명, 토큰 사용량 등
        """
        attempts = 0
        while attempts < self.max_retries:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return {
                    "text": response.choices[0].message.content,
                    "model": self.model,
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    },
                    "finish_reason": response.choices[0].finish_reason,
                    "timestamp": time.time()
                }
            except Exception as e:
                attempts += 1
                logger.warning(f"OpenAI API 오류: {str(e)} (시도 {attempts}/{self.max_retries})")
                if attempts < self.max_retries:
                    time.sleep(self.retry_delay * attempts)
                else:
                    logger.error(f"OpenAI API {self.max_retries}회 실패: {str(e)}")
                    raise

class SLLMClient(BaseLLMClient):
    """Ollama 등 self-hosted LLM용 클라이언트"""
    def __init__(self, model: str = None, base_url: str = None):
        import ollama
        self.model = model or os.getenv("SLLM_MODEL", "llama3")
        self.base_url = base_url or os.getenv("SLLM_BASE_URL", "http://localhost:11434")
        self.ollama = ollama
        # ollama 패키지는 환경변수로 base_url을 읽음

    def generate_completion(self, prompt: str, temperature: float = 0.7, max_tokens: int = 2000) -> Dict[str, Any]:
        """
        Ollama API를 사용해 프롬프트에 대한 응답을 생성한다.
        """
        try:
            response = self.ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                stream=False
            )
            return {
                "text": response['message']['content'],
                "model": self.model,
                "usage": {},  # Ollama는 토큰 사용량 미제공
                "finish_reason": response.get('done', 'stop'),
                "timestamp": time.time()
            }
        except Exception as e:
            logger.error(f"SLLM(Ollama) 오류: {str(e)}")
            raise

class Llama3Client(BaseLLMClient):
    """Meta-Llama-3-8B-Instruct (PyTorch, 4bit QLoRA) 모델용 클라이언트"""
    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        device: str = "auto",
        load_in_4bit: bool = True,
        torch_dtype: torch.dtype = torch.float16,
        quantization_config: Optional[BitsAndBytesConfig] = None
    ):
        """
        Meta-Llama-3-8B-Instruct PyTorch 모델을 4bit QLoRA(양자화)로 로드한다.

        Args:
            model_name (str): HuggingFace 모델명
            device (str): 'auto', 'cpu', 또는 'cuda' (기본값: 'auto')
            load_in_4bit (bool): 4bit 양자화 적용 여부
            torch_dtype (torch.dtype): 텐서 데이터 타입 (기본값: float16)
            quantization_config (BitsAndBytesConfig, optional): 커스텀 양자화 설정
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if quantization_config is None and load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch_dtype
            )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device,
            use_flash_attention_2=False,
            torch_dtype=torch_dtype,
            quantization_config=quantization_config if load_in_4bit else None
        )
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map=device,
            max_new_tokens=1024
        )

    def generate_completion(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1024) -> Dict[str, Any]:
        """
        Meta-Llama-3-8B-Instruct (4bit QLoRA) 모델을 사용해 프롬프트에 대한 응답을 생성한다.
        """
        # Llama-3의 chat 포맷에 맞게 프롬프트 구성
        chat_prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{prompt}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n"
        result = self.pipe(
            chat_prompt,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_tokens,
            return_full_text=False
        )
        return {
            "text": result[0]["generated_text"],
            "model": self.model_name,
            "usage": {},
        }

    def chat_with_history(
        self,
        history: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024
    ) -> Dict[str, Any]:
        """
        대화 히스토리(역할별 메시지 리스트)를 받아 챗봇처럼 응답을 생성한다.

        Args:
            history (list[dict[str, str]]): 대화 히스토리 (role: user/assistant, content)
            temperature (float): 창의성 조절 파라미터
            max_tokens (int): 최대 토큰 수
        Returns:
            Dict[str, Any]: 응답 텍스트, 모델명 등
        """
        # Llama-3 chat 포맷으로 프롬프트 누적 생성
        prompt = "<|begin_of_text|>"
        for turn in history:
            if turn["role"] == "user":
                prompt += "<|start_header_id|>user<|end_header_id|>\n" + turn["content"] + "<|eot_id|>\n"
            elif turn["role"] == "assistant":
                prompt += "<|start_header_id|>assistant<|end_header_id|>\n" + turn["content"] + "<|eot_id|>\n"
        prompt += "<|start_header_id|>assistant<|end_header_id|>\n"
        # 한글 주석: 대화 히스토리를 누적하여 Llama-3 chat 포맷으로 변환
        result = self.pipe(
            prompt,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_tokens,
            return_full_text=False
        )
        return {
            "text": result[0]["generated_text"],
            "model": self.model_name,
            "usage": {},
        }

def get_llm_client(llm_type: Optional[str] = None) -> BaseLLMClient:
    """
    환경변수 또는 인자로 LLM 종류를 받아 적절한 클라이언트 인스턴스를 반환한다.

    Args:
        llm_type (Optional[str]): 'openai', 'sllm', 'llama3' (기본값: 환경변수 LLM_TYPE)
    Returns:
        BaseLLMClient: LLM 클라이언트 인스턴스
    """
    llm_type = llm_type or os.getenv("LLM_TYPE", "openai").lower()
    print("DEBUG llm_type:", llm_type)
    if llm_type == "openai":
        return GPTClient()
    elif llm_type == "sllm":
        return SLLMClient()
    elif llm_type == "llama3":
        return Llama3Client()
    else:
        raise ValueError(f"지원하지 않는 LLM_TYPE: {llm_type}")
