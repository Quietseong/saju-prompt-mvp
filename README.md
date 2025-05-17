[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/Quietseong/saju-prompt-mvp)

# 🔮 SajuMate - 한국 전통 사주 운세 생성 도구

SajuMate는 LLM(대규모 언어 모델)을 활용해 한국 전통 사주 운세를 생성하는 도구입니다. 이름과 생년월일시 정보를 입력하면 다양한 템플릿을 사용하여 개인화된 사주 해석을 제공합니다.

## ✨ 주요 기능

- 🧩 **다양한 LLM 지원**: OpenAI API, 허깅페이스 API, 로컬 Llama 모델 지원
- 📝 **템플릿 시스템**: 다양한 유형의 사주 해석을 위한 템플릿 제공 (요약, 상세, 피드백)
- 🌐 **다국어 지원**: 한국어 및 영어로 결과 생성
- 🖥️ **사용자 친화적 UI**: Gradio 기반의 직관적인 웹 인터페이스
- 💾 **샘플 페르소나**: 테스트용 샘플 페르소나 데이터 제공
- 💬 **다양한 말투/톤**: 챗봇 페르소나 선택 기능 지원
- 💻 **로컬 모델 최적화**: 4bit 양자화 및 성능 최적화 지원

## 🚀 설치 방법

### 방법 1: pip로 설치

```bash
# 기본 설치
pip install git+https://github.com/quietseong/sajumate.git

# 개발자 의존성 포함 설치
pip install git+https://github.com/quietseong/sajumate.git#egg=sajumate[dev]
```

### 방법 2: 소스코드에서 설치

```bash
# 저장소 복제
git clone https://github.com/quietseong/sajumate.git
cd sajumate

# 설치
pip install -e .
```

### 방법 3: Google Colab에서 실행

1. Google Colab에서 다음 코드를 실행하여 저장소를 클론합니다:

```python
!git clone https://github.com/quietseong/sajumate.git
%cd sajumate
```

2. 필요한 패키지를 설치합니다:

```python
!pip install -e .
```

3. 애플리케이션을 시작합니다 (공유 URL 활성화):

```python
!python app.py --share
```

또는 제공된 설정 스크립트를 사용할 수 있습니다:

```python
!python colab_setup.py
```

API 키를 함께 제공하려면:

```python
!python colab_setup.py --api-key="YOUR_HUGGINGFACE_API_KEY"
```

## 📊 사용 방법

### 1. 명령줄에서 실행

설치 후 다음 명령어로 웹 인터페이스를 시작할 수 있습니다:

```bash
# 기본 실행
sajumate

# 또는 모듈로 실행
python -m app
```

### 2. 로컬 개발 환경에서 실행

```bash
# 저장소 루트 디렉토리에서
python app.py
```

웹 브라우저에서 http://localhost:7860 으로 접속하여 SajuMate 인터페이스를 사용할 수 있습니다.

## 🔑 API 키 설정

SajuMate는 다양한 LLM 제공자를 지원합니다. 각 제공자의 API 키를 설정하는 방법은 다음과 같습니다:

### 환경변수를 통한 설정

`.env` 파일을 프로젝트 루트 디렉토리에 생성하고 다음 내용을 추가합니다:

```
# OpenAI
OPENAI_API_KEY=sk-your-openai-key

# HuggingFace
HUGGINGFACE_API_KEY=hf_your-huggingface-key

# 로컬 Llama (필요한 경우)
LLAMA_MODEL_PATH=/path/to/your/local/model
LLAMA_HOST=http://localhost:8000
```

### 웹 인터페이스를 통한 설정

또는 웹 인터페이스에서 직접 API 키를 입력할 수도 있습니다. 이 방법은 API 키가 로컬에 저장되지 않고 현재 세션에만 사용됩니다.

## 💡 API 선택 가이드

### 허깅페이스 API (권장)

허깅페이스 API는 다음과 같은 이유로 대부분의 사용자에게 권장됩니다:

- **비용 효율성**: OpenAI보다 일반적으로 저렴한 가격 정책
- **다양한 모델**: 다양한 오픈소스 LLM에 접근 가능
- **쉬운 설정**: 계정 생성과 API 키 발급이 간단함
- **맞춤 모델**: 미세 조정된 모델을 쉽게 테스트 가능

허깅페이스 API 키는 [huggingface.co](https://huggingface.co/settings/tokens)에서 무료로 받을 수 있습니다.

권장 모델: `meta-llama/Llama-3-8b-instruct`, `mistralai/Mistral-7B-Instruct-v0.2`

### OpenAI API

OpenAI API는 다음과 같은 경우에 적합합니다:

- **최고의 품질**: GPT-4 모델을 통한 최고 수준의 결과가 필요한 경우
- **안정성**: 안정적인 서비스와 응답 품질이 중요한 경우
- **다국어 지원**: 다양한 언어에 대한 우수한 지원이 필요한 경우

### 로컬 Llama

로컬 Llama 모델은 다음과 같은 경우에 적합합니다:

- **프라이버시**: 데이터를 외부로 전송하지 않아야 하는 경우
- **비용 절감**: API 요금 없이 무제한 사용이 필요한 경우
- **오프라인 사용**: 인터넷 연결 없이 사용해야 하는 경우

로컬 실행을 위해서는 충분한 컴퓨팅 자원(최소 8GB RAM, GPU 권장)이 필요합니다.

## 📚 사용자 정의

### 템플릿 추가

`prompts/templates/` 디렉토리에 새 JSON 템플릿 파일을 추가하여 사용자 정의 사주 해석 템플릿을 만들 수 있습니다.

### 챗봇 페르소나 추가

`data/tone_example.json` 파일에 새로운 페르소나 설정을 추가하여 사주 해석 결과의 말투와 스타일을 사용자화할 수 있습니다.

## 🤝 기여하기

프로젝트에 기여하는 방법:

1. 이 저장소를 포크합니다
2. 새로운 기능 브랜치를 만듭니다 (`git checkout -b feature/amazing-feature`)
3. 변경 사항을 커밋합니다 (`git commit -m 'Add some amazing feature'`)
4. 브랜치에 푸시합니다 (`git push origin feature/amazing-feature`)
5. Pull Request를 제출합니다

## 📄 라이선스

이 프로젝트는 MIT 라이선스에 따라 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.
