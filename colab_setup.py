#!/usr/bin/env python3
"""
SajuMate Colab 설정 스크립트
코랩에서 SajuMate를 실행하기 위한 간단한 설정 스크립트입니다.
"""

import os
import sys
import subprocess
import argparse

def run_command(cmd, description=None):
    """명령어 실행 및 결과 출력"""
    if description:
        print(f"\n{'='*50}\n{description}\n{'='*50}")
    
    print(f"실행 중: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"오류 발생: {result.stderr}")
        return False
    
    print(result.stdout)
    return True

def main():
    parser = argparse.ArgumentParser(description="SajuMate Colab 설정")
    parser.add_argument('--install-only', action='store_true', help='의존성만 설치하고 앱을 실행하지 않음')
    parser.add_argument('--api-key', type=str, help='HuggingFace API 키')
    args = parser.parse_args()
    
    print("\n🔮 SajuMate Colab 설정을 시작합니다...")
    
    # 1. 필수 패키지 설치
    run_command("pip install python-dotenv gradio transformers", "필수 패키지 설치 중")
    
    # 2. 현재 디렉토리가 SajuMate 프로젝트인지 확인
    if not os.path.exists("app.py") and not os.path.exists("setup.py"):
        # 프로젝트 클론
        print("\n프로젝트 파일이 없습니다. GitHub에서 클론합니다...\n")
        repo_url = input("SajuMate GitHub 저장소 URL을 입력하세요: ")
        if not repo_url:
            repo_url = "https://github.com/yourusername/saju-prompt-mvp.git"  # 기본값 설정
        
        run_command(f"git clone {repo_url}", "저장소 클론 중")
        
        # 클론된 디렉토리로 이동
        repo_name = repo_url.split("/")[-1].replace(".git", "")
        os.chdir(repo_name)
        print(f"작업 디렉토리: {os.getcwd()}")
    
    # 3. 프로젝트 의존성 설치
    run_command("pip install -e .", "프로젝트 의존성 설치 중")
    
    # 4. API 키 설정 (제공된 경우)
    if args.api_key:
        os.environ["HUGGINGFACE_API_KEY"] = args.api_key
        print(f"\nHuggingFace API 키가 설정되었습니다.")
    
    # 5. 앱 실행 (--install-only가 아닌 경우)
    if not args.install_only:
        print("\n🚀 SajuMate 앱을 실행합니다...\n")
        run_command("python app.py --share", "앱 실행 중")
    else:
        print("\n✅ 설치가 완료되었습니다. 앱을 실행하려면 다음 명령어를 사용하세요:")
        print("python app.py --share")

if __name__ == "__main__":
    main() 