# src/data_normalizer.py

import re
from datetime import datetime
from typing import Dict, Optional, Tuple, Literal, Union, List, Any
import json
import os

GenderType = Literal["남자", "여자"]
CalendarType = Literal["양력", "음력"]

class SajuNormalizer:
    """사주 입력 데이터를 정규화하는 클래스"""
    
    def __init__(self, 
                manse_file: str = "data/manse_year.json", 
                month_ganji_file: str = "data/month_ganji.json",
                time_ganji_file: str = "data/time_ganji.json"):
        """
        Args:
            manse_file: 만세력 정보가 있는 JSON 파일 경로 (날짜-간지 매핑)
            month_ganji_file: 월간지 매핑 정보가 있는 JSON 파일 경로
            time_ganji_file: 시간간지 매핑 정보가 있는 JSON 파일 경로
        """
        self.manse_file = manse_file
        self.month_ganji_file = month_ganji_file
        self.time_ganji_file = time_ganji_file
        
        # 데이터 로드
        self.manse_data = self._load_manse_data()
        self.month_ganji_data = self._load_month_ganji_data()
        self.time_ganji_data = self._load_time_ganji_data()
    
    def _load_manse_data(self) -> List[Dict[str, Any]]:
        """만세력 매핑 데이터 로드 (날짜 -> 간지 변환용)"""
        # 파일이 존재하지 않으면 빈 리스트 반환 (MVP에서는 LLM에 맡김)
        if not os.path.exists(self.manse_file):
            return []
        
        with open(self.manse_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _load_month_ganji_data(self) -> Dict[str, Any]:
        """월간지 매핑 데이터 로드
        
        Returns:
            월간지 매핑 데이터. 파일이 없으면 빈 사전 반환
        """
        if not os.path.exists(self.month_ganji_file):
            print(f"월간지 매핑 파일이 없습니다: {self.month_ganji_file}")
            return {}
        
        with open(self.month_ganji_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _load_time_ganji_data(self) -> Dict[str, Any]:
        """시간간지 매핑 데이터 로드
        
        Returns:
            시간간지 매핑 데이터. 파일이 없으면 빈 사전 반환
        """
        if not os.path.exists(self.time_ganji_file):
            print(f"시간간지 매핑 파일이 없습니다: {self.time_ganji_file}")
            return {}
        
        with open(self.time_ganji_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _find_date_in_manse(self, year: int, month: int, day: int, 
                           is_lunar: bool = False) -> Optional[Dict[str, Any]]:
        """날짜에 해당하는 만세력 데이터 찾기
        
        Args:
            year: 년도
            month: 월
            day: 일
            is_lunar: 음력 여부 (True: 음력, False: 양력)
            
        Returns:
            해당 날짜의 만세력 데이터. 없으면 None 반환
        """
        # 만세력 데이터가 없으면 None 반환
        if not self.manse_data:
            return None
        
        for entry in self.manse_data:
            if is_lunar:
                # 음력 날짜로 검색
                if (int(entry.get("cd_ly", 0)) == year and 
                    int(entry.get("cd_lm", 0)) == month and 
                    int(entry.get("cd_ld", 0)) == day):
                    return entry
            else:
                # 양력 날짜로 검색
                if (int(entry.get("cd_sy", 0)) == year and 
                    int(entry.get("cd_sm", 0)) == month and 
                    int(entry.get("cd_sd", 0)) == day):
                    return entry
        
        return None
    
    def _parse_date(self, date_str: str) -> Tuple[int, int, int]:
        """다양한 형식의 날짜 문자열을 파싱하여 (년, 월, 일) 반환
        
        Args:
            date_str: 날짜 문자열 (YYYY-MM-DD, YYYYMMDD, YYYY/MM/DD 등 다양한 형식)
            
        Returns:
            (년, 월, 일) 튜플
            
        Raises:
            ValueError: 날짜 파싱 실패 시
        """
        # 날짜 형식 패턴들
        patterns = [
            # YYYY-MM-DD
            r'^(\d{4})-(\d{1,2})-(\d{1,2})$',
            # YYYY/MM/DD
            r'^(\d{4})/(\d{1,2})/(\d{1,2})$',
            # YYYY.MM.DD
            r'^(\d{4})\.(\d{1,2})\.(\d{1,2})$',
            # YYYYMMDD
            r'^(\d{4})(\d{2})(\d{2})$',
            # YYYY MM/DD
            r'^(\d{4})\s+(\d{1,2})/(\d{1,2})$',
            # YYYY MM-DD
            r'^(\d{4})\s+(\d{1,2})-(\d{1,2})$',
            # MM/DD/YYYY
            r'^(\d{1,2})/(\d{1,2})/(\d{4})$',
            # M/D/YYYY (미국식)
            r'^(\d{1,2})/(\d{1,2})/(\d{4})$',
        ]
        
        # 각 패턴을 시도하여 매칭되는 것 사용
        for pattern in patterns:
            match = re.match(pattern, date_str.strip())
            if match:
                groups = match.groups()
                # 패턴에 따라 다르게 처리 (MM/DD/YYYY 형식의 경우)
                if pattern == r'^(\d{1,2})/(\d{1,2})/(\d{4})$':
                    month, day, year = map(int, groups)
                else:
                    year, month, day = map(int, groups)
                return year, month, day
        
        # 표준 형식으로 변환 시도
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            return dt.year, dt.month, dt.day
        except ValueError:
            pass
        
        # 공백이나 구분자 제거 후 YYYYMMDD 형식으로 해석 시도
        cleaned = re.sub(r'[^\d]', '', date_str)
        if len(cleaned) == 8:  # YYYYMMDD
            year = int(cleaned[:4])
            month = int(cleaned[4:6])
            day = int(cleaned[6:8])
            return year, month, day
        
        # 모든 시도 실패 시 예외 발생
        raise ValueError(f"날짜 형식을 인식할 수 없습니다: {date_str}")
    
    def _parse_time(self, time_str: str) -> Tuple[int, int]:
        """시간 문자열을 파싱하여 (시, 분) 반환
        
        Args:
            time_str: 시간 문자열 (HH:MM, HH시MM분, H:M 등 다양한 형식)
            
        Returns:
            (시, 분) 튜플
            
        Raises:
            ValueError: 시간 파싱 실패 시
        """
        if not time_str:
            raise ValueError("시간 정보가 없습니다.")
        
        # 시간 형식 패턴들
        patterns = [
            # HH:MM
            r'^(\d{1,2}):(\d{1,2})$',
            # HH시 MM분
            r'^(\d{1,2})시\s*(\d{1,2})분$',
            # HH시
            r'^(\d{1,2})시$',
            # HH
            r'^(\d{1,2})$',
        ]
        
        # 각 패턴을 시도하여 매칭되는 것 사용
        for pattern in patterns:
            match = re.match(pattern, time_str.strip())
            if match:
                groups = match.groups()
                if len(groups) == 1:  # 시만 있는 경우
                    hour = int(groups[0])
                    minute = 0
                else:
                    hour, minute = map(int, groups)
                
                # 시간이 범위 내에 있는지 확인
                if 0 <= hour <= 23 and 0 <= minute <= 59:
                    return hour, minute
                else:
                    raise ValueError(f"시간 범위가 올바르지 않습니다: {time_str}")
        
        # 모든 시도 실패 시 예외 발생
        raise ValueError(f"시간 형식을 인식할 수 없습니다: {time_str}")
    
    def _find_time_branch(self, hour: int, minute: int) -> str:
        """시간에 해당하는 지지 찾기
        
        Args:
            hour: 시간 (0-23)
            minute: 분 (0-59)
            
        Returns:
            해당 시간의 지지(地支). 없으면 빈 문자열 반환
        """
        # 시간 문자열 생성 (HH:MM 형식)
        time_str = f"{hour:02d}:{minute:02d}"
        
        # 시간 범위 데이터
        time_ranges = self.time_ganji_data.get("time_branch_ranges", {})
        
        # 각 지지별 시간 범위 확인
        for branch, range_info in time_ranges.items():
            start_time = range_info.get("start", "")
            end_time = range_info.get("end", "")
            
            # 시작 시간과 종료 시간 파싱
            start_h, start_m = map(int, start_time.split(':'))
            end_h, end_m = map(int, end_time.split(':'))
            
            # 자정을 넘는 경우 (예: 23:00 - 01:00)
            if start_h > end_h:  # 23 > 1
                if (hour > start_h or hour < end_h) or (hour == start_h and minute >= start_m) or (hour == end_h and minute < end_m):
                    return branch
            # 일반적인 경우
            elif (start_h < hour < end_h) or (hour == start_h and minute >= start_m) or (hour == end_h and minute < end_m):
                return branch
        
        return ""  # 매칭되는 지지가 없는 경우
    
    def normalize(self, 
                 birth_date: str, 
                 birth_time: Optional[str] = None,
                 calendar_type: CalendarType = "양력",
                 gender: GenderType = "여자",
                 birthplace: str = "") -> str:
        """
        입력 데이터를 정규화된 사주 마킹 포맷으로 변환
        
        Args:
            birth_date: 다양한 형식의 생년월일 (YYYY-MM-DD, YYYYMMDD, YYYY/MM/DD 등)
            birth_time: 다양한 형식의 출생 시간 (HH:MM, HH시MM분, H:M 등)
            calendar_type: 양력 또는 음력
            gender: 남자 또는 여자
            birthplace: 출생지
            
        Returns:
            정규화된 사주 마킹 포맷 (예: "기해년 병신월 정축일 경인시, 양력, 여자, 서울")
        """
        try:
            # 날짜 파싱 (다양한 형식 지원)
            year, month, day = self._parse_date(birth_date)
            
            # 음력 여부 확인
            is_lunar = calendar_type == "음력"
            
            # 만세력 데이터에서 해당 날짜 정보 조회
            manse_info = self._find_date_in_manse(year, month, day, is_lunar)
            
            # 만세력 데이터가 있으면 간지 정보로 사주 생성
            if manse_info:
                # 시간 정보 파싱 (있는 경우)
                hour = minute = None
                if birth_time:
                    try:
                        hour, minute = self._parse_time(birth_time)
                    except ValueError:
                        # 시간 파싱 실패 시 None 유지
                        pass
                
                # 사주 생성 (만세력 데이터 + 시간 정보)
                saju_format = self._convert_to_saju_format(manse_info, hour, minute)
                
                # 띠 정보 추가
                ddi = manse_info.get("cd_ddi", "")
                ddi_info = f", 띠: {ddi}" if ddi else ""
                
                # 정규화된 포맷 생성
                normalized = f"{saju_format}, {calendar_type}, {gender}{ddi_info}"
            else:
                # 만세력 데이터가 없으면 기본 포맷 사용
                time_info = "시간 모름" if not birth_time else birth_time
                normalized = f"{year}-{month:02d}-{day:02d}, {time_info}, {calendar_type}, {gender}"
            
            # 출생지 정보가 있으면 추가
            if birthplace:
                normalized += f", {birthplace}"
                
            return normalized
            
        except Exception as e:
            # 예외 발생 시 가능한 한 원본 데이터 유지하여 기본 포맷 반환
            try:
                # 날짜 정보를 표준 형식으로 변환 시도
                year, month, day = self._parse_date(birth_date)
                formatted_date = f"{year}-{month:02d}-{day:02d}"
            except:
                # 변환 실패 시 원본 그대로 사용
                formatted_date = birth_date
                
            time_info = "시간 모름" if not birth_time else birth_time
            return f"{formatted_date}, {time_info}, {calendar_type}, {gender}{', ' + birthplace if birthplace else ''}"
    
    def _convert_to_saju_format(self, manse_info: Dict[str, Any], 
                               hour: Optional[int] = None, 
                               minute: Optional[int] = None) -> str:
        """
        만세력 데이터를 사용하여 사주 표기 생성
        
        Args:
            manse_info: 만세력 데이터 항목
            hour: 시간 (0-23)
            minute: 분 (0-59)
            
        Returns:
            사주 표기 문자열 (예: "기해년 병신월 정축일 경인시")
        """
        # 간지 정보 추출
        year_ganji = manse_info.get("cd_kyganjee", "")  # 년간지 (한글)
        day_ganji = manse_info.get("cd_kdganjee", "")   # 일간지 (한글)
        
        # 사주 구성 (년주, 월주, 일주, 시주)
        result = []
        
        # 1. 년주 (년간지)
        if year_ganji:
            result.append(f"{year_ganji}년")
        
        # 2. 월주 (월간지) - JSON 파일의 매핑 데이터 사용
        if year_ganji and self.month_ganji_data:
            # 년간지의 천간(첫 글자)
            year_stem = year_ganji[0]
            
            # 음력 월 (양력에서 음력으로 변환)
            lunar_month = int(manse_info.get("cd_lm", 0))
            
            # 월간 지지
            month_branch_map = self.month_ganji_data.get("month_branch_map", {})
            month_branch = month_branch_map.get(str(lunar_month), "")
            
            # 월간 천간
            month_stem_map = self.month_ganji_data.get("month_stem_map", {})
            key = f"{year_stem}-{lunar_month}"
            month_stem = month_stem_map.get(key, "")
            
            if month_stem and month_branch:
                result.append(f"{month_stem}{month_branch}월")
            else:
                result.append("월")  # 계산 실패 시 빈 값
        else:
            result.append("월")
        
        # 3. 일주 (일간지)
        if day_ganji:
            result.append(f"{day_ganji}일")
        
        # 4. 시주 (시간간지) - JSON 파일의 매핑 데이터 사용
        if hour is not None and day_ganji and self.time_ganji_data:
            # 일간의 천간(첫 글자) 추출
            day_stem = day_ganji[0]
            
            # 시간 지지 찾기
            time_branch = self._find_time_branch(hour, minute)
            
            # 시간 천간 찾기
            time_stem_map = self.time_ganji_data.get("time_stem_map", {})
            key = f"{day_stem}-{time_branch}"
            time_stem = time_stem_map.get(key, "")
            
            if time_stem and time_branch:
                result.append(f"{time_stem}{time_branch}시")
            else:
                # 시간 정보는 있지만 변환 실패 시 원시 시간 표시
                result.append(f"{hour:02d}:{minute:02d}시")
        else:
            result.append("시간 모름")
            
        return " ".join(result)
    
    def _convert_to_gan_ji(self, date_str: str, time_str: Optional[str] = None) -> str:
        """
        날짜와 시간을 간지 표기로 변환 (예: 기해년 병신월 정축일 경인시)
        
        Args:
            date_str: YYYY-MM-DD 형식의 날짜
            time_str: HH:MM 형식의 시간
            
        Returns:
            간지 표기 문자열
        """
        try:
            # 날짜 파싱
            year, month, day = self._parse_date(date_str)
            
            # 시간 파싱 (있는 경우)
            hour = minute = None
            if time_str:
                try:
                    hour, minute = self._parse_time(time_str)
                except ValueError:
                    pass
            
            # 만세력 데이터에서 해당 날짜 정보 조회 (양력 기준)
            manse_info = self._find_date_in_manse(year, month, day, False)
            
            # 만세력 데이터가 있으면 간지 정보 활용
            if manse_info:
                return self._convert_to_saju_format(manse_info, hour, minute)
            
            # 없으면 원본 반환
            time_part = f"{hour:02d}:{minute:02d}" if hour is not None else "시간 모름"
            return f"{year}-{month:02d}-{day:02d} {time_part}"
            
        except Exception as e:
            # 예외 발생 시 원본 반환
            return f"{date_str} {time_str or '시간 모름'}"