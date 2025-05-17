from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional
import re

class Gender(Enum):
    """성별 Enum (남/여)"""
    MALE = "male"
    FEMALE = "female"

@dataclass
class BirthInfo:
    """
    생년월일시 정보를 담는 데이터 클래스
    
    Attributes:
        year (int): 출생 연도
        month (int): 출생 월
        day (int): 출생 일
        hour (int): 출생 시 (24시간 형식, 0-23)
        minute (int): 출생 분 (0-59)
    """
    year: int
    month: int
    day: int
    hour: int
    minute: int = 0
    
    def to_datetime(self) -> datetime:
        """BirthInfo를 datetime 객체로 변환"""
        return datetime(self.year, self.month, self.day, self.hour, self.minute)
    
    def to_dict(self) -> Dict[str, int]:
        """BirthInfo를 딕셔너리로 변환"""
        return {
            "year": self.year,
            "month": self.month,
            "day": self.day,
            "hour": self.hour,
            "minute": self.minute
        }

@dataclass
class PersonaData:
    """
    사용자 페르소나 데이터 클래스
    
    Attributes:
        id (str): 페르소나 고유 식별자
        name (str): 이름
        birth_info (BirthInfo): 생년월일시 정보
        gender (str): 성별 ('남성' 또는 '여성')
        location (Optional[str]): 출생지 위치 (선택)
        additional_info (Optional[Dict[str, Any]]): 추가 정보 (선택)
    """
    id: str
    name: str
    birth_info: BirthInfo
    gender: str
    location: Optional[str] = None
    additional_info: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """PersonaData를 딕셔너리로 변환"""
        result = {
            "id": self.id,
            "name": self.name,
            "birth_info": self.birth_info.to_dict(),
            "gender": self.gender
        }
        
        if self.location:
            result["location"] = self.location
            
        if self.additional_info:
            result["additional_info"] = self.additional_info
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PersonaData':
        """딕셔너리에서 PersonaData 객체 생성"""
        birth_info_data = data.get("birth_info", {})
        birth_info = BirthInfo(
            year=birth_info_data.get("year", 2000),
            month=birth_info_data.get("month", 1),
            day=birth_info_data.get("day", 1),
            hour=birth_info_data.get("hour", 0),
            minute=birth_info_data.get("minute", 0)
        )
        
        return cls(
            id=data.get("id", "unknown"),
            name=data.get("name", ""),
            birth_info=birth_info,
            gender=data.get("gender", "남성"),
            location=data.get("location"),
            additional_info=data.get("additional_info")
        )

@dataclass
class SajuInput:
    """
    사주 입력 데이터 모델

    Attributes:
        birth_date (datetime): 생년월일 (datetime 객체)
        birth_time (str): 출생 시간 (HH:MM 형식)
        birth_location (str): 출생지
        gender (Gender): 성별 (Gender Enum)

    Example:
        >>> from datetime import datetime
        >>> s = SajuInput(
        ...     birth_date=datetime(1990, 5, 15),
        ...     birth_time="13:45",
        ...     birth_location="서울",
        ...     gender=Gender.FEMALE
        ... )
        >>> s.validate()
        True
    """
    birth_date: datetime
    birth_time: str  # HH:MM format
    birth_location: str
    gender: Gender

    def validate(self) -> bool:
        """
        입력값의 유효성을 검증합니다.

        Returns:
            bool: 모든 값이 유효하면 True
        Raises:
            ValueError: 유효하지 않은 값이 있을 때 예외 발생
        """
        if not isinstance(self.birth_date, datetime):
            raise ValueError("birth_date는 datetime 객체여야 합니다.")
        if not re.match(r'^([01]?\d|2[0-3]):[0-5]\d$', self.birth_time):
            raise ValueError("birth_time은 HH:MM 형식이어야 합니다.")
        if not self.birth_location or not isinstance(self.birth_location, str):
            raise ValueError("birth_location은 비어있지 않은 문자열이어야 합니다.")
        if not isinstance(self.gender, Gender):
            raise ValueError("gender는 Gender Enum이어야 합니다.")
        return True

def parse_saju_input(data: Any) -> SajuInput:
    """
    다양한 입력(dict, str 등)에서 SajuInput 객체로 변환하고 검증합니다.

    Args:
        data (Any): 입력 데이터 (dict, JSON str 등)
    Returns:
        SajuInput: 검증된 SajuInput 객체
    Raises:
        ValueError: 필수 필드 누락, 포맷 오류, 변환 실패 등
    Example:
        >>> parse_saju_input({
        ...     'birth_date': '1990-05-15',
        ...     'birth_time': '13:45',
        ...     'birth_location': '서울',
        ...     'gender': 'female'
        ... })
        SajuInput(...)
    """
    import json
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except Exception as e:
            raise ValueError(f"입력 문자열을 JSON으로 파싱할 수 없습니다: {e}")
    if not isinstance(data, dict):
        raise ValueError("입력 데이터는 dict 또는 JSON 문자열이어야 합니다.")

    # 필수 필드 체크
    for field in ["birth_date", "birth_time", "birth_location", "gender"]:
        if field not in data:
            raise ValueError("필수 입력값이 누락되었습니다")

    # birth_date 처리
    birth_date_raw = data["birth_date"]
    if isinstance(birth_date_raw, datetime):
        birth_date = birth_date_raw
    else:
        try:
            birth_date = datetime.strptime(birth_date_raw, "%Y-%m-%d")
        except Exception:
            raise ValueError("birth_date는 'YYYY-MM-DD' 형식이어야 합니다.")

    # birth_time 처리
    birth_time = data["birth_time"]
    if not isinstance(birth_time, str):
        raise ValueError("birth_time은 문자열이어야 합니다.")

    # birth_location 처리
    birth_location = data["birth_location"]
    if not isinstance(birth_location, str):
        raise ValueError("birth_location은 문자열이어야 합니다.")

    # gender 처리
    gender_raw = data["gender"]
    try:
        if isinstance(gender_raw, Gender):
            gender = gender_raw
        else:
            gender_str = str(gender_raw).strip().lower()
            if gender_str in ("male", "남자"):
                gender = Gender.MALE
            elif gender_str in ("female", "여자"):
                gender = Gender.FEMALE
            else:
                raise ValueError
    except Exception:
        raise ValueError("gender는 'male', 'female', '남자', '여자' 중 하나여야 합니다.")

    saju = SajuInput(
        birth_date=birth_date,
        birth_time=birth_time,
        birth_location=birth_location,
        gender=gender
    )
    saju.validate()
    return saju