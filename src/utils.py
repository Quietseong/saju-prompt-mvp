from datetime import datetime
from typing import Optional
import pytz
import json
import os
from calendar import monthrange
from src.models import BirthInfo

CITY_TZ_MAP_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "city_timezone_map.json")

# 캐싱을 위한 전역 변수
_city_timezone_map_cache = None

DEFAULT_TIMEZONE = "America/New_York"  # fallback 용도만 사용

def validate_birth_info(birth_info: BirthInfo) -> bool:
    """
    생년월일시 정보의 유효성을 검증합니다.
    
    Args:
        birth_info (BirthInfo): 검증할 생년월일시 정보
    
    Returns:
        bool: 유효한 경우 True
    
    Raises:
        ValueError: 유효하지 않은 경우 예외 발생
    """
    # 연도 검증 (1900년 이후, 현재 이전)
    current_year = datetime.now().year
    if birth_info.year < 1900 or birth_info.year > current_year:
        raise ValueError(f"유효하지 않은 연도입니다: {birth_info.year}. 1900년부터 {current_year}년 사이여야 합니다.")
    
    # 월 검증 (1-12)
    if birth_info.month < 1 or birth_info.month > 12:
        raise ValueError(f"유효하지 않은 월입니다: {birth_info.month}. 1에서 12 사이여야 합니다.")
    
    # 해당 월의 최대 일수 계산
    days_in_month = monthrange(birth_info.year, birth_info.month)[1]
    
    # 일 검증 (1부터 해당 월의 최대 일수까지)
    if birth_info.day < 1 or birth_info.day > days_in_month:
        raise ValueError(f"유효하지 않은 일입니다: {birth_info.day}. 1에서 {days_in_month} 사이여야 합니다.")
    
    # 시간 검증 (0-23)
    if birth_info.hour < 0 or birth_info.hour > 23:
        raise ValueError(f"유효하지 않은 시간입니다: {birth_info.hour}. 0에서 23 사이여야 합니다.")
    
    # 분 검증 (0-59)
    if birth_info.minute < 0 or birth_info.minute > 59:
        raise ValueError(f"유효하지 않은 분입니다: {birth_info.minute}. 0에서 59 사이여야 합니다.")
    
    # 모든 검증 통과
    return True

def load_city_timezone_map() -> dict:
    """
    도시명-타임존 매핑 데이터를 JSON 파일에서 로드합니다.
    Returns:
        dict: 도시명(str) -> 타임존(str) 매핑
    Raises:
        FileNotFoundError: 매핑 파일이 없을 때
        json.JSONDecodeError: JSON 파싱 실패 시
    """
    global _city_timezone_map_cache
    if _city_timezone_map_cache is not None:
        return _city_timezone_map_cache
    with open(CITY_TZ_MAP_PATH, "r", encoding="utf-8") as f:
        _city_timezone_map_cache = json.load(f)
    return _city_timezone_map_cache

def get_timezone_from_location(location: Optional[str]) -> str:
    """
    출생지(도시명)로부터 타임존 문자열을 반환합니다.
    Args:
        location (str): 도시명 (한글 또는 영문)
    Returns:
        str: IANA 타임존 문자열 (예: 'Asia/Seoul')
    Notes:
        location이 None이거나 매핑이 없으면 DEFAULT_TIMEZONE 반환
    """
    if not location:
        return DEFAULT_TIMEZONE
    city_map = load_city_timezone_map()
    key = location.strip().lower()
    tz = city_map.get(key)
    if not tz:
        return DEFAULT_TIMEZONE
    return tz

def to_local_timezone(dt: datetime, location: Optional[str]) -> datetime:
    """
    출생지에 맞는 타임존으로 datetime을 변환합니다.
    Args:
        dt (datetime): 변환할 datetime 객체 (naive 또는 aware)
        location (str): 도시명 (한글 또는 영문, None 가능)
    Returns:
        datetime: 해당 지역 타임존의 datetime 객체 (없으면 Asia/Seoul)
    """
    tz_name = get_timezone_from_location(location)
    tz = pytz.timezone(tz_name)
    if dt.tzinfo is None:
        return tz.localize(dt)
    return dt.astimezone(tz)

def to_default_timezone(dt: datetime, tz_name: Optional[str] = None) -> datetime:
    """
    입력 datetime을 DEFAULT_TIMEZONE(America/New_York)로 변환합니다.

    Args:
        dt (datetime): 변환할 datetime 객체 (timezone-aware 또는 naive)
        tz_name (Optional[str]): 입력 datetime의 타임존 이름 (예: 'UTC', 'Asia/Seoul').
            None이면 naive로 간주하고 DEFAULT_TIMEZONE으로 직접 변환

    Returns:
        datetime: DEFAULT_TIMEZONE 타임존의 datetime 객체

    Raises:
        ValueError: tz_name이 잘못되었거나 변환 실패 시

    Example:
        >>> from datetime import datetime
        >>> import pytz
        >>> dt_utc = datetime(2023, 5, 1, 12, 0, tzinfo=pytz.UTC)
        >>> to_default_timezone(dt_utc)
        datetime.datetime(2023, 5, 1, 8, 0, tzinfo=<DstTzInfo 'America/New_York' EDT-1 day, 20:00:00 DST>)
    """
    tz = pytz.timezone(DEFAULT_TIMEZONE)
    if dt.tzinfo is None:
        # naive datetime: 입력 타임존이 있으면 적용, 없으면 DEFAULT_TIMEZONE으로 간주
        if tz_name:
            try:
                input_tz = pytz.timezone(tz_name)
            except Exception as e:
                raise ValueError(f"알 수 없는 타임존: {tz_name}") from e
            dt = input_tz.localize(dt)
        else:
            return tz.localize(dt)
    # aware datetime: 입력 타임존에서 DEFAULT_TIMEZONE으로 변환
    return dt.astimezone(tz)

def convert_to_timezone(dt: datetime, timezone_str: str) -> datetime:
    """Convert naive datetime to a specific timezone-aware datetime."""
    tz = pytz.timezone(timezone_str)
    return tz.localize(dt)