import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from datetime import datetime
from src.models import SajuInput, Gender, parse_saju_input


def test_valid_saju_input():
    """정상 입력값에 대해 SajuInput.validate가 True를 반환하는지 테스트합니다."""
    saju = SajuInput(
        birth_date=datetime(1990, 1, 1),
        birth_time="13:45",
        birth_location="Seoul, South Korea",
        gender=Gender.FEMALE
    )
    # 정상 입력은 True 반환
    assert saju.validate() is True


def test_invalid_time():
    """잘못된 시간 형식이 들어오면 ValueError가 발생하는지 테스트합니다."""
    saju = SajuInput(
        birth_date=datetime(1990, 1, 1),
        birth_time="25:00",  # 잘못된 시간
        birth_location="Seoul, South Korea",
        gender=Gender.FEMALE
    )
    # 잘못된 시간은 ValueError 발생
    with pytest.raises(ValueError):
        saju.validate()


def test_invalid_date():
    """birth_date가 datetime이 아니면 ValueError가 발생하는지 테스트합니다."""
    saju = SajuInput(
        birth_date="1990-01-01",  # 문자열로 잘못 입력
        birth_time="13:45",
        birth_location="Seoul, South Korea",
        gender=Gender.FEMALE
    )
    # 잘못된 날짜 타입은 ValueError 발생
    with pytest.raises(ValueError):
        saju.validate()


def test_invalid_gender():
    """gender가 Gender Enum이 아니면 ValueError가 발생하는지 테스트합니다."""
    saju = SajuInput(
        birth_date=datetime(1990, 1, 1),
        birth_time="13:45",
        birth_location="Seoul, South Korea",
        gender="female"  # 문자열로 잘못 입력
    )
    # 잘못된 gender 타입은 ValueError 발생
    with pytest.raises(ValueError):
        saju.validate()


def test_empty_location():
    """birth_location이 빈 문자열이면 ValueError가 발생하는지 테스트합니다."""
    saju = SajuInput(
        birth_date=datetime(1990, 1, 1),
        birth_time="13:45",
        birth_location="",  # 빈 문자열
        gender=Gender.FEMALE
    )
    # 빈 장소는 ValueError 발생
    with pytest.raises(ValueError):
        saju.validate()


# 정상 입력 케이스
@pytest.mark.parametrize(
    "input_data",
    [
        {
            "birth_date": "2000-02-29",  # 윤년
            "birth_time": "00:00",
            "birth_location": "seoul",
            "gender": "female"
        },
        {
            "birth_date": datetime(1999, 12, 31),
            "birth_time": "23:59",
            "birth_location": "new york",
            "gender": "male"
        },
        {
            "birth_date": "2023-01-01",
            "birth_time": "09:30",
            "birth_location": "tokyo",
            "gender": "female"
        },
    ]
)
def test_parse_saju_input_valid(input_data):
    saju = parse_saju_input(input_data)
    assert isinstance(saju, SajuInput)
    assert saju.validate() is True


# 잘못된 날짜/시간/성별/필드 누락 등 예외 케이스
@pytest.mark.parametrize(
    "input_data,expected_error",
    [
        # 잘못된 날짜
        ({"birth_date": "2023-02-30", "birth_time": "12:00", "birth_location": "서울", "gender": "female"}, "birth_date는 'YYYY-MM-DD' 형식이어야 합니다."),
        # 잘못된 시간
        ({"birth_date": "2023-01-01", "birth_time": "25:00", "birth_location": "서울", "gender": "female"}, "birth_time은 HH:MM 형식이어야 합니다."),
        # 잘못된 성별
        ({"birth_date": "2023-01-01", "birth_time": "12:00", "birth_location": "서울", "gender": "other"}, "gender는 'male', 'female', '남자', '여자' 중 하나여야 합니다."),
        # 필드 누락
        ({"birth_time": "12:00", "birth_location": "서울", "gender": "female"}, "필수 입력값이 누락되었습니다"),
        ({"birth_date": "2023-01-01", "birth_location": "서울", "gender": "female"}, "필수 입력값이 누락되었습니다"),
        ({"birth_date": "2023-01-01", "birth_time": "12:00", "gender": "female"}, "필수 입력값이 누락되었습니다"),
        ({"birth_date": "2023-01-01", "birth_time": "12:00", "birth_location": "서울"}, "필수 입력값이 누락되었습니다"),
    ]
)
def test_parse_saju_input_invalid(input_data, expected_error):
    with pytest.raises(ValueError) as exc:
        parse_saju_input(input_data)
    assert expected_error in str(exc.value)


# 문자열(JSON) 입력 테스트
def test_parse_saju_input_json_str():
    json_str = '{"birth_date": "2010-10-10", "birth_time": "10:10", "birth_location": "LA", "gender": "male"}'
    saju = parse_saju_input(json_str)
    assert isinstance(saju, SajuInput)
    assert saju.birth_location == "LA"
    assert saju.gender == Gender.MALE


# datetime 객체 입력 테스트
def test_parse_saju_input_datetime():
    saju = parse_saju_input({
        "birth_date": datetime(2020, 1, 1),
        "birth_time": "01:01",
        "birth_location": "paris",
        "gender": Gender.FEMALE
    })
    assert isinstance(saju, SajuInput)
    assert saju.birth_date == datetime(2020, 1, 1)
    assert saju.gender == Gender.FEMALE 