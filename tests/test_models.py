import pytest
from datetime import datetime
from src.models import SajuInput, Gender


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