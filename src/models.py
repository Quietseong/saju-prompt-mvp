from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import re
from typing import Dict

class Gender(Enum):
    MALE = "male"
    FEMALE = "female"

@dataclass
class SajuInput:
    birth_date: datetime
    birth_time: str  # HH:MM format
    birth_location: str
    gender: Gender

    def validate(self) -> bool:
        """Validate all input fields. Raises ValueError if invalid."""
        # Validate date
        if not isinstance(self.birth_date, datetime):
            raise ValueError("Birth date must be a datetime object")
        # Validate time (HH:MM)
        if not re.match(r'^([01]?[0-9]|2[0-3]):[0-5][0-9]$', self.birth_time):
            raise ValueError("Birth time must be in HH:MM format")
        # Validate location
        if not self.birth_location or not isinstance(self.birth_location, str):
            raise ValueError("Birth location must be a non-empty string")
        # Validate gender
        if not isinstance(self.gender, Gender):
            raise ValueError("Gender must be a valid Gender enum value")
        return True

def parse_saju_input(data: Dict) -> SajuInput:
    """Parse dict to SajuInput, with validation."""
    birth_date = datetime.strptime(data["birth_date"], "%Y-%m-%d")
    birth_time = data["birth_time"]
    birth_location = data["birth_location"]
    gender = Gender(data["gender"].lower())
    saju = SajuInput(
        birth_date=birth_date,
        birth_time=birth_time,
        birth_location=birth_location,
        gender=gender
    )
    saju.validate()
    return saju