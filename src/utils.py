from datetime import datetime
import pytz

def convert_to_timezone(dt: datetime, timezone_str: str) -> datetime:
    """Convert naive datetime to a specific timezone-aware datetime."""
    tz = pytz.timezone(timezone_str)
    return tz.localize(dt)