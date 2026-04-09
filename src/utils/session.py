"""Session time utilities."""
from datetime import datetime


def get_session(utc_hour: int) -> str:
    if 0 <= utc_hour < 8:
        return "Asia_Session"
    elif 8 <= utc_hour < 13:
        return "London_Open"
    elif 13 <= utc_hour < 17:
        return "London_NY_Overlap"
    elif 17 <= utc_hour < 22:
        return "NY_Session"
    else:
        return "Dead_Zone"


def get_session_from_timestamp(ts) -> str:
    if isinstance(ts, str):
        ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    return get_session(ts.hour)


def is_weekend(ts) -> bool:
    if isinstance(ts, str):
        ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    return ts.weekday() >= 5
