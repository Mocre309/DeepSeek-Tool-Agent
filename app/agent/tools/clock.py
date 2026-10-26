"""Clock tool — the model has no idea what "today" is without this."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from langchain_core.tools import tool

_DEFAULT_ZONE = "Asia/Shanghai"
_WEEKDAYS = {
    "Monday": "星期一",
    "Tuesday": "星期二",
    "Wednesday": "星期三",
    "Thursday": "星期四",
    "Friday": "星期五",
    "Saturday": "星期六",
    "Sunday": "星期日",
}


@tool
def current_time(timezone: str = _DEFAULT_ZONE) -> str:
    """Get the current date and time, plus the weekday, in an IANA timezone.

    Call this whenever the answer depends on "now" (today's date, elapsed days,
    year, weekday, ...) — you do not know the current date otherwise.

    Args:
        timezone: An IANA zone name such as ``"Asia/Shanghai"``, ``"UTC"``,
            ``"America/New_York"``, ``"Europe/London"``. Defaults to
            ``"Asia/Shanghai"``.
    """
    zone_name = (timezone or _DEFAULT_ZONE).strip() or _DEFAULT_ZONE
    try:
        tz = ZoneInfo(zone_name)
    except (ZoneInfoNotFoundError, ValueError):
        return (
            f"错误: 未知时区 {zone_name!r}。请使用 IANA 名称，"
            "例如 Asia/Shanghai、UTC、America/New_York、Europe/London、Asia/Tokyo。"
        )

    now = datetime.now(tz)
    weekday = _WEEKDAYS.get(now.strftime("%A"), now.strftime("%A"))
    return "\n".join(
        [
            f"时区: {zone_name}",
            f"日期: {now:%Y-%m-%d}",
            f"时间: {now:%H:%M:%S}",
            f"星期: {weekday}",
            f"ISO8601: {now.isoformat(timespec='seconds')}",
            f"UTC 偏移: {now.strftime('%z')}",
        ]
    )
