from datetime import datetime

def get_available_months(start_year: int, start_month: int, end_year: int, end_month: int):
    """
    Inclusive month range in YYYYMM string format.
    """
    months = []
    current = datetime(start_year, start_month, 1)
    end = datetime(end_year, end_month, 1)
    while current <= end:
        months.append(current.strftime("%Y%m"))
        if current.month == 12:
            current = datetime(current.year + 1, 1, 1)
        else:
            current = datetime(current.year, current.month + 1, 1)
    return months

__all__ = ["get_available_months"]
