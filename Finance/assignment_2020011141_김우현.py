from pykrx.website import krx   # 한국거래소 API와 통신하는 모듈
import re                       # 정규표현식 처리
import datetime
from pandas import DataFrame

"""
    BPS | 주당순자산가치 (Book-value Per Share)
    PER | 주가수익비율 (Price Earnings Ratio)
    PBR	| 주가순자산비율 (Price Book-value Ratio)
    EPS	| 주당순이익 (Earnings Per Share)
    DIV	| 배당수익률 (Dividend Yield)
    DPS	| 주당배당금 (Dividend Per Share)
"""

# 2021/01/08(O), 2021-01-08(O), 20210108(O)
regex_yymmdd = re.compile(r"\d{4}[-/]?\d{2}[-/]?\d{2}")

# datetime 객체를 문자열로 변환
def datetime2string(dt, freq='d'):
    if freq.upper() == 'Y':
        return dt.strftime("%Y")        # "2021"
    elif freq.upper() == 'M':
        return dt.strftime("%Y%m")      # "202101"
    else:
        return dt.strftime("%Y%m%d")    # "20210108"
    
# 일별 데이터를 월별/연별 첫 번째 값으로 집계
def resample_ohlcv(df, freq, how):
    """
    :param df   : KRX OLCV format의 DataFrame
    :param freq : d - 일 / m - 월 / y - 년
    """
    if freq != "d" and len(df) > 0:
        if freq == "m":
            df = df.resample("M").apply(how)
        elif freq == "y":
            df = df.resample("Y").apply(how)
        else:
            print("choose a freq parameter in ('m', 'y', 'd')")
            raise RuntimeError
    return df

# *args: 전달되는 여러 개의 위치 인수를 하나의 튜플로 묶어서 받는다
# **kwargs: 키워드 인수를 딕셔너리 형태로 받는 가변 매개변수
def get_market_fundamental(*args, **kwargs):
    """종목의 PER/PBR/배당수익률 조회

    Args:
        특정 종목의 지정된 기간 PER/PBR/배당수익률 조회

        fromdate     (str           ): 조회 시작 일자 (YYYYMMDD)
        todate       (str           ): 조회 종료 일자 (YYYYMMDD)
        ticker       (str           ): 조회 종목 티커
        freq         (str , optional): d - 일 / m - 월 / y - 년
        name_display (bool, optional): 종목 이름 출력 여부 (True/False)

        특정 일자의 전종목 PER/PBR/배당수익률 조회

        date   (str           ): 조회 일자 (YYYYMMDD)
        market (str,  optional): 조회 시장 (KOSPI/KOSDAQ/KONEX/ALL)
        prev   (bool, optional): 휴일일 경우 이전/이후 영업일 선택
    """
    # 날짜 추출
    dates = list(filter(regex_yymmdd.match, [str(i) for i in args]))

    # 날짜가 2개이거나, fromdate와 todate가 있다면 특정 종목의 지정된 기간 PER/PBR/배당수익률 조회
    if len(dates) == 2 or ("fromdate" in kwargs and "todate" in kwargs):
        return get_market_fundamental_by_date(*args, **kwargs)
    
    # 날짜가 1개면 특정 일자의 전종목 PER/PBR/배당수익률 조회
    else:
        return get_market_fundamental_by_ticker(*args, **kwargs)


# 특정 종목의 지정된 기간 PER/PBR/배당수익률 조회
def get_market_fundamental_by_date(fromdate: str, todate: str, ticker: str, freq: str = "d",
                                  name_display: bool = False) -> DataFrame:
    """
    특정 종목의 지정된 기간 PER/PBR/배당수익률 조회
    Args:
        fromdate     (str           ): 조회 시작 일자 (YYYYMMDD)
        todate       (str           ): 조회 종료 일자 (YYYYMMDD)
        ticker       (str           ): 조회 종목 티커
        freq         (str , optional): d - 일 / m - 월 / y - 년 별 데이터로 집계
        name_display (bool, optional): 종목 이름 출력 여부 (True/False)
    """
    # datetime 객체를 문자열로 변환
    if isinstance(fromdate, datetime.datetime):
        fromdate = datetime2string(fromdate)
    if isinstance(todate, datetime.datetime):
        todate = datetime2string(todate)
    
    # 구분자 제거
    fromdate = fromdate.replace("-", "").replace("/", "")
    todate = todate.replace("-", "").replace("/", "")

    df = krx.get_market_fundamental_by_date(fromdate, todate, ticker)

    # 잘못된 티커, 휴장일일 경우 빈 데이터일 가능성 있음
    # 빈 데이터일 경우 에러 방지 차원에서 처리 중단
    if df.empty:
        return df

    # 종목명 표시
    if name_display:
        df.columns.name = krx.get_stock_name(str(ticker))
    
    # 해당 기간의 첫 번째 값(시가)을 사용
    how = {
        "BPS": "first",
        "PER": "first",
        "PBR": "first",
        "EPS": "first",
        "DIV": "first",
        "DPS": "first"
    }
    return resample_ohlcv(df, freq, how)

# 특정 일자의 전종목 PER/PBR/배당수익률 조회
def get_market_fundamental_by_ticker(date: str, market: str = "KOSPI", alternative: bool = False) \
        -> DataFrame:
    """특정 일자의 전종목 PER/PBR/배당수익률 조회

    Args:
        date        (str           ): 조회 일자 (YYYYMMDD)
        market      (str,  optional): 조회 시장 (KOSPI/KOSDAQ/KONEX/ALL)
        alternative (bool, optional): 휴일일 경우 이전 영업일 선택 여부
    """
    if isinstance(date, datetime.datetime):
        date = datetime2string(date)
    
    date = date.replace("-", "").replace("/", "")

    df = krx.get_market_fundamental_by_ticker(date, market)

    # 모든 값이 0이면 휴일로 판단
    holiday = (df[['BPS', 'PER', 'PBR', 'EPS', 'DIV', 'DPS']] == 0).all(axis=None)
    if holiday and alternative: # alternative(기본값: False)가 True면 이전 영업일 데이터 조회
        target_date = krx.get_nearest_business_day_in_a_week(date=date, prev=True)
        df = krx.get_market_fundamental_by_ticker(target_date, market)
    return df

print(get_market_fundamental("2021/01/08"))
print(get_market_fundamental("2021/01/08", "20210120", "095570"))