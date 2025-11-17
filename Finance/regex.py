"""
정규식(Regular Expression) 학습 가이드
Python re 모듈을 사용한 다양한 정규식 패턴과 예제
"""

import re

print("=" * 80)
print("1. 정규식 기본 - 문자열 검색")
print("=" * 80)

# re.search() - 첫 번째 매칭 찾기
text = "안녕하세요, 제 전화번호는 010-1234-5678입니다."
pattern = r'\d{3}-\d{4}-\d{4}'
match = re.search(pattern, text)
if match:
    print(f"전화번호 찾음: {match.group()}")
    print(f"위치: {match.start()} ~ {match.end()}")

print("\n" + "=" * 80)
print("2. 기본 메타 문자")
print("=" * 80)

# . (점) - 줄바꿈을 제외한 모든 문자
text = "cat, bat, hat, mat"
pattern = r'.at'
matches = re.findall(pattern, text)
print(f"'.at' 패턴: {matches}")  # ['cat', 'bat', 'hat', 'mat']

# ^ (캐럿) - 문자열의 시작
text = "Python is great"
pattern = r'^Python'
if re.search(pattern, text):
    print("문자열이 'Python'으로 시작합니다")

# $ (달러) - 문자열의 끝
text = "I love Python"
pattern = r'Python$'
if re.search(pattern, text):
    print("문자열이 'Python'으로 끝납니다")

print("\n" + "=" * 80)
print("3. 문자 클래스 [ ]")
print("=" * 80)

# [abc] - a, b, c 중 하나
text = "apple, banana, cherry, date"
pattern = r'[abc]'
matches = re.findall(pattern, text)
print(f"'[abc]' 패턴: {matches}")

# [a-z] - 소문자 알파벳
text = "Hello World 123"
pattern = r'[a-z]+'
matches = re.findall(pattern, text)
print(f"'[a-z]+' 패턴 (소문자만): {matches}")

# [^abc] - a, b, c가 아닌 문자
text = "abcdef"
pattern = r'[^abc]'
matches = re.findall(pattern, text)
print(f"'[^abc]' 패턴 (abc 제외): {matches}")

print("\n" + "=" * 80)
print("4. 특수 문자 클래스")
print("=" * 80)

text = "Price: $123.45, Date: 2024-01-15"

# \d - 숫자 [0-9]
digits = re.findall(r'\d+', text)
print(f"\\d+ (숫자): {digits}")

# \D - 숫자가 아닌 것
non_digits = re.findall(r'\D+', text)
print(f"\\D+ (비숫자): {non_digits}")

# \w - 문자, 숫자, 언더스코어 [a-zA-Z0-9_]
words = re.findall(r'\w+', text)
print(f"\\w+ (단어 문자): {words}")

# \W - \w가 아닌 것
non_words = re.findall(r'\W+', text)
print(f"\\W+ (비단어 문자): {non_words}")

# \s - 공백 문자 (스페이스, 탭, 줄바꿈)
text2 = "Hello\tWorld\nPython"
spaces = re.findall(r'\s+', text2)
print(f"\\s+ (공백): {repr(spaces)}")

print("\n" + "=" * 80)
print("5. 반복 수량자")
print("=" * 80)

# * - 0회 이상
text = "a aa aaa aaaa"
pattern = r'a*'
matches = re.findall(r'a+', text)  # a+ 사용 (1회 이상이 더 유용)
print(f"'a+' 패턴: {matches}")

# + - 1회 이상
text = "color colour"
pattern = r'colou?r'  # u가 0회 또는 1회
matches = re.findall(pattern, text)
print(f"'colou?r' 패턴: {matches}")

# {n} - 정확히 n번
text = "1234 12 123456"
pattern = r'\d{4}'
matches = re.findall(pattern, text)
print(f"'\\d{{4}}' 패턴 (4자리 숫자): {matches}")

# {n,m} - n번 이상 m번 이하
text = "1234 12 123456"
pattern = r'\d{2,4}'
matches = re.findall(pattern, text)
print(f"'\\d{{2,4}}' 패턴 (2~4자리): {matches}")

print("\n" + "=" * 80)
print("6. 그룹과 캡처")
print("=" * 80)

# () - 그룹화
text = "John: 010-1234-5678, Jane: 010-9876-5432"
pattern = r'(\w+): (\d{3}-\d{4}-\d{4})'
matches = re.findall(pattern, text)
for name, phone in matches:
    print(f"이름: {name}, 전화번호: {phone}")

# 명명된 그룹
text = "2024-01-15"
pattern = r'(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})'
match = re.search(pattern, text)
if match:
    print(f"연도: {match.group('year')}")
    print(f"월: {match.group('month')}")
    print(f"일: {match.group('day')}")

print("\n" + "=" * 80)
print("7. 실전 예제 - 이메일 검증")
print("=" * 80)

emails = [
    "user@example.com",
    "invalid.email",
    "test.user@domain.co.kr",
    "bad@",
    "good_email123@test-domain.com"
]

# 이메일 패턴 상세 설명:
# r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
#
# r'...'        = Raw string (백슬래시를 이스케이프 없이 사용)
# ^             = 문자열의 시작을 의미
# [a-zA-Z0-9._%+-]  = 문자 클래스: 다음 문자들 중 하나
#   a-z         = 소문자 a부터 z까지
#   A-Z         = 대문자 A부터 Z까지
#   0-9         = 숫자 0부터 9까지
#   .           = 점(마침표)
#   _           = 언더스코어
#   %           = 퍼센트
#   +           = 플러스
#   -           = 하이픈
# +             = 앞의 문자 클래스가 1회 이상 반복 (사용자명 부분)
# @             = 골뱅이 기호 (정확히 1개)
# [a-zA-Z0-9.-] = 문자 클래스: 영문자, 숫자, 점, 하이픈
# +             = 1회 이상 반복 (도메인명 부분)
# \.            = 이스케이프된 점 (실제 점 문자를 의미, 메타문자 아님)
# [a-zA-Z]      = 문자 클래스: 영문자만 (최상위 도메인)
# {2,}          = 2회 이상 반복 (com, kr, net 등)
# $             = 문자열의 끝을 의미

email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'

print("이메일 패턴 분석:")
print("  사용자명: [a-zA-Z0-9._%+-]+ → 영문/숫자/특수문자(._%+-) 1개 이상")
print("  @: 구분자")
print("  도메인: [a-zA-Z0-9.-]+ → 영문/숫자/점/하이픈 1개 이상")
print("  .: 점 구분자")
print("  TLD: [a-zA-Z]{2,} → 영문자 2개 이상\n")

for email in emails:
    if re.match(email_pattern, email):
        print(f"✓ 유효: {email}")
    else:
        print(f"✗ 무효: {email}")

print("\n" + "=" * 80)
print("8. 실전 예제 - 전화번호 검증")
print("=" * 80)

phones = [
    "010-1234-5678",
    "02-123-4567",
    "031-1234-5678",
    "010.1234.5678",
    "01012345678",
    "123-456"
]

# 다양한 형식 허용
phone_pattern = r'^(\d{2,3})[-.・]?(\d{3,4})[-.・]?(\d{4})$'

for phone in phones:
    match = re.match(phone_pattern, phone)
    if match:
        print(f"✓ 유효: {phone} -> {'-'.join(match.groups())}")
    else:
        print(f"✗ 무효: {phone}")

print("\n" + "=" * 80)
print("9. 문자열 치환 - re.sub()")
print("=" * 80)

# 기본 치환
text = "I have 3 apples and 5 oranges"
result = re.sub(r'\d+', 'many', text)
print(f"원본: {text}")
print(f"치환: {result}")

# 함수를 이용한 치환
def double_number(match):
    num = int(match.group())
    return str(num * 2)

text = "I have 3 apples and 5 oranges"
result = re.sub(r'\d+', double_number, text)
print(f"숫자 2배: {result}")

# 그룹 참조를 이용한 치환
text = "John Doe, Jane Smith"
result = re.sub(r'(\w+) (\w+)', r'\2, \1', text)
print(f"이름 순서 변경: {result}")

print("\n" + "=" * 80)
print("10. 문자열 분할 - re.split()")
print("=" * 80)

# 여러 구분자로 분할
text = "apple,banana;cherry:date|elderberry"
parts = re.split(r'[,;:|]', text)
print(f"분할 결과: {parts}")

# 공백으로 분할 (여러 개의 공백도 처리)
text = "one    two  three     four"
parts = re.split(r'\s+', text)
print(f"공백으로 분할: {parts}")

print("\n" + "=" * 80)
print("11. 실전 예제 - URL 파싱")
print("=" * 80)

url = "https://www.example.com:8080/path/to/page?name=value&key=data#section"
url_pattern = r'^(https?://)?([^/:]+):?(\d+)?(/[^?#]*)?(\?[^#]*)?(#.*)?$'

match = re.match(url_pattern, url)
if match:
    protocol, domain, port, path, query, fragment = match.groups()
    print(f"프로토콜: {protocol}")
    print(f"도메인: {domain}")
    print(f"포트: {port}")
    print(f"경로: {path}")
    print(f"쿼리: {query}")
    print(f"프래그먼트: {fragment}")

print("\n" + "=" * 80)
print("12. 실전 예제 - HTML 태그 제거")
print("=" * 80)

html = "<p>This is <b>bold</b> and <i>italic</i> text.</p>"
clean_text = re.sub(r'<[^>]+>', '', html)
print(f"원본 HTML: {html}")
print(f"태그 제거: {clean_text}")

print("\n" + "=" * 80)
print("13. 탐욕적(greedy) vs 비탐욕적(non-greedy) 매칭")
print("=" * 80)

text = "<div>First</div><div>Second</div>"

# 탐욕적 (기본)
greedy = re.findall(r'<div>.*</div>', text)
print(f"탐욕적 매칭: {greedy}")

# 비탐욕적 (? 사용)
non_greedy = re.findall(r'<div>.*?</div>', text)
print(f"비탐욕적 매칭: {non_greedy}")

print("\n" + "=" * 80)
print("14. 실전 예제 - 주민등록번호 마스킹")
print("=" * 80)

text = "주민번호: 901234-1234567, 연락처: 010-1234-5678"
# 뒷자리 마스킹
masked = re.sub(r'(\d{6})-(\d{7})', r'\1-*******', text)
print(f"마스킹 결과: {masked}")

print("\n" + "=" * 80)
print("15. 실전 예제 - 금액 포맷팅")
print("=" * 80)

def format_number(match):
    num = match.group()
    return "{:,}".format(int(num))

text = "가격은 1000000원이고, 할인가는 850000원입니다."
formatted = re.sub(r'\d+', format_number, text)
print(f"원본: {text}")
print(f"포맷: {formatted}")

print("\n" + "=" * 80)
print("16. Lookahead와 Lookbehind")
print("=" * 80)

# Positive Lookahead (?=...)
text = "password123"
# 숫자가 뒤따르는 단어만 찾기
matches = re.findall(r'\w+(?=\d)', text)
print(f"Positive Lookahead: {matches}")

# Negative Lookahead (?!...)
text = "cat dog bird"
# 'dog'가 아닌 단어 찾기
matches = re.findall(r'\b\w+\b(?! dog)', text)
print(f"Negative Lookahead: {matches}")

# Positive Lookbehind (?<=...)
text = "$100 €200 £300"
# 통화 기호 뒤의 숫자만 찾기
matches = re.findall(r'(?<=\$)\d+', text)
print(f"Positive Lookbehind ($ 뒤): {matches}")

print("\n" + "=" * 80)
print("17. 플래그(Flags) 사용")
print("=" * 80)

text = "Python is GREAT\nPython is FUN"

# re.IGNORECASE - 대소문자 무시
matches = re.findall(r'python', text, re.IGNORECASE)
print(f"대소문자 무시: {matches}")

# re.MULTILINE - 여러 줄 처리
matches = re.findall(r'^Python', text, re.MULTILINE)
print(f"멀티라인: {matches}")

# re.DOTALL - . 이 줄바꿈도 매칭
match = re.search(r'Python.*FUN', text, re.DOTALL)
if match:
    print(f"DOTALL: {match.group()}")

print("\n" + "=" * 80)
print("18. 정규식 컴파일 (성능 향상)")
print("=" * 80)

# 패턴을 미리 컴파일하면 반복 사용시 성능 향상
email_regex = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')

test_emails = ["user@example.com", "invalid", "test@domain.kr"]
for email in test_emails:
    if email_regex.match(email):
        print(f"✓ {email}")
    else:
        print(f"✗ {email}")

print("\n" + "=" * 80)
print("19. 실전 예제 - 로그 파싱")
print("=" * 80)

log = """
2024-01-15 10:30:45 ERROR: Database connection failed
2024-01-15 10:31:12 INFO: Retrying connection
2024-01-15 10:31:15 ERROR: Connection timeout
2024-01-15 10:32:00 SUCCESS: Connected to database
"""

log_pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) (\w+): (.+)'
matches = re.findall(log_pattern, log)

for timestamp, level, message in matches:
    if level == 'ERROR':
        print(f"🔴 [{timestamp}] {level}: {message}")
    elif level == 'SUCCESS':
        print(f"🟢 [{timestamp}] {level}: {message}")
    else:
        print(f"⚪ [{timestamp}] {level}: {message}")

print("\n" + "=" * 80)
print("20. 연습 문제")
print("=" * 80)

print("""
다음 패턴들을 정규식으로 작성해보세요:

1. 한국 우편번호 (5자리 숫자): 12345
2. 날짜 형식 (YYYY-MM-DD): 2024-01-15
3. 비밀번호 (8자 이상, 영문+숫자+특수문자 포함)
4. IPv4 주소: 192.168.0.1
5. 신용카드 번호 (4자리씩 구분): 1234-5678-9012-3456
6. 시간 형식 (HH:MM): 14:30
7. 파일 확장자 추출: example.txt -> txt
8. 해시태그 추출: #Python #RegEx
9. 가격 추출: ₩1,234,567
10. 16진수 색상 코드: #FF5733
""")

# 정답 예시
print("\n정답 예시:")
print("1. 우편번호: r'^\\d{5}$'")
print("2. 날짜: r'^\\d{4}-\\d{2}-\\d{2}$'")
print("3. 비밀번호: r'^(?=.*[A-Za-z])(?=.*\\d)(?=.*[@$!%*#?&])[A-Za-z\\d@$!%*#?&]{8,}$'")
print("4. IPv4: r'^((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\\.){3}(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$'")
print("5. 신용카드: r'^\\d{4}-\\d{4}-\\d{4}-\\d{4}$'")
print("6. 시간: r'^([01]?[0-9]|2[0-3]):[0-5][0-9]$'")
print("7. 확장자: r'\\.([^.]+)$'")
print("8. 해시태그: r'#\\w+'")
print("9. 가격: r'₩([0-9,]+)'")
print("10. 색상 코드: r'^#[0-9A-Fa-f]{6}$'")

print("\n" + "=" * 80)
print("학습 완료! 위 예제들을 수정하며 실험해보세요.")
print("=" * 80)

print("=" * 80)
ssn_tests = [
    "901234-1234567",   # 유효
    "9012341234567",    # 유효
    "901234-123456",    # 무효 (뒷자리 6자리)
    "90123-1234567",    # 무효 (앞자리 5자리)
    "901234--1234567",  # 무효 (구분자 2개)
    "abcdef-abcdefg"    # 무효 (숫자 아님)
]
pattern = r'\d{6}\W?\d{7}'
for text in ssn_tests:
    if re.match(pattern, text):
        print(f"{text}: True")
    else:
        print(f"{text}: False")
print("=" * 80)

print("=" * 80)
phone_tests = [
    "010-1234-5678",    # 유효
    "01012345678",      # 유효
    "010 1234 5678",    # 유효
    "010.1234.5678",    # 유효
    "010-123-4567",     # 무효 (중간 3자리)
    "010-12345-6789",   # 무효 (중간 5자리)
    "010-1234-567",     # 무효 (마지막 3자리)
    "010-ABCD-5678"     # 무효 (숫자 아님)
]
pattern = r'010\W?\d{4}\W?\d{4}'
for text in phone_tests:
    if re.match(pattern, text):
        print(f"{text}: True")
    else:
        print(f"{text}: False")
print("=" * 80)

print("=" * 80)
email_tests = [
    "user@example.com",        # 유효
    "user.name+tag@domain.co", # 유효
    "user_name@domain.com",    # 유효
    "user@domain",             # 무효 (TLD 없음)
    "user@domain.c",           # 무효 (TLD 1자리)
    "user@@domain.com",        # 무효 (@ 2개)
    "user@.com",               # 무효 (도메인 없음)
    "user@domain..com",        # 무효 (도메인에 점 2개)
    "user@domain.com.org"      # 유효
]
pattern = r'^[a-zA-Z0-9\W]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
for text in email_tests:
    if re.match(pattern, text):
        print(f"{text}: True")
    else:
        print(f"{text}: False")
print("=" * 80)