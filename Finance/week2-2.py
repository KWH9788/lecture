import numpy as np

data2d = [          # 2차원 리스트
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]
# 2차원 배열 인덱싱 중요
print("1번째 행 :", data2d[0])    # 1번째 행
# 행단위 접근만 가능

arr = np.array(data2d)
print("1번째 열 :", arr[:, 0])    # 1번째 열
# [행 index, 열 index]
print("="*30)

data1 = [1, 2, 3, 4]
arr1 = np.array(data1)
print("data1 :", data1)
print("arr1 :", arr1)
print("type(arr1) :", type(arr1))
print("="*30)

"""
    transformer : Bert + GPT
    incoder : Bert
    decoder : GPT(AI)
"""

data2 = [
    [1, 2],
    [3, 4]
]
arr2 = np.array(data2)

print(arr2)
print(type(arr2))
print(type(arr2[0]))
print("="*30)

print("arr2.shape :", arr2.shape)   # (행, 열)
print("arr2.ndim :", arr2.ndim)    # 차원
print(arr2.dtype)   # 원소 타입
print("="*30)

print("arr1.shape :", arr1.shape)   # (4,) = (1, 4)
print("="*30)

data = [
    [1],
    [2],
    [3],
    [4]
]
c = np.array(data)
print(c.shape)
print("="*30)

print(np.zeros(3))
print(np.ones(3))
print("="*30)

size = (3, 4)
print(np.zeros(size))
print("="*30)
"""
    결측치(null) 때문에 사용
"""

print(np.arange(5))         # [0, 5)
print(np.arange(1, 5))      # [1, 5)
print(np.arange(1, 20, 2))  # [1, 20) d = 2
print("="*30)

ndarr1 = np.arange(6)           # (6,)
print(ndarr1)
ndarr2 = ndarr1.reshape(2, 3)   # (2, 3)
print(ndarr2)
print("="*30)

"""
boolean = 논리형
int = 정수형
float = 실수형
string = 문자형
메모리에서 데이터가 차지하는 사이즈를 줄이기 위해 존재 = 최적화를 위해
양자화
"""

arr = np.arange(4)
print("arr :", arr)
print("arr[0] :", arr[0])
print("="*30)

arr = np.arange(4).reshape(2, 2)
print("np.arange(4).reshape(2, 2) :\n", arr)
print(arr[0])
print("="*30)

print(arr[0][0])
print(arr[0, 0])
print("="*30)

# 슬라이싱 중요
arr = np.arange(4)
print("arr :", arr)
print("arr[:2] :", arr[:2])
print("arr[::] :", arr[::])
print("arr[1:3] :", arr[1:3])
print("="*30)

print(arr[1:2], arr[1])
print(type(arr[1:2]), type(arr[1]))
print("="*30)

arr = np.arange(20).reshape(4, 5)
print(arr[:2])
print("="*30)

result = []
for row in arr:
    row_01 = [row[0], row[1]]
    print("row_01 :", row_01)
    result.append(row_01)
    print("result :", result)

print(result)
print("="*30)

print(arr[:2, :2])
print("="*30)

print(arr[1:4, 2:5])
print(arr[1:, 2:])
print("="*30)

a = np.array( [1, 2, 3] )
b = np.array( [2, 3, 4] )
print("a :", a)
print("b :", b)
print("a + b :", a + b)
print("a * b :", a * b)
print("a % b :", a % b)
print("="*30)

print("a + 3 :", a + 3)
print("="*30)

"""
브로드 캐스팅 조건
1. 뒤축(열)의 길이가 같다
2. 어느 한 축의 길이가 1이다
"""

high = [92700, 92400, 92100, 94300, 92300]
low = [90000, 91100, 91700, 92100, 90900]

arr_high = np.array(high)
arr_low = np.array(low)
print("arr_high :", arr_high)
print("arr_low :", arr_low)

arr_diff = arr_high - arr_low
print("arr_diff :", arr_diff)
print("="*30)

arr_high_x3 = arr_high * 3
arr_low_x2 = arr_low * 2
print(arr_high_x3 + arr_low_x2)
print("="*30)

data = [
    [92700, 92400, 92100, 94300, 92300],
    [90000, 91100, 91700, 92100, 90900]
]
arr = np.array(data)
print(arr[0] * 3 + arr[1]  * 2)
weight = np.array([3, 2]).reshape(2, 1)     # 가중치
print((weight * arr))
print((weight * arr).sum(axis=0))           # axis = 0 : x축 axis = 1 : y축
print("="*30)

arr = np.array( [10, 20, 30] )
print(arr > 10)
print("="*30)

arr = np.array([10, 20, 30])
cond = [False, True, True]
print(arr[ cond ])
print("="*30)

arr = np.array([10, 20, 30])
cond = arr > 10
print(arr[ cond ])
print("="*30)

arr = np.array([10, 20, 30])
cond0 = arr > 10
print(f"cond0 : {cond0}")
cond1 = arr < 30
print(f"cond1 : {cond1}")
print(f"cond0 & cond1 : {cond0 & cond1}")
print(f"cond0 | cond1 : {cond0 | cond1}")
print(arr[ cond0 & cond1 ])
print("="*30)

arr = np.array([10, 20, 30])
arr = np.where( arr > 10, 1, 0)
print(arr)
print("="*30)

arr = np.array([10, 20, 30])
arr = np.where( arr > 10, arr+10, arr-10)
print(arr)
print("="*30)

arr = np.arange(8).reshape(4, 2)
print(arr)
print("="*30)

print(arr.sum(axis=0))
print(arr.sum(axis=1))
print("="*30)

print(f"np.random.randint(46, size=(2, 5)) :\n{np.random.randint(46, size=(2, 5))}")
print("="*30)

x = np.linspace(0, 10, 3)
print(x)
print("="*30)

a = np.arange(4)
b = np.arange(4, 8)
c = np.vstack([a, b])
print(a)
print(b)
print(f"np.vstack([a, b]) :\n{np.vstack([a, b])}")
print(f"np.hstack([a, b]) :\n{np.hstack([a, b])}")
print("="*30)