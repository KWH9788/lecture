import numpy as np

data2d = [          # 2차원 리스트
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

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