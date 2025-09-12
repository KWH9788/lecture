import numpy as np

data = [1, 2, 3, 4]
arr = np.array(data)
print(arr)
print(type(arr))

# python list 보단 numpy ndarray 사용
# numpy c로 구현 = 속도가 빠르다

"""
    data = [1, 2, 3]

    result = []
    for i in data:
        result.append(i * 10)
        print("for result :",result)

    print("final result :",result)
"""

arr = np.array([1, 2, 3])
result = arr * 10
print("final result :", result)

# 로그 작업 필수
# 주석 작업 필수
# asdasd
