PI = 3.141592
class Math:
    def solv(self, r):
        return PI * (r ** 2) # 반지름을 계산하는 클래스! r ** 는 r 제곱을 의미
def sum(a,b):
    return a + b

if __name__ == "__main__":
    print(PI)
    a = Math()
    print(a.solv(2))
    print(sum(PI, 4.4))