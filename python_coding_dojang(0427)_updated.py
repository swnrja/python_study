#Ch5-------------------------------
m = 12
max = 0.2467 * m + 4.159
print(int(max))

print(102*0.6+225)

#ch6-------------------------------

a = int(input("첫번째 숫자"))
b = int(input("첫번째 숫자"))
c = int(input("첫번째 숫자"))

print("%d+%d+%d=%d" %(a,b,c,a+b+c))

a = 50
b = 100
c= None

print(a)
print(b)
print(c)

#ch7-------------------------------

year = 2000
month = 10
day = 27
hour = 11
minute = 43
second = 59

print(year, month, day, sep = "/")
print(hour, minute, second, sep = ":")



year, month, day, hour, minute, second = input().split("")
print(hour, minute, second, sep='-')
print(hour, minute, second, sep=':')

#ch8-------------------------------
korean = 92
english = 47
mathmatics = 86
science = 81
print(korean < 50 and english < 50and mathmatics < 50 and science < 50)


korean = int(input("국어점수는?"))
english = int(input("영어점수는?"))
mathmatics = int(input("수학점수는?"))


science = int(input("과학점수는?"))

korean >= 90 and english >80 and mathmatics >85 and science >80


#ch10-------------------------------
list(range(5,-10,-2))

i = int(input())
tuple(range(-10,11,i))

#ch11-------------------------------
year = [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018]
population = [10249679, 10195318, 10143645, 10103233, 10022181, 9930616, 9857426, 9838892]

print(year[4:-1])
print(population[4:-1])

x = input().split()
tuple(x)

#심사문제 ing

#ch12-------------------------------
camille = {
    'health': 575.6,
    'health_regen': 1.7,
    'mana': 338.8,
    'mana_regen': 1.63,
    'melee': 125,
    'attack_damage': 60,
    'attack_speed': 0.625,
    'armor': 26,
    'magic_resistance': 32.1,
    'movement_speed': 340
}
print(camille['health'])
print(camille['movement_speed'])

#심사문제 ing


#ch13-------------------------------
x = 5
if x!=10:
    print("ok")

coupon = int(input())
cash = int(input())
if coupon == 3000:
    cash - 3000
if coupon == 5000:
    cash - 5000

print("가격:%d" %(cash - coupon))


#ch14-------------------------------
written_test = 75
coding_test = True

if written_test >= 80 and coding_test == True:    
    print('합격')
else:
    print('불합격')


korean = int(input())
english = int(input())
mathmatics = int(input())
science = int(input())

if (korean + english + mathmatics + science)/4 >= 90:
    print("합격")
elif (korean + english + mathmatics + science)/4 >= 90:
    print("합격")
else:
    print("잘못된 점수")

    
#ch15-------------------------------

x = int(input())

if 11 <= x <= 20:
    print('11~20')
elif 21 <= x <= 30:
    print('21~30')
else:
    print('아무것도 해당하지 않음')


age = int(input())
balance = 9000    # 교통카드 잔액

if 7 <= age <= 12:
    balance = 9000 - 650
elif 13 <= age <= 18:
    balance = 9000 - 1050
else:
    balance = 9000 - 1250

print("잔액 = %d" %balance)

#ch16-------------------------------

x = [49, -17, 25, 102, 8, 62, 21]
for i in x:
    print(i * 10, end=' ')


i = int(input())

for j in range(1, 10):
    j*i
    print("%d*%d=%d" %(i,j,i*j))

#ch17-------------------------------
i = 2
j = 5

while i<=32 and j >=1:
    i *= 2
    j -= 1

######################헷갈림...다시 봐야 함.
amount = int(input())
while amount > 1350:

    amount = amount- 1350
    print(amount)

#ch18-------------------------------
i = 0
while True:
    if i % 10 != 3:
        i+=1
        continue
    if i>73:
        break
    print(i,end=" ")
    i+=1

start, stop = map(int,input().split())
i = start
while True:
    if i % 10 ==3:
        i+=1
        continue
    if i > stop:
        break
    
    print(i, end = " ")
    i +=1


#ch19-------------------------------
for i in range(0,5):
    print("\n")
    for j in range(0,5):
        if j < i:
            print(" ", end='')
        else:
            print("*",end='')
    

#산 모양으로 별 출력하기....?---------


#ch20-------------------------------
for i in range(0,101):
    if(i%2 ==0 and i%11 ==0):
        print("FizzBuzz")
    elif i%2 ==0:
        print("Fizz")
    elif i%11==0:
        print("Buzz")

#ch25-------------------------------딕셔너리 조작....ㅍvalue 입력 하려면 어떻게 ...?
keys = ['a','b','c','d']
x = {key:value for key , value in dict.fromkeys(keys).items()}

{key:2 for key in dict.fromkeys(['a','b','c','d']).keys()} #키만 가지고 옴
{value:1 for value in{'a':10,'b':20,'c':30,'d':40}.values()} #값을 키로 사용
{value: key for key, value in{'a':10,'b':20,'c':30,'d':40}.items()} #키-값 자리를 바꿈


maria = {'korean':94,'english':91,'mathmatics':89,'science':83}
print(sum(maria.values())/len(maria))

###?? - 30값에서 에러가 남. 
keys = input().split()
values = map(int, input().split())
x = dict(zip(keys,values))

x.pop('delta')
x.pop({value:30 for value in{values}.values()})

print(x)


#ch26

num1=int(input())
num2=int(input())

a = {i for i in range(1, num1+1) if num1%i==0}
b = {i for i in range(1, num2+1) if num2%i==0}



divisor = a&b  
result = 0
if type(divisor) ==set:
    result = sum(divisor)
print(result)


#ch27

#파일 여러 줄에 쓰기
with open('hello.txt','w') as file:
    for i in range(3):
        file.write('hello world{0}\n'.format(i))

#리스트에 있는 문자열을 파일에 쓰기
lines = ['안녕하세요.\n','파이썬\n','코딩도장입니다\n']

with open('hello.txt','w') as file:
    file.writelines(lines)

#파일의 내용을 한줄씩 읽기
with open('hello.txt','r') as file: 
    line = None #변수 line을 None으로 초기화
    while line!='':
        line = file.readline()
        print(line.strip('\n'))#파일에서 읽어온 문자열에서 \n을 삭제하여 출력

#readline으로 여러 줄로 파일을 읽을 때는 while반복문을 사용해야 함.
#왜냐하면 파일에 몇 줄이나 있는지 모르기 떄문임.
#while은 특정 조건을 만족 할 때까지 계속 반복하므로 계속 읽을 수 있음
#파일이 빌 때까지 반복해서 읽어서 변수 line에 저장.
#초기에 line을 None이 아닌 ''으로 초기화 하면 코드가 그냥 끝남. while문이 실행이 안 됨. 


#그래서 그냥 line으로 읽는 게 좋은듯?????
with open ('hello.txt','r') as file:
    for line in file:
        print(line.strip('\n'))


#pickling<->txt
#파이썬 객체를 파일에 저장하는 과정 - pickling
#파일에서 객체를 읽어오는 과정 - unpickling

#pickling
import pickle
name = 'james'
age = 17
address = '서울시 서초구 반포동'
score = {'korean':90,'english':95,'mathmatics':85,'science':82}

with open('james.p','wb') as file: #hello.txt파일을 바이너리 쓰기 모드로 열기
    pickle.dump(name,file)
    pickle.dump(age,file)
    pickle.dump(address,file)
    pickle.dump(score,file)
    
#unpickling
import pickle

with open('james.p','rb') as file: #바이너리 읽기 모드로 읽기
    name = pickle.load(file)
    age = pickle.load(file)
    address = pickle.load(file)
    score = pickle.load(file)
    
    print(name)
    print(age)
    print(address)
    print(score)


#ch28

#회문 - 양 끝 문자열이 같은 문자열

word = input('단어를 입력하세요: ')

is_palindrome = True            # 회문 판별 값을 저장할 변수, 초기값은 True
for i in range(len(word)//2):   # 0부터 문자열의 길이의 절반만큼 반복
    if word[i] != word[-1-i]:   # 왼쪽 문자와 오른쪽 문자를 비교하여 문자가 다르면
        is_palindrome = False   # 회문이 아님
        break                   # 따라서 같으면 회문임. 위의 알고리즘으로 판단한 다음. 종료
    
print(is_palindrome)

#N-gram(글자 단위)
text = 'hello'

for i in range(len(text)-1):         #2-gram이므로 문자열의 끝에서 한 글자 앞까지만 반복함
    print(text[i],text[i+1],sep=' ') #현재 문자와 그 다음 문자 출력


#N-gram(단어 단위)
text = "this is python script"
words = text.split()     #split으로 단어별로 리스트화

for i in range(len(words)-1):
    print(words[i],words[i+1])

    with open('hello3.txt','r') as file:
    while word!='':
        word = file.readline()
        line.strip('\n')
        if word[::-1] == word:
            print(True)
        
        #print(line.strip('\n'))


#ch29 - docstring
def sum(a,b):
    """이 함수는 sum 함수입니다."""
    return a+b

a = sum(1,2)
print(a)
print(sum.__doc__)



#값을 여러개 반환, 튜플로 반환되나 리스트로 변환할 수 있음.

def add_sub(a,b):
    return a+b, a-b
a = add_sub(10,20)


x,y=map(int,input().split())
def calc(x,y):
    return x+y, x-y, x/y, x*y
a,s,m,d = calc(x,y)
print("덧셈:{0},뺼셈:{1},곱셈:{2},나눗셈:{3}".format(a,s,m,d))


#ch30

#list unpacking
def pk(a,b,c):
    print(a)
    print(b)
    print(c)
    
x = [10,20,30]
print(*x)

korean, english, mathmatics, science = map(int,input().split())


#가장 낮은 점수, 높은 점수, 평균 점수 구하기...? 인풋으로 위치볁수 여러개 넣은 다음에
#그걸 받아서 쓰려고 했는데 안 나오네..
def get_min_max_score(*input):
    return max(*input), min(*input),sum(*input)/len(*input)
min_score, max_score = get_min_max_score(korean, english, mathematics, science)
average_score = get_average(korean=korean, english=english,
                            mathematics=mathematics, science=science)
print('낮은 점수: {0:.2f}, 높은 점수: {1:.2f}, 평균 점수: {2:.2f}'
      .format(min_score, max_score, average_score))
 
min_score, max_score = get_min_max_score(english, science)
average_score = get_average(english=english, science=science)
print('낮은 점수: {0:.2f}, 높은 점수: {1:.2f}, 평균 점수: {2:.2f}'
      .format(min_score, max_score, average_score))


#ch31

#31.1.1  재귀호출에 종료 조건 만들기 - 이렇게 코드 짜면 왜 안 되는 거임?
def hello(count):
    print("hello")
    hello()
    count = count -1
    if count ==0:
        return

hello(5)

def factorial(n):
    if n == 1:      # n이 1일 때
        return 1    # 1을 반환하고 재귀호출을 끝냄
    return n * factorial(n - 1)    # n과 factorial 함수에 n - 1을 넣어서 반환된 값을 곱함

print(factorial(5))


def is_palindrome(word):
    if len(word) < 2:
        return True
    if word[0] != word[-1]:
        return False
    return is_palindrome(word[1:-1])
        
print(is_palindrome('hello'))
print(is_palindrome('level'))

#때려 맞춤...;;;
def fibo(n):
    if n==0:
        return 0
    return n + fibo(n-1)

n = int(input())
print(fibo(n))

#ch32 lambda표현식의 사용
#왜 사용?: 함수의 인수 부분에 간단하게 함수를 만들기 위해서 사용
#대표적인 예가 map

def plus_ten(x):
    return x+10
list(map(plus_ten,[1,2,3]))

# -->
list(map(lambda x:x+10, [1,2,3]))

#근데 그냥 함수로 쓰는 게 가독성이 좋을 듯 - 매개변수로 쓸 때만 사용이 좋을 듯 함. 

files = ['font', '1.png', '10.jpg', '11.gif', '2.jpg', '3.png', 'table.xslx', 'spec.docx']
list(filter(lambda x: x.find('.jpg') != -1 or x.find('.png') != -1, files))


#심사 문제..?????????????
files = input().split()
list(map(lambda x:'{03d}.{1}'.format,files))