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