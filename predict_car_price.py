import os
import json

mileage = input("Enter the mileage of the car: ")

if (not mileage.isdigit() or int(mileage) < 0):
    print("Please enter a valid mileage")
    exit()

if (not os.path.exists("params.json")):
    print("Missing params.json file")
    exit()

try:
    with open("params.json", "r") as file:
        params = json.load(file)
except Exception as e:
    print("Error with JSON : ", e)
    exit()

theta0 = 0
theta1 = 0
try:
    theta0 = params["theta0"]
    theta1 = params["theta1"]
    price = theta0 + theta1 * int(mileage)
except Exception as e:
    print("Missing value in JSON : ", e)
    exit()

print("The estimated price of the car is: " + str(price))
