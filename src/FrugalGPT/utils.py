import json
import csv

def help():
    print("Welcome to use FrugalGPT!")
    print("FrugalGPT currently support the following methods")
    print("LLMCascade, and LLMforAll!")
    return 

def getservicename(configpath='config/serviceinfo.json'):
    service = json.load(open(configpath))
    names = [provider + "/" + name for provider in service.keys() for name in service[provider]]
    return names

def formatdata(data, prefix):
    for i in range(len(data)):
        data[i][0]=prefix+data[i][0]
    return data

def loadcsvdata(filename):
    # Initialize an empty list to store the data
    data = []

    # Open the file in read mode
    with open(filename, "r") as csvfile:
        reader = csv.reader(csvfile)
        # Iterate over each row in the CSV file
        for row in reader:
            # Append the row as a list to the data list
            data.append(row)
    return data        