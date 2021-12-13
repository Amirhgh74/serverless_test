import json 

def lambda_handler(event, context):



    dataID = event['queryStringParameters']['dataID']


    response = {}
    response["text"] = "selected dataset is " + dataID


    responseObject = {}
    responseObject["statusCode"] = 200
    responseObject["headers"] = {}
    responseObject["headers"]["Content-type"] = "application/json"
    responseObject["body"] = json.dumps(response)

    return responseObject

    
