import json 

def lambda_handler(event, context):

    response = {}
    response["text"] = "this works"


    responseObject = {}
    responseObject["statusCode"] = 200
    responseObject["headers"] = {}
    responseObject["headers"]["Content-type"] = "application/json"
    responseObject["body"] = json.dumps(response)

    return responseObject

    
