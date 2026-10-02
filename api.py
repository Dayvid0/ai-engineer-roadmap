from fastapi import FastAPI

app = FastAPI()

# A route: visiting "/" returns this
@app.get("/")
def home():
    return {"message": "API is running"}

# A route that takes input and returns a computed result
@app.get("/calculate")
def calculate(expression: str):
    try:
        result = eval(expression)
        return {"expression": expression, "result": result}
    except Exception as e:
        return {"error": str(e)}
