import uvicorn


def main():
    uvicorn.run("weather_pipeline.api:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()