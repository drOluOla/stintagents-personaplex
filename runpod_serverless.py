import runpod

from presonaplex import runpod_handler


if __name__ == "__main__":
    runpod.serverless.start({"handler": runpod_handler})
