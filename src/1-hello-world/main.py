import modal

app = modal.App("evals-workshop-1-hello-world")


# Decorate your function to make it remotely runnable
@app.function()
def hello():
    greeting = "Hello from Modal!"
    print("Remote:", greeting)
    return greeting


# Call it from a local client
@app.local_entrypoint()
def main():
    greeting = hello.remote()
    print("Local:", greeting)
