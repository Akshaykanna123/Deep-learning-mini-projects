
responses = {
    "greeting": "Hello! How can I help you?",
    "weather": "I can't check live weather, but it might be sunny today.",
    "time": "You can check the time on your device.",
    "bye": "Goodbye! Have a nice day."
}

while True:
    user = input("You: ").lower()

    if "hello" in user or "hi" in user:
        print("Bot:", responses["greeting"])

    elif "weather" in user:
        print("Bot:", responses["weather"])

    elif "time" in user:
        print("Bot:", responses["time"])

    elif "bye" in user:
        print("Bot:", responses["bye"])
        break

    else:
        print("Bot: Sorry, I don't understand the question.")
