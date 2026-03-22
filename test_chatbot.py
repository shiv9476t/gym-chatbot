import requests
import uuid

BASE_URL = "http://localhost:8000"
SESSION_ID = str(uuid.uuid4())

def chat(question):
    res = requests.post(f"{BASE_URL}/chat", json={
        "question": question,
        "session_id": SESSION_ID
    })
    answer = res.json()["answer"]
    print(f"Q: {question}")
    print(f"A: {answer}")
    print()

# Test 1 — basic date awareness
chat("what day is it today?")

# Test 2 — today's classes
chat("what classes are on today?")

# Test 3 — follow-up with tomorrow
chat("what about tomorrow?")

# Test 4 — follow-up with specific day
chat("what about monday?")

# Test 5 — pricing
chat("how much is a membership?")

# Test 6 — follow-up pricing
chat("what does the premium one include?")

# Test 7 — cancellation
chat("can I cancel my membership?")

# Test 8 — out of scope
chat("do you have a basketball court?")