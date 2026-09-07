"""Text front-end for TARS."""

from brain import TARS

tars = TARS()
print("TARS is online. Type 'quit' to exit.\n")

while True:
    user_input = input("Luca: ")
    if user_input.lower().strip() == "quit":
        print("TARS: Powering down. Don't break anything without me.")
        break
    print(f"TARS: {tars.respond(user_input)}\n")
