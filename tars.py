"""Text front-end for TARS."""

import argparse

from brain import SUPPORTED_MODELS, TARS


def main():
    parser = argparse.ArgumentParser(description="Chat with TARS in the terminal.")
    parser.add_argument("--brain-model", choices=SUPPORTED_MODELS)
    args = parser.parse_args()

    tars = TARS(model=args.brain_model)
    tars.enable_memory_notes()
    print(f"TARS is online with {tars.model}. Type 'quit' to exit.\n")

    try:
        while True:
            user_input = input("Luca: ")
            if user_input.lower().strip() == "quit":
                print("TARS: Powering down. Don't break anything without me.")
                break
            print(f"TARS: {tars.respond(user_input)}\n")
            tars.submit_note()
    finally:
        tars.close()


if __name__ == "__main__":
    main()
