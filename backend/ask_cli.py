import argparse
from .query import answer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ask questions about the indexed project")
    parser.add_argument("question", help="Your question")
    parser.add_argument("--collection", default="default", help="Collection name")
    args = parser.parse_args()

    try:
        result = answer(args.question, args.collection)
    except Exception as e:
        print(f"Fatal error while answering query: {e}")
    else:
        print(result)

