from utils.summarizer import summarize_text

if __name__ == "__main__":

    text = """
    Artificial Intelligence is transforming the world.
    It is used in healthcare, finance, and education.
    AI helps automate tasks and improve efficiency.
    However, it also raises ethical concerns.
    """

    summary = summarize_text(text, 2)

    print("\n===== SUMMARY =====\n")
    print(summary)