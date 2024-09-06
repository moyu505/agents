import re


def split_words(text: str, ignore_punctuation: bool = True) -> list[str]:
    # fmt: off
    punctuations = [".", ",", "!", "?", ";", ":", "'", '"', "(", ")", "[", "]", "{", "}", "<", ">",
                    "—"]
    # fmt: on

    if ignore_punctuation:
        for p in punctuations:
            # TODO(theomonnom): Ignore acronyms
            text = text.replace(p, "")

    # words = re.split("[ \n]+", text)
    words = re.split("([\n，。！？,.!?])", text)
    new_words = []
    for word in words:
        if not word:
            continue  # ignore empty
        if len(new_words) == 0:
            new_words.append(word)
        elif len(word) == 1:
            new_words[-1] += word
        else:
            new_words.append(word)
    # print(f'words===>{words} new_words===>{new_words}')
    return new_words
