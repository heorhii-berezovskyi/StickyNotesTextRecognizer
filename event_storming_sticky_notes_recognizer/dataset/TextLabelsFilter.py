import argparse


class TextLabelsFilter:
    def __init__(self, text_labels_path: str, filter_length: int):
        self.path = text_labels_path
        self.length = filter_length

    def load_text_labels(self) -> list:
        with open(self.path, encoding='utf-8') as f:
            result = f.readlines()
        return result

    def filter_by_length(self, labels: list) -> list:
        stripped = [w.strip() for w in labels]
        filtered = filter(
            lambda x: (len(x) <= self.length and 'ё' not in x and '-' not in x and '.' not in x and ' ' not in x),
            stripped)
        return list(filtered)

    @staticmethod
    def save(labels: list, to: str):
        with open(to, 'w', encoding='utf-8') as f:
            for item in labels:
                f.write("%s\n" % item)


def run(args):
    labels_filter = TextLabelsFilter(text_labels_path=args.labels_to_filter,
                                     filter_length=args.filter_length)
    filtered_text_labels = labels_filter.filter_by_length(labels=labels_filter.load_text_labels())
    labels_filter.save(labels=filtered_text_labels, to=args.write_to)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Removes whitespaces and words which are longer than specified length.')
    parser.add_argument('--labels_to_filter', type=str, help='Path to labels to filter.',
                        default=r'D:\russian_words_corpus\russian_unicode.txt')

    parser.add_argument('--filter_length', type=int, help='Maximal valid word length.',
                        default=10)

    parser.add_argument('--write_to', type=str, help='Path to save the results.',
                        default=r'D:\russian_words_corpus\russian_unicode_filtered.txt')

    _args = parser.parse_args()

    run(_args)
