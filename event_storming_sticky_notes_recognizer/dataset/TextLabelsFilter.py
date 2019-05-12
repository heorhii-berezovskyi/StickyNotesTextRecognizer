import argparse


class TextLabelsFilter:
    def __init__(self, text_labels_path: str, filter_length: int):
        self.path = text_labels_path
        self.length = filter_length

    def load_text_labels(self) -> list:
        with open(self.path) as f:
            result = f.readlines()
        return result

    def filter_by_length(self, labels: list) -> list:
        stripped = [w.strip() for w in labels]
        filtered = filter(lambda x: (len(x) <= self.length), stripped)
        return list(filtered)

    @staticmethod
    def save(labels: list, to: str):
        with open(to, 'w') as f:
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
                        default=r'C:\Users\heorhii.berezovskyi\Documents\corpus\google-10000-english.txt')

    parser.add_argument('--filter_length', type=int, help='Maximal valid word length.',
                        default=16)

    parser.add_argument('--write_to', type=str, help='Path to save the results.',
                        default=r'C:\Users\heorhii.berezovskyi\Documents\words\words.txt')

    _args = parser.parse_args()

    run(_args)
