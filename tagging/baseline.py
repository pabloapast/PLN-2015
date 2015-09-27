from collections import defaultdict


class BaselineTagger:

    def __init__(self, tagged_sents):
        """
        tagged_sents -- training sentences, each one being a list of pairs.
        """
        self.counts = counts = defaultdict(defaultdict)
        self.frequent_tag = ''
        tag_count = defaultdict(int)

        # Count of tags for word
        for tagged_sent in tagged_sents:
            for word, tag in tagged_sent:
                try:
                    counts[word][tag] += 1
                except KeyError:
                    counts[word][tag] = 1
                tag_count[tag] += 1

        # For each word, order tags by ocurrence
        for word, tags in counts.items():
            counts[word] = sorted(tags.items(), key=lambda tup: tup[1],
                                  reverse=True)

        self.frequent_tag = max(tag_count.items(), key=lambda tup: tup[1])[0]

        # print(self.counts)

    def tag(self, sent):
        """Tag a sentence.

        sent -- the sentence.
        """
        return [self.tag_word(w) for w in sent]

    def tag_word(self, w):
        """Tag a word.

        w -- the word.
        """
        tag = ''
        if self.unknown(w):
            tag = self.frequent_tag
        else:
            tag = self.counts[w][0][0]
        return tag

    def unknown(self, w):
        """Check if a word is unknown for the model.

        w -- the word.
        """
        unkn = False
        tags = self.counts.get(w)

        if tags is None:
            unkn = True
        return unkn
