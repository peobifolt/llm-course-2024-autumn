import pandas as pd
import numpy as np


from minhash import MinHash

class MinHashLSH(MinHash):
    def __init__(self, num_permutations: int, num_buckets: int, threshold: float):
        self.num_permutations = num_permutations
        self.num_buckets = num_buckets
        self.threshold = threshold

    def get_buckets(self, minhash: np.array) -> np.array:
        '''
        Возвращает массив из бакетов, где каждый бакет представляет собой N строк матрицы сигнатур.
        '''
        step = len(minhash) // self.num_buckets
        result = []
        left = 0
        for i in range(min(self.num_buckets, len(minhash))):
            cur_len = step
            if (len(minhash) - left) % (self.num_buckets - i):
                cur_len += 1
            result.append(minhash[left:left + cur_len])
            left += cur_len
        return result

    def get_similar_candidates(self, buckets) -> list[tuple]:
        '''
        Находит потенциально похожих кандижатов.
        Кандидаты похожи, если полностью совпадают мин хеши хотя бы в одном из бакетов.
        Возвращает список из таплов индексов похожих документов.
        '''
        similar_candidates = set()
        count_docs = len(buckets[0][0])
        for bucket in buckets:
            for i in range(count_docs):
                for j in range(i + 1, count_docs):
                    if np.all(bucket[:, i] == bucket[:, j]):
                        similar_candidates.add((i, j))
        # similar_candidates &= set(self.get_similar_pairs(np.concat(buckets)))

        return list(similar_candidates)

    def run_minhash_lsh(self, corpus_of_texts: list[str]) -> list[tuple]:
        occurrence_matrix = self.get_occurrence_matrix(corpus_of_texts)
        minhash = self.get_minhash(occurrence_matrix)
        buckets = self.get_buckets(minhash)
        similar_candidates = self.get_similar_candidates(buckets)

        return set(similar_candidates)

