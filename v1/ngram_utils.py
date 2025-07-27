from typing import List, Generator, Dict, Set
from collections import Counter


class NgramLookup:
    def __init__(self):
        self.n_gram_lookup = {}


class NgramGenerator:
    def __init__(self):
        self.root = NgramLookup()
    
    def n_gram_generator(self, word: str) -> Generator[str, None, None]:
        for i in range(len(word) - 1):
            bi_gram = word[i:i+2]
            yield bi_gram
    
    def generate_n_gram_for_word(self, word: str) -> None:
        curr = self.root
        n_grams_generator_obj = self.n_gram_generator(word)
        n_gram_list = list(n_grams_generator_obj)
        
        for n_gram in n_gram_list:
            if n_gram not in curr.n_gram_lookup:
                curr.n_gram_lookup[n_gram] = set()  # Change: use set()
            
            curr.n_gram_lookup[n_gram].add(word)  # Change: use add()
    
    def get_reccomendations_for_spell_check(self, word: str, relevance_threshold: float = 0.3, max_candidates: int = 100) -> List[tuple]:
        """
        Get spell check recommendations using n-gram relevance scoring.
        
        Args:
            word: The misspelled word
            relevance_threshold: Minimum relevance score (0.0 to 1.0)
            max_candidates: Maximum number of candidates to return
            
        Returns:
            List of tuples: [(candidate_word, relevance_score), ...]
            Sorted by relevance score in descending order
        """
        curr = self.root
        misspelled_ngrams = list(self.n_gram_generator(word))
        
        # If no n-grams generated, return empty list
        if not misspelled_ngrams:
            return []
        
        total_ngrams_in_misspelled = len(misspelled_ngrams)
        
        # Dictionary to track n-gram overlap counts for each candidate
        candidate_overlap_counts = Counter()
        
        # Collect overlap counts for each n-gram
        for n_gram in misspelled_ngrams:
            if n_gram in curr.n_gram_lookup:
                candidate_words = curr.n_gram_lookup[n_gram]
                for candidate_word in candidate_words:
                    candidate_overlap_counts[candidate_word] += 1
        
        # Calculate relevance scores and filter
        scored_candidates = []
        
        for candidate_word, overlap_count in candidate_overlap_counts.items():
            # Calculate overlap ratio (shared n-grams / total n-grams in misspelled word)
            relevance_score = overlap_count / total_ngrams_in_misspelled
            
            # Apply threshold filtering
            if relevance_score >= relevance_threshold:
                scored_candidates.append((candidate_word, relevance_score))
        
        # Sort by relevance score in descending order
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Return top candidates up to max_candidates limit
        return scored_candidates[:max_candidates]
    
    def get_candidate_words_only(self, word: str, relevance_threshold: float = 0.3, max_candidates: int = 100) -> Set[str]:
        """
        Convenience method to get only candidate words without scores.
        
        Args:
            word: The misspelled word
            relevance_threshold: Minimum relevance score (0.0 to 1.0)
            max_candidates: Maximum number of candidates to return
            
        Returns:
            Set of candidate words
        """
        scored_candidates = self.get_reccomendations_for_spell_check(word, relevance_threshold, max_candidates)
        return {candidate for candidate, score in scored_candidates}
    
    def analyze_ngram_distribution(self, word: str) -> Dict[str, int]:
        """
        Debug method to analyze n-gram overlap distribution.
        
        Args:
            word: The misspelled word
            
        Returns:
            Dictionary mapping candidate words to their overlap counts
        """
        curr = self.root
        misspelled_ngrams = list(self.n_gram_generator(word))
        candidate_overlap_counts = Counter()
        
        for n_gram in misspelled_ngrams:
            if n_gram in curr.n_gram_lookup:
                candidate_words = curr.n_gram_lookup[n_gram]
                for candidate_word in candidate_words:
                    candidate_overlap_counts[candidate_word] += 1
        
        return dict(candidate_overlap_counts)


