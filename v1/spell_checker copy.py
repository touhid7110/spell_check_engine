#note: pip install nltk
#note: pip install PyPDF2
import os
import nltk
import pickle
from trie_utils import Trie
from ngram_utils import NgramGenerator
from edit_distance_utils import EditDistance
import PyPDF2
import re
import logging
from typing import Set, Optional

# --- Production Best Practice: Setup logging ---
# In a real application, this would be configured globally.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def pickle_data_structure(data_structure,file_name):
    try:
        with open(file_name, 'wb') as f:
            pickle.dump(data_structure, f)
        print(f"Trie successfully pickled and saved to {file_name}")
        return "Success"
    except Exception as e:
        print(f"Error pickling Trie: {e}")
        return "Error"

def build_and_save_trie(corpus_path):
    """
    Extracts all unique words from a PDF in a robust, production-ready manner.
    And inserts it into TRIE

    Args:
        corpus_path (str): The path to the PDF file.

    Returns:
        Optional[Set[str]]: A set of lowercase words from the PDF, or None if
                             the file cannot be processed.
    """
    my_trie=Trie()
    my_ngram_generator=NgramGenerator()
    # A slightly more inclusive regex, but still has limitations.
    # For true unicode support, the 'regex' library is better than 're'.
    word_pattern = re.compile(r"\b[a-zA-Z']+\b")

    try:
        with open(corpus_path, 'rb') as file:
            try:
                reader = PyPDF2.PdfReader(file)

                # Handle encrypted PDFs
                if reader.is_encrypted:
                    logging.warning(f"Skipping encrypted PDF: {corpus_path}")
                    return None # Or attempt decryption if password is known

                for page_num, page in enumerate(reader.pages):
                    #print(page_num)
                    text = page.extract_text()
                    if not text:
                        logging.debug(f"No text found on page {page_num + 1} of {corpus_path}")
                        continue

                    for match in word_pattern.findall(text):
                        my_trie.insert(match.lower())
                        #print(match.lower())
                        my_ngram_generator.generate_n_gram_for_word(match.lower())
                trie_filename = r"trie\spell_checker\my_trie_dictionary.pkl"
                pickle_data_structure(my_trie,trie_filename)
                loaded_ngram_filename=r"trie\spell_checker\loaded_ngrams.pkl"
                pickle_data_structure(my_ngram_generator,loaded_ngram_filename)
                return trie_filename,loaded_ngram_filename
            except PyPDF2.errors.PdfReadError as e:
                logging.error(f"Failed to read PDF file '{corpus_path}'. It may be corrupted. Error: {e}")
                return None

    except FileNotFoundError:
        logging.error(f"PDF file not found at path: {corpus_path}")
        return None
    except Exception as e:
        # Catch-all for any other unexpected errors
        logging.critical(f"An unexpected error occurred while processing {corpus_path}: {e}", exc_info=True)
        return None

def load_data_structure(file_path):
    """Loads a Data Structure from a pickled file."""
    print(f"Loading Data Structure from {file_path}...")
    try:
        with open(file_path, 'rb') as f:
            loaded_data_structure = pickle.load(f)
        print("Data Structure loaded successfully.")
        return loaded_data_structure
    except FileNotFoundError:
        print(f"Error: Pickled Data Structure file not found at {file_path}")
        return None # Or raise an error
    except Exception as e:
        print(f"Error loading pickled Data Structure: {e}")
        return None # Or raise an error

def check_spelling(sentence: str, trie,indexed_n_grams) -> list:
    """
    Checks for misspelled words in a sentence using a Trie, preserving original
    word positions and handling contractions correctly.

    Args:
        sentence (str): The input sentence to check.
        trie: The pre-built Trie object containing the dictionary.

    Returns:
        list: A list of tuples, where each tuple is (word_index, misspelled_word).
    """
    spell_correction_map = []
    edit_distance_obj=EditDistance()
    # --- THIS IS THE KEY CHANGE ---
    # Use the SAME regex as when you built the trie. This includes apostrophes.
    word_pattern = re.compile(r"\b[a-zA-Z']+\b")
    
    # Use re.finditer to get both the word and its position (span).
    # We enumerate to get a simple word count/index.
    for i, match in enumerate(word_pattern.finditer(sentence)):
        word = match.group(0) # The actual matched string, e.g., "didn't"
        
        # Check the lowercase version of the word against the trie
        if not trie.search(word.lower()):
            potential_reccomendations=indexed_n_grams.get_candidate_words_only(word.lower(), relevance_threshold=0.5)
            # The word is not in the dictionary.
            # We use i + 1 for a 1-based index to match your desired output.
            spell_correction_map.append({"index":i + 1,"mispelled_word":word,"potential_reccomendations":list(potential_reccomendations)})
    
    spell_check_final_response=edit_distance_obj.calculate_edit_distance(spell_correction_map,6)
    return spell_check_final_response

if __name__=="__main__":
    import time
    start_time_perf = time.perf_counter()

    current_directory = os.getcwd()
    loaded_trie_file_name = r"trie\spell_checker\my_trie_dictionary.pkl"  # Replace with the path to your file
    loaded_ngram_file_name= r"trie\spell_checker\loaded_ngrams.pkl"
    
    if not os.path.exists(loaded_trie_file_name):
        corpus_path=r"trie\spell_checker\english_language_dictionary.pdf"
        print(f"Failure: The file '{loaded_trie_file_name}' does not exist.")
        response=build_and_save_trie(corpus_path)
        loaded_trie_file_name=response[0]
        loaded_ngram_file_name=response[1]

    my_trie=load_data_structure(loaded_trie_file_name)
    indexed_n_grams=load_data_structure(loaded_ngram_file_name)

    #list_of_misspelled_words=check_spelling("You merly adopted the dark. I was bon in it, molded by it. I didn't see the light untl I was already a man, by then it was nohing to me but blinding !",my_trie,indexed_n_grams)
    #print(list_of_misspelled_words)
    
    #list_of_misspelled_words=check_spelling("When the wprld was fallng apart,. You came saved my life turned it upside down in real sense you are my life",my_trie,indexed_n_grams)
    #print(list_of_misspelled_words)

    #list_of_misspelled_words=check_spelling("She recived the letter yesturday and was very happi.",my_trie,indexed_n_grams)

    #list_of_misspelled_words=check_spelling("The weathr forecast predictes rain tomorow morning",my_trie,indexed_n_grams)

    list_of_misspelled_words=check_spelling("I can’t beleive how quikly the sun sets in the evening.",my_trie,indexed_n_grams)

    import json
    file_name = "output.json"
    try:
        with open("output.json", "w") as json_file:
            json.dump(list_of_misspelled_words, json_file, indent=4)
        print("List of dictionaries successfully written to output.json")
    except IOError as e:
        print(f"Error writing to file: {e}")
    
    end_time_perf = time.perf_counter()
    elapsed_time_perf = end_time_perf - start_time_perf
    print(f"Elapsed time (perf_counter): {elapsed_time_perf}")
