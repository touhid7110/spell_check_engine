import streamlit as st
import re
from spell_checker import check_spelling  # Your function
import json

from spell_checker import load_data_structure

# --- MOCKED STUBS for Trie and Indexed N-grams ---
# Replace the below mocks with your actual initialization!

loaded_trie_file_name = r"my_trie_dictionary.pkl"  # Replace with the path to your file
loaded_ngram_file_name= r"loaded_ngrams.pkl"
trie=load_data_structure(loaded_trie_file_name)
indexed_n_grams=load_data_structure(loaded_ngram_file_name)


def highlight_misspelled_words(sentence, misspelled_data):
    """
    Returns HTML string with misspelled words underlined and colored.
    """
    # Build a map {position_index: misspelled_word} from list
    misspelled_index_to_word = {item['index']: item['mispelled_word'] for item in misspelled_data}

    # Use the same regex your trie uses
    word_pattern = re.compile(r"\b[a-zA-Z']+\b")

    result_fragments = []
    word_count = 0

    last_pos = 0
    for match in word_pattern.finditer(sentence):
        word_count += 1
        start, end = match.span()
        word = match.group(0)
        # Append text before this word unchanged
        result_fragments.append(sentence[last_pos:start])
        # Check if word is misspelled (case-sensitive original from misspelled list)
        if word_count in misspelled_index_to_word:
            # Underline & color misspelled word (red with subtle underline)
            result_fragments.append(f'<span style="text-decoration: underline wavy red; color: crimson; font-weight: 600;" title="Potential mispelling">{word}</span>')
        else:
            # Normal word
            result_fragments.append(word)
        last_pos = end

    # Append trailing text after last word
    result_fragments.append(sentence[last_pos:])
    return "".join(result_fragments)


def display_recommendations(misspelled_data):
    """
    For each misspelled word, show UI with recommendations
    """
    for idx, item in enumerate(misspelled_data, 1):
        word = item['mispelled_word']
        recs = item['potential_reccomendations']

        with st.expander(f"#{idx} Word: '{word}' - Recommendations ({len(recs)})"):
            # Sort recommendations by edit_distance ascending just in case
            sorted_recs = sorted(recs, key=lambda x: x['edit_distance'])
            for rec in sorted_recs:
                candidate = rec['potential_candidate']
                dist = rec['edit_distance']
                st.markdown(f"- **{candidate}**  _(Edit Distance: {dist})_")

def main():
    st.set_page_config(page_title="Spell Checker", page_icon="🔤", layout="centered")
    st.title("📝 Advanced Spell Checker")
    st.write("""
    Enter your sentence below to check spelling.
    Misspelled words will be underlined in red. Click the expanders below
    to see suggested corrections with edit distances.
    """)

    sentence_input = st.text_area("Enter sentence here:", height=130, max_chars=1000)

    if st.button("Check Spelling") and sentence_input.strip():
        with st.spinner("Checking spelling..."):
            try:
                # Call your spell checker function
                result = check_spelling(sentence_input, trie, indexed_n_grams)
                # result is a list of dicts as per your format

                if not result:
                    st.success("No spelling errors detected! 🎉")
                    st.write(sentence_input)
                else:
                    # Show sentence with misspelled words highlighted
                    st.markdown("### Sentence with detected misspelled words:")
                    html_sentence = highlight_misspelled_words(sentence_input, result)

                    st.markdown(html_sentence, unsafe_allow_html=True)

                    st.markdown("---")
                    st.markdown("### Suggested Corrections:")
                    display_recommendations(result)

            except Exception as e:
                st.error(f"An error occurred during spell checking: {e}")

    st.markdown("---")
    st.caption("Spell Checker UI powered by Streamlit | Backend: Trie + Edit Distance + N-grams")

if __name__ == "__main__":
    main()
