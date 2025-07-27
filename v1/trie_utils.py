class TrieNode:
    def __init__(self):
        self.children={}
        self.end_of_word=False
    
class Trie:
    def __init__(self):
        self.root=TrieNode()
    def insert(self,word:str)->None:
        cur=self.root
        for character in word:
            if character not in cur.children:
                cur.children[character]=TrieNode()
            cur=cur.children[character]
        cur.end_of_word=True
    def search(self,word:str)->None:
        cur=self.root
        for character in word:
            if character not in cur.children:
                return False
            cur=cur.children[character]
        return cur.end_of_word
    def starts_with(self,word:str)->None:
        cur=self.root
        flag=False
        for character in word:
            if character not in cur.children:
                return False
            cur=cur.children[character]
        return True
    

if __name__=="__main__":
    obj=Trie()
    obj.insert("hello")
    obj.insert("touhid")
    print(obj.search("touhid"))
    print(obj.search("max"))
    print(obj.starts_with("max"))
    