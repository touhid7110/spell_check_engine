from typing import List, Generator, Dict, Set

class PotentialReccomendations:
    def __init__(self):
        self.filtered_reccomendation_list=[]


class EditDistance:
    def __init__(self):
        self.root=PotentialReccomendations()
    def edit_distance_generator_function(self,mispelled_word,potential_reccomendations)-> Generator[int, None, None]:
        for potential_candidate in potential_reccomendations:
            m=len(mispelled_word)
            n=len(potential_candidate)
            cache= [[0]*(n+1) for _ in range(m+1)]
            for i in range(m+1):
                cache[i][0]=i
            for j in range(n+1):
                cache[0][j]=j
            for i in range(1,m+1):
                for j in range(1,n+1):
                    if mispelled_word[i-1]==potential_candidate[j-1]:
                        cache[i][j]=cache[i-1][j-1]
                    else:
                        cache[i][j]=1+min(cache[i-1][j-1],cache[i-1][j],cache[i][j-1])
            yield {"potential_candidate":potential_candidate,"edit_distance":cache[m][n]}
    def calculate_edit_distance(self,spell_correction_map,top_k):
        curr=self.root
        
        for spell_fix_dict in spell_correction_map:
            reccomendation_list_with_edit_distance={}
    
            mispelled_word=spell_fix_dict['mispelled_word']
            potential_reccomendations=spell_fix_dict['potential_reccomendations']
            response=self.edit_distance_generator_function(mispelled_word,potential_reccomendations)

            sorted_response = sorted(response, key=lambda x: x['edit_distance'], reverse=False)
            reccomendation_list_with_edit_distance['index']=spell_fix_dict['index']
            reccomendation_list_with_edit_distance['mispelled_word']=spell_fix_dict['mispelled_word']
            reccomendation_list_with_edit_distance['potential_reccomendations']=sorted_response[:top_k]

            curr.filtered_reccomendation_list.append(reccomendation_list_with_edit_distance)
        return curr.filtered_reccomendation_list



# if __name__=="__main__":
#     obj=EditDistance()
#     check_list=[
#     {
#         "index": 3,
#         "mispelled_word": "wprld",
#         "potential_reccomendations": [
#             "worldwide",
#             "rld",
#             "worldview",
#             "otherworldly",
#             "worldly",
#             "worlds",
#             "underworld",
#             "elderly",
#             "worldcom",
#             "world",
#             "world's",
#             "properly",
#             "netherworld",
#             "worldviews"
#         ]
#     },
#     {
#         "index": 5,
#         "mispelled_word": "fallng",
#         "potential_reccomendations": [
#             "namecalling",
#             "stalling",
#             "nightfall",
#             "fagradalsfjall",
#             "fanatically",
#             "swallowing",
#             "enthralling",
#             "alleging",
#             "falloff",
#             "fallout",
#             "fallacious",
#             "pitfalls",
#             "fallujah",
#             "fallin",
#             "linguistically",
#             "falls",
#             "windfall",
#             "fatally",
#             "ballooning",
#             "stonewalling",
#             "appalling",
#             "falluja",
#             "pitfall",
#             "fall",
#             "fallopian",
#             "smallness",
#             "fallen",
#             "alluring",
#             "deadfalls",
#             "downfall",
#             "footballing",
#             "allowing",
#             "challenging",
#             "galloping",
#             "snowfall",
#             "fallow",
#             "calling",
#             "infallible",
#             "freefall",
#             "icefall",
#             "falling",
#             "challengers",
#             "rallying",
#             "challenge",
#             "challenger",
#             "falli",
#             "challenged",
#             "shortfall",
#             "recalling",
#             "farallones",
#             "waterfalls",
#             "jingalls",
#             "deadfall",
#             "waterfall",
#             "snowfalls",
#             "challenges"
#         ]
#     }
# ]
#     print(obj.calculate_edit_distance(check_list,10))