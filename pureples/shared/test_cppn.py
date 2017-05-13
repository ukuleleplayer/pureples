import cPickle as pickle 
from pureples.es_hyperneat.es_hyperneat import find_pattern
from pureples.shared.visualize import draw_pattern

path_to_cppn = "es_hyperneat_xor_small_cppn.pkl"
with open(path_to_cppn, 'rb') as input:  # For now, path_to_cppn should match path in visualize.py, sorry.
    cppn = pickle.load(input)  
    pattern = find_pattern(cppn, (0.0, -1.0))
    draw_pattern(pattern)

